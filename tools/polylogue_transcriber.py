"""
title: Audio Transcriber
author: Sergei Vyaznikov
description: Переводит аудиофайл в текст с разбиением на спикеров. Ключевые слова: транскрибация, распознавание речи, аудиофайлы
version: 0.5
requirements: pyannote.audio==3.4.0
"""

import os
import aiohttp
import torch
import torchaudio
import logging
import asyncio
from io import BytesIO
from typing import Any, Literal, List, Tuple, Callable
from pydantic import BaseModel, Field, field_validator, computed_field
import pyannote
from collections import defaultdict
from pyannote.core import Segment, Annotation

from pyannote.audio import Pipeline as PyannotePipeline
from transformers import pipeline as TransformersPipeline
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

torch.serialization.add_safe_globals(
    [
        pyannote.audio.core.task.Specifications,
        pyannote.audio.core.task.Problem,
        pyannote.audio.core.task.Resolution,
        torch.torch_version.TorchVersion,
    ]
)


class SpeakerSegment(BaseModel):
    speaker: str
    text: str
    start: float
    end: float

    @computed_field
    @property
    def ru_speaker(self) -> str:
        return self.speaker.replace("SPEAKER_", "Спикер ")


class EventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Any] = None):
        self.event_emitter = event_emitter

    async def status(
        self,
        description: str = "Unknown State",
        status: str = "in_progress",
        done: bool = False,
    ) -> None:
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": status,
                        "description": description,
                        "done": done,
                    },
                }
            )

    async def message(self, message: str) -> None:
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "message",
                    "data": {"content": message},
                }
            )


async def async_diarization(pyannote_pipeline, waveform, sample_rate):
    """
    Asynchronous wrapper around the synchronous pyannote_pipeline call.
    Runs the diarization process in a thread pool to avoid blocking the event loop.

    :param pyannote_pipeline: The pyannote pipeline instance.
    :param waveform: The waveform tensor.
    :param sample_rate: The sample rate of the audio.
    :return: The result of pyannote_pipeline.
    """
    loop = asyncio.get_running_loop()

    # Define a synchronous callable for the executor
    def sync_diarization():
        return pyannote_pipeline({"waveform": waveform, "sample_rate": sample_rate})

    # Run the sync function in a thread pool
    return await loop.run_in_executor(None, sync_diarization)


async def async_transcription(faster_whisper_model, waveform):
    """
    Asynchronous wrapper around the synchronous faster_whisper_model.transcribe call.
    Runs the transcription process in a thread pool to avoid blocking the event loop.

    :param faster_whisper_model: The faster-whisper model instance.
    :param waveform: The waveform tensor.
    :return: The result of faster_whisper_model.transcribe (a tuple: transcription_result, transcription_info).
    """
    loop = asyncio.get_running_loop()

    # Define a synchronous callable for the executor
    def sync_transcription():
        return faster_whisper_model.transcribe(waveform[0].numpy())

    # Run the sync function in a thread pool
    return await loop.run_in_executor(None, sync_transcription)


def match_speakers_to_transcripts(diarization_iterable, transcription_segments):
    """
    Matches speakers from pyannote diarization to Whisper transcription segments.

    Args:
    - diarization_iterable: Iterable of (segment, speaker) where segment has .start and .end (floats in seconds).
    - transcription_segments: List of Whisper segments with .start, .end, .text.

    Returns:
    - list[SpeakerSegment]: List of SpeakerSegment objects.
    """
    result = []
    for trans in transcription_segments:
        trans_start = trans.start
        trans_end = trans.end
        speaker_durations = defaultdict(float)
        for dia_seg, _, speaker in diarization_iterable.itertracks(yield_label=True):
            overlap_start = max(trans_start, dia_seg.start)
            overlap_end = min(trans_end, dia_seg.end)
            if overlap_start < overlap_end:
                duration = overlap_end - overlap_start
                speaker_durations[speaker] += duration

        if speaker_durations:
            # Select the speaker with the maximum overlap duration
            max_speaker = max(speaker_durations, key=speaker_durations.get)
            result.append(
                SpeakerSegment(
                    speaker=max_speaker,
                    text=trans.text.strip(),
                    start=trans_start,
                    end=trans_end,
                )
            )
        else:
            # Fallback if no overlap (rare, but possible)
            result.append(
                SpeakerSegment(
                    speaker="Неопределено",
                    text=trans.text.strip(),
                    start=trans_start,
                    end=trans_end,
                )
            )

    return result


def get_waveform_from_bytes(
    audio_bytes: BytesIO, sample_rate: int = 16000
) -> Tuple[torch.Tensor, int]:
    waveform, original_sample_rate = torchaudio.load(audio_bytes)

    if original_sample_rate != sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=original_sample_rate, new_freq=sample_rate
        )
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    return waveform, sample_rate


def get_audio_segment(
    segment: Segment, waveform: torch.Tensor, sample_rate: int = 16000
) -> torch.Tensor:
    """Получает фрагмент из аудиопотока"""
    return waveform[0][
        int(segment.start * sample_rate) : int(segment.end * sample_rate)
    ]


def merge_segments_by_speakers(
    annotation: Annotation,
    pause_threshold: float = 5.0,
    duration_threshold: float = 0.3,
) -> List[Tuple[Segment, int, str]]:
    """Объединяет аудиосегменты по спикерам

    :param annotation: Объект с распознанными спикерами
    :type annotation: pyannote.core.Annotation

    :param pause_threshold: Максимальная длина паузы внутри одного сегмента (в секундах)
    :type pause_threshold: float

    :param duration_threshold: Минимальная длительность одного фрагмента (в секундах)
    :type duration_threshold: float

    :return: Объединённые по спикерам фрагменты
    :rtype: List[Tuple[str, Segment]]
    """

    if not isinstance(annotation, Annotation):
        raise TypeError("Expected 'annotation' to be an Annotation object")

    if pause_threshold < 0:
        raise ValueError("pause_threshold cannot be negative")

    segments = list()
    last_speaker = None
    last_segment = None
    cur_idx = 0
    for cur_segment, _, cur_speaker in annotation.itertracks(yield_label=True):

        if last_segment is None:
            last_segment = cur_segment
            last_speaker = cur_speaker
            continue

        if (
            cur_speaker != last_speaker
            or (cur_segment.start - last_segment.end) > pause_threshold
        ):
            segments.append((last_segment, cur_idx, last_speaker))
            cur_idx += 1
            last_segment = cur_segment
            last_speaker = cur_speaker
            continue

        last_segment = last_segment | cur_segment
        last_speaker = cur_speaker

    if last_speaker is not None:
        segments.append((last_segment, cur_idx, last_speaker))

    # Возвращаем только фрагменты длины большей, чем duration_threshold
    segments = list(filter(lambda x: x[0].duration >= duration_threshold, segments))
    return segments


async def get_file_content(auth_data: str, file_id: str, valves: dict) -> BytesIO:
    headers = {"Authorization": auth_data}
    download_url = f"{valves.INTERNAL_URL}/api/v1/files/{file_id}/content"

    async with aiohttp.ClientSession() as session:
        async with session.get(download_url, headers=headers) as response:
            if response.status == 200:
                content = await response.read()
                file_content = BytesIO(content)
            else:
                raise Exception(f"Failed to fetch file: {response.status}")
    return file_content


def get_faster_whisper_model(valves: dict) -> WhisperModel:
    whisper_model_index = os.environ.get("WHISPER_MODEL", "base")
    faster_whisper_model = WhisperModel(
        whisper_model_index,
        download_root="/app/backend/data/cache/whisper/models",
        device=valves.DEVICE,
    )
    return faster_whisper_model


def get_user_auth_data(request) -> str:
    """Получить jwt-токен текущего пользователя"""
    auth_data = request.headers["authorization"]
    return auth_data


class Tools:

    class Valves(BaseModel):

        DEBUG_MODE: bool = Field(default=False, description="Режим отладки")

        INTERNAL_URL: str = Field(
            default="http://127.0.0.1:8080",
            description="Домен для доступа к OpenWebUI изнутри контейнера",
        )

        PYANNOTE_DIR: str = Field(
            default="/app/backend/data/cache/pyannote",
            # default="/app/backend/data/cache/embedding/models",
            description="Директория со снапшотами моделей pyannote",
        )
        DEVICE: Literal["cuda", "cpu"] = Field(
            default="cuda" if torch.cuda.is_available() else "cpu",
            description="Устройство для инференса моделей",
        )

        @field_validator("DEVICE")
        @classmethod
        def restrict_device(cls, value: str) -> str:
            allowed_devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
            if value not in allowed_devices:
                raise ValueError(f"Доступные устройства: {allowed_devices}")
            return value

    def __init__(self):
        self.valves = self.Valves()

    async def transcribe_audio(
        self,
        __files__=None,
        __request__=None,
        __event_emitter__=None,
    ) -> str:
        """
        Инструмент для транскрибации аудио

        Используется для перевода аудиофайлов в текст с разбиением на спикеров

        :return: Статус транскрибации аудиофайла.
        :rtype: str.
        """
        try:
            emitter = EventEmitter(__event_emitter__)
            if not __files__ or len(__files__) <= 0:
                return "Файлы не были обнаружены. Ваша задача заключается в том, чтобы уведомить пользователя об ошибке и предложить приложить файл к сообщению."

            await emitter.status("Извлечение данных из аудиофайла...", done=False)

            file_data = __files__[-1].get("file")
            file_id = file_data.get("id")

            auth_data = __request__.headers["authorization"]

            audio_bytes = await get_file_content(
                auth_data=auth_data, file_id=file_id, valves=self.valves
            )

            waveform, sample_rate = get_waveform_from_bytes(audio_bytes)

            pyannote_pipeline = PyannotePipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                # use_auth_token=hf_token
            )
            pyannote_pipeline.to(torch.device(self.valves.DEVICE))

            # whisper_pipeline = get_whisper_model()
            faster_whisper_model = get_faster_whisper_model(valves=self.valves)

            await emitter.status("Обработка аудио...", done=False)

            # Create tasks for parallel execution
            diarization_task = asyncio.create_task(
                async_diarization(pyannote_pipeline, waveform, sample_rate)
            )
            transcription_task = asyncio.create_task(
                async_transcription(faster_whisper_model, waveform)
            )

            # Await both tasks in parallel using gather
            results = await asyncio.gather(diarization_task, transcription_task)

            # Unpack the results
            diarization_result = results[0]
            transcription_result, transcription_info = results[1]
            transcription_result = list(transcription_result)

            await emitter.status("Сопоставление спикеров...", done=False)

            speaker_segments = match_speakers_to_transcripts(
                diarization_result, transcription_result
            )

            await emitter.message(
                f"\n<details>\n\n<summary>Результат транскрибации</summary>\n"
            )

            for segment in speaker_segments:
                speaker = segment.ru_speaker
                text = segment.text
                await emitter.message(f"{speaker}: {text}\n")

            await emitter.message(f"\n</details>\n")

            await emitter.status(f"Речевые фрагменты распознаны!", done=True)

            await __event_emitter__(
                {
                    "type": "notification",
                    "data": {
                        "type": "info",  # "success", "warning", "error"
                        "content": "Транскрибация успешно завершена!",
                    },
                }
            )

            response = "Транскрибация для указанного файла была проведена. Транскрипция была отправлена пользователю. \
            Ваша задача заключается в том, чтобы уведомить пользователя о выполнении его запроса. \
            Сообщите, что посмотреть транскрипцию можно во вкладке 'Результат транскрибации'."
            return response

        except Exception as e:
            await emitter.status(f"Ошибка при транскрибации: {e}", done=True)
            return f"Ошибка при транскрибации: {e}"
