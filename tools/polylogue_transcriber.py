"""
title: Audio Transcriber
author: Sergei Vyaznikov
description: Переводит аудиофайл в текст с разбиением на спикеров. Ключевые слова: транскрибация, распознавание речи, аудиофайлы
version: 0.4
requirements: pyannote.audio==3.4.0
"""

import os
import aiohttp
import torch
import torchaudio
from io import BytesIO
from typing import Any, Literal, List, Tuple
from pydantic import BaseModel, Field, field_validator
from pyannote.core import Segment, Annotation

from pyannote.audio import Pipeline as PyannotePipeline
from transformers import pipeline as TransformersPipeline
from faster_whisper import WhisperModel


def get_waveform_from_bytes(
    audio_bytes: BytesIO, sample_rate: int = 22050
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
    segment: Segment, waveform: torch.Tensor, sample_rate: int = 22050
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

            if not __files__ or len(__files__) <= 0:
                return "Файлы не были обнаружены. Ваша задача заключается в том, чтобы уведомить пользователя об ошибке и предложить приложить файл к сообщению."

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Извлечение данных из аудиофайла...",
                        "done": False,
                        "hidden": False,
                    },
                }
            )

            file_data = __files__[-1].get("file")
            file_id = file_data.get("id")

            auth_data = __request__.headers["authorization"]

            audio_bytes = await get_file_content(
                auth_data=auth_data, file_id=file_id, valves=self.valves
            )

            waveform, sample_rate = get_waveform_from_bytes(audio_bytes)

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Данные извлечены!",
                        "done": True,
                        "hidden": False,
                    },
                }
            )

            pyannote_pipeline = PyannotePipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                # use_auth_token=hf_token
            )
            pyannote_pipeline.to(torch.device(self.valves.DEVICE))

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Разделение аудио на спикеров...",
                        "done": False,
                        "hidden": False,
                    },
                }
            )

            diarization_result = pyannote_pipeline(
                {"waveform": waveform, "sample_rate": sample_rate},
                # max_speakers=2
            )

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Спикеры определены!",
                        "done": True,
                        "hidden": False,
                    },
                }
            )

            merged_segments = merge_segments_by_speakers(
                diarization_result, pause_threshold=2.0, duration_threshold=0.0
            )

            # whisper_pipeline = get_whisper_model()
            faster_whisper_model = get_faster_whisper_model(valves=self.valves)

            await __event_emitter__(
                {
                    "type": "message",  # or simply "message"
                    "data": {
                        "content": f"\n\n<details>\n\n<summary>Результат транскрибации</summary>\n\n"
                    },
                }
            )

            for idx, (
                segment,
                _,
                speaker,
            ) in enumerate(
                merged_segments
            ):  # diarization_result.itertracks(yield_label=True):

                speaker_ru = speaker.replace("SPEAKER_", "Спикер ")

                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Распознавание речи... ({100 * ((idx + 1) / len(merged_segments)):.2f}%)",
                            "done": False,
                            "hidden": False,
                        },
                    }
                )
                try:
                    audio_segment = get_audio_segment(segment, waveform)
                    segments_generator, transcription_info = (
                        faster_whisper_model.transcribe(audio_segment.numpy())
                    )
                    recognized_segments = list(segments_generator)
                    # recognized_text = str(recognized_segments[0].__dir__())
                    recognized_text = " ".join(
                        [fragment.text for fragment in recognized_segments]
                    )

                    # recognition_result = whisper_pipeline(audio_segment, return_timestamps=True)
                    # recognized_text = recognition_result  # recognition_result.get("text")
                    # segments_generator, transcription_info  = recognition_result  # recognition_result.get("text")

                    await __event_emitter__(
                        {
                            "type": "message",
                            "data": {
                                "content": f"\n\n**{speaker_ru}**: {recognized_text}\n\n"
                            },
                        }
                    )
                except Exception as e:
                    await __event_emitter__(
                        {
                            "type": "message",
                            "data": {
                                "content": f"\n\n**{speaker_ru}**: Текст распознан с ошибкой: {e}\n\n"
                            },
                        }
                    )

            await __event_emitter__(
                {
                    "type": "message",
                    "data": {"content": f"\n\n</details>\n\n"},
                }
            )

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Речевые фрагменты распознаны!",
                        "done": True,
                        "hidden": False,
                    },
                }
            )

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
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Ошибка при транскрибации: {e}",
                        "done": True,
                        "hidden": False,
                    },
                }
            )
