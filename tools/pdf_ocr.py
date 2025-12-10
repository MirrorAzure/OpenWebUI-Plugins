"""
title: PDF text recognizer
author: Sergei Vyaznikov
description: Распознаёт текст из PDF и возвращает документ с размеченным текстом. Ключевые слова: OCR, распознавание PDF, сканы документов
version: 0.4
"""

import os
import asyncio
import aiohttp
import logging
import tempfile
import datetime
import ocrmypdf
from io import BytesIO
from typing import Any, Literal, List, Tuple, Callable
from pydantic import BaseModel, Field, field_validator


logger = logging.getLogger(__name__)


async def async_ocr(
    input_file,
    output_file,
    language: List[str] = None,
    force_ocr: bool = False,
    **kwargs,
):
    """
    Asynchronous wrapper around the synchronous ocrmypdf.ocr function.
    Runs the OCR process in a thread pool to avoid blocking the event loop.

    :param input_file: Input file path, bytes, or file-like object.
    :param output_file: Output file path.
    :param language: List of languages, e.g., ["rus", "eng"].
    :param force_ocr: Whether to force OCR even if text is present.
    :param kwargs: Additional arguments for ocrmypdf.ocr.
    :return: The result of ocrmypdf.ocr (typically None on success).
    """
    loop = asyncio.get_running_loop()

    # Define a synchronous callable for the executor
    def sync_ocr():
        return ocrmypdf.ocr(
            input_file, output_file, language=language, force_ocr=force_ocr, **kwargs
        )

    # Run the sync function in a thread pool
    return await loop.run_in_executor(None, sync_ocr)


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


def get_user_auth_data(request) -> str:
    """Получить jwt-токен текущего пользователя"""
    auth_data = request.headers["authorization"]
    return auth_data


def get_content_type(filename: str) -> str:
    """Получить MIME-type для

    :param filename: Полное имя файла (на данный момент корректно обрабатывает docx и xlsx)
    :return: MIME-type для файла (по умолчанию возвращает application/octet-stream)
    """
    # TODO: добавить больше MIME-типов
    content_types = {
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "pdf": "application/pdf",
    }
    file_extension = filename.split(".")[-1]
    mime_type = content_types.get(file_extension, "application/octet-stream")
    return mime_type


async def upload_document_to_server(
    filename: str, file_path: str, valves: dict, auth_data: str
) -> dict:
    """Загружает файл на сервер OpenWebUI

    :param filename: Название файла для загрузки
    :param file_path: Путь до файла

    :return: Статус загрузки файла
    """
    headers = {
        "Authorization": auth_data,
        "Accept": "application/json",
    }

    try:
        # Используется параметр quote_fields, чтобы не экранировать название файла
        form = aiohttp.FormData(quote_fields=False)
        mime_type = get_content_type(filename)
        form.add_field(
            "file",
            open(file_path, "rb"),
            filename=filename,
            content_type=mime_type,
        )

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60)
        ) as session:
            async with session.post(
                "http://127.0.0.1:8080/api/v1/files/",
                headers=headers,
                data=form,
            ) as resp:
                status = resp.status

                # Сначала проверяем статус
                if status < 200 or status >= 300:
                    text = await resp.text()
                    return {"error": f"HTTP {status}", "raw_response": text}

                # Только при успешном статусе парсим JSON
                try:
                    data = await resp.json()
                except Exception:
                    text = await resp.text()
                    return {
                        "error": "Invalid JSON in response",
                        "status_code": status,
                        "raw_response": text,
                    }

        # Получаем ID файла
        file_id = data.get("id") or data.get("uuid") or data.get("file_id")
        if not file_id:
            return {"error": "No file ID in response", "json": data}

        # Формируем ссылку для загрузки файла
        download_url = f"{valves.PUBLIC_DOWNLOAD_DOMAIN}/api/v1/files/{file_id}/content"
        return {"url": download_url}

    except Exception as e:
        return {"error": f"Upload failed: {e}"}


class Tools:

    class Valves(BaseModel):

        DEBUG_MODE: bool = Field(default=False, description="Режим отладки")

        INTERNAL_URL: str = Field(
            default="http://127.0.0.1:8080",
            description="Домен для доступа к OpenWebUI изнутри контейнера",
        )

        PUBLIC_DOWNLOAD_DOMAIN: str = Field(
            default="http://localhost:3000",
            description="Домен для доступа к OpenWebUI (без символа `/` в конце)",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def recognize_pdf_text(
        self,
        __files__=None,
        __request__=None,
        __event_emitter__=None,
    ) -> str:
        """
        Инструмент для распознавания текста в PDF

        Используется для создания копируемого текста в PDF-документе.

        :return: Статус распознавания PDF-файла.
        :rtype: str.
        """
        try:
            emitter = EventEmitter(__event_emitter__)

            await emitter.status("Распознаю PDF-документ...", done=False)

            if not __files__ or len(__files__) <= 0:
                return "Файлы не были обнаружены. Ваша задача заключается в том, чтобы уведомить пользователя об ошибке и предложить приложить файл к сообщению."

            file_data = __files__[-1].get("file")
            file_id = file_data.get("id")

            logger.info(f"Информация о загруженном файле: {file_data}")

            auth_data = __request__.headers["authorization"]

            file_bytes = await get_file_content(
                auth_data=auth_data, file_id=file_id, valves=self.valves
            )

            unique_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            with tempfile.NamedTemporaryFile(
                prefix=unique_name, suffix=".pdf", delete=False
            ) as temp_file:
                await async_ocr(
                    file_bytes, temp_file.name, language=["rus", "eng"], force_ocr=True
                )
                # ocrmypdf.ocr(
                #     file_bytes, temp_file.name, language=["rus", "eng"], force_ocr=True
                # )
                upload_response = await upload_document_to_server(
                    filename=f"{unique_name}.pdf",
                    file_path=temp_file.name,
                    valves=self.valves,
                    auth_data=auth_data,
                )

            document_url = upload_response.get("url")
            if not document_url:
                error = upload_response.get("error")
                raw_response = upload_response.get("raw_response")
                response = f"""Во время загрузки файла произошла ошибка: {error}. Полный текст: {raw_response}. Ваша задача заключается в том, чтобы сообщить об ошибке пользователю."""
                await emitter.status(
                    f"Во время загрузки файла произошла ошибка: {error}."
                )
                return response
            formatted_url = f"""
\n\n
[Скачать документ]({document_url})
\n\n
        """

            await emitter.message(formatted_url)

            response = "Распознавание текста для указанного файла было проведено. Размеченный файл был отправлен пользователю. \
            Ваша задача заключается в том, чтобы уведомить пользователя о выполнении его запроса. \
            Сообщите, что получить изменённый файл можно по кнопке 'скачать документ'."
            return response

        except Exception as e:
            await emitter.status(f"Ошибка при распознавании документа: {e}", done=True)
            return f"Ошибка при распознавании документа: {e}"
