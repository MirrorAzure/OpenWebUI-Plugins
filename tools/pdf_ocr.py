"""
title: PDF text recognizer
author: Sergei Vyaznikov
description: Распознаёт текст из PDF и возвращает документ с размеченным текстом. Ключевые слова: OCR, распознавание PDF, сканы документов
version: 0.3
"""

import os
import aiohttp
import tempfile
import datetime
import ocrmypdf
from io import BytesIO
from typing import Any, Literal, List, Tuple
from pydantic import BaseModel, Field, field_validator


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
            if not __files__ or len(__files__) <= 0:
                return "Файлы не были обнаружены. Ваша задача заключается в том, чтобы уведомить пользователя об ошибке и предложить приложить файл к сообщению."

            file_data = __files__[-1].get("file")
            file_id = file_data.get("id")

            auth_data = __request__.headers["authorization"]

            file_bytes = await get_file_content(
                auth_data=auth_data, file_id=file_id, valves=self.valves
            )

            unizue_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            with tempfile.NamedTemporaryFile(
                prefix=unizue_name, suffix=".pdf", delete=False
            ) as temp_file:
                ocrmypdf.ocr(
                    file_bytes, temp_file.name, language=["rus", "eng"], force_ocr=True
                )
                upload_response = await upload_document_to_server(
                    filename=f"{unizue_name}.pdf",
                    file_path=temp_file.name,
                    valves=self.valves,
                    auth_data=auth_data,
                )

            document_url = upload_response.get("url")
            if not document_url:
                error = upload_response.get("error")
                raw_response = upload_response.get("raw_response")
                response = f"""Во время загрузки файла произошла ошибка: {error}. Полный текст: {raw_response}. Ваша задача заключается в том, чтобы сообщить об ошибке пользователю."""

            formatted_url = f"""
\n\n
[Скачать документ]({document_url})
\n\n
        """

            await __event_emitter__(
                {
                    "type": "message",
                    "data": {"content": formatted_url},
                }
            )

            response = "Распознавание текста для указанного файла было проведено. Размеченный файл был отправлен пользователю. \
            Ваша задача заключается в том, чтобы уведомить пользователя о выполнении его запроса. \
            Сообщите, что получить изменённый файл можно по кнопке 'скачать документ'."
            return response

        except Exception as e:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Ошибка при распознавании документа: {e}",
                        "done": True,
                        "hidden": False,
                    },
                }
            )
