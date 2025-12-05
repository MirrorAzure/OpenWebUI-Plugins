"""
title: Document Writter
description: Пишет документ по пользовательскому запросу. Ключевые слова: статья, документ, приказ, распоряжение, пояснительная записка, акт
author: Sergei Vyaznikov
version: 0.8
"""

import os
import re
import aiohttp
import tempfile
import subprocess

from pydantic import Field, BaseModel
from typing import Optional, Callable, Awaitable, Any, Literal
from datetime import datetime
from fastapi import HTTPException

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


def convert_latex_to_docx(latex_code: str) -> str:
    """Конвертирует LaTeX-код в файл .docx

    :param latex_code: Код на LaTeX
    :return: Путь до сгенерированного .docx-файла
    """
    # Создаём временные файлы
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, encoding="utf-8"
    ) as temp_tex_file:
        temp_tex_file.write(latex_code)
        tex_file_path = temp_tex_file.name

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, encoding="utf-8"
    ) as temp_docx_file:
        docx_file_path = temp_docx_file.name

    # Вызов команды pandoc через subprocess
    subprocess.run(
        ["pandoc", tex_file_path, "-f", "latex", "-t", "docx", "-o", docx_file_path],
        check=True,
    )  # check=True вызовет исключение, если команда завершится с ошибкой

    # Удаляем LaTeX-файл после конвертации
    os.remove(tex_file_path)

    return docx_file_path


def get_content_type(filename: str) -> str:
    """Получить MIME-type для

    :param filename: Полное имя файла (на данный момент корректно обрабатывает docx и xlsx)
    :return: MIME-type для файла (по умолчанию возвращает application/octet-stream)
    """
    # TODO: добавить больше MIME-типов
    content_types = {
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
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
                f"{valves.INTERNAL_URL}/api/v1/files/",
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

        OLLAMA_BASE_URL: str = Field(
            default="http://localhost:11434",
            description="Домен для доступа к Ollama (указывается изнутри контейнера)",
        )

        OPEN_AI_BASE_URL: str = Field(
            default="http://localhost:8080/v1",
            description="Домен для доступа к OpenAI-совместимому API из контейнера (http://host:port/v1)",
        )

        LLM_BACKEND: Literal["ollama", "OpenAI"] = Field(
            default="ollama",
            description="Бэкэнд для инференса языковых моделей",
        )

        MODEL_NAME: str = Field(
            default="qwen3:14b",
            description="Название модели для генерации документов",
        )

        PUBLIC_DOWNLOAD_DOMAIN: str = Field(
            default="http://localhost:3000",
            description="Домен для доступа к OpenWebUI (без символа `/` в конце)",
        )

        INTERNAL_URL: str = Field(
            default="http://localhost:3000",
            description="Домен для доступа контейнера к самому себе",
        )

        MAX_FILENAME_LEN: int = Field(
            default=64, description="Задаёт максимальную длину названия для файлов"
        )

    def __init__(self):
        self.file_handler = True
        self.valves = self.Valves()

    async def write_table(self) -> str:
        """
        Инструмент для составления таблиц

        :return: Дальнейшие инструкции
        :rtype: str
        """
        response = "Ваша задача заключается в том, чтобы составить таблицу в формате markdown и отправить её пользователю."
        return response

    async def write_document(
        self,
        title: str,
        __model__=None,
        __request__=None,
        __messages__=None,
        __event_emitter__=None,
        __event_call__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> dict:
        """
        Инструмент для написания документов (статей, распоряжений, приказов и т.д.)

        Используется для генерации любых документов, требующих форматирования.
        Инструмент автоматически сгенерирует и отправит пользователю нужный документ и вернёт статус написания.

        :param title: Название документа для генерации.
        :type title: str.

        :return: Статус написания документа.
        :rtype: str.
        """
        try:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Генерирую документ...",
                        "done": False,
                        "hidden": False,
                    },
                }
            )

            context = str(list(__messages__[-10:]))

            prompt_template = """/no_think
Ты — технический писатель. Твоя задача — составлять документы и статьи на основе запроса пользователя.

Правила формирования ответа:
1.  Твой ответ — это ИСКЛЮЧИТЕЛЬНО валидный LaTeX-код, ничего более.
2.  Для листингов кода используй пакет minted.
3.  Для таблиц используй \\begin{{tabular}}.
4.  Для научных статей оформляй документ согласно стандартным требованиям (шрифт, отступы, интервалы). В конце таких статей обязательно добавляй раздел "Список литературы", используя команды \\bibliography и \\bibliographystyle.

ЖЕСТКАЯ ИНСТРУКЦИЯ ПО ШАБЛОНУ:
Начинай документ СТРОГО следующей преамбулой (НИЧЕГО НЕ МЕНЯЯ и НИЧЕГО НЕ ДОБАВЛЯЯ в нее). Этих пакетов ДОСТАТОЧНО:

\\documentclass[a4paper,14pt]{{extarticle}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T2A]{{fontenc}}
\\usepackage[russian]{{babel}}
\\usepackage{{graphicx}}
\\usepackage[margin=2cm]{{geometry}}
\\usepackage{{parskip}}
\\usepackage{{indentfirst}}
\\usepackage{{minted}}
\\usepackage{{color}}
\\usepackage[numbers]{{natbib}}

\\definecolor{{codebackground}}{{rgb}}{{0.95,0.95,0.95}}

\\begin{{document}}

После преамбулы НЕМЕДЛЕННО переходи к генерации запрошенного пользователем содержимого. Закончи документ командой \\end{{document}}

Название документа: {title}

Предоставленный контекст:
<context>
{context}
</context>
                """

            prompt = PromptTemplate.from_template(prompt_template)
            prompt = prompt.invoke({"title": title, "context": context})

            if self.valves.LLM_BACKEND == "ollama":
                llm = ChatOllama(
                    model=self.valves.MODEL_NAME, base_url=self.valves.OLLAMA_BASE_URL
                )
            elif self.valves.LLM_BACKEND == "llama.cpp":
                llm = ChatOpenAI(
                    model=self.valves.MODEL_NAME,
                    openai_api_base=self.valves.OPEN_AI_BASE_URL,
                    openai_api_key="not-needed",
                    temperature=0.2,
                )

            await __event_emitter__(
                {
                    "type": "message",
                    "data": {
                        "content": "<details>\n<summary>Исходный код документа</summary>\n```latex\n"
                    },
                }
            )

            chunks = []
            is_thinking = False
            batch = []
            batch_size = 20

            async for chunk in llm.astream(prompt):
                chunk_text = chunk.content
                if chunk_text == "<think>":
                    is_thinking = True

                if chunk_text == "</think>":
                    is_thinking = False
                    continue

                if is_thinking:
                    continue

                batch.append(chunk_text)
                chunks.append(chunk_text)
                if len(batch) == batch_size:
                    await __event_emitter__(
                        {
                            "type": "message",
                            "data": {"content": "".join(batch)},
                        }
                    )
                    batch = []

            if batch:
                await __event_emitter__(
                    {
                        "type": "message",
                        "data": {"content": "".join(batch)},
                    }
                )

            latex_code = "".join(chunks).strip()

            # await __event_emitter__(
            #     {
            #         "type": "message",
            #         "data": {"content": latex_code},
            #     }
            # )

            await __event_emitter__(
                {
                    "type": "message",
                    "data": {"content": "\n```\n</details>"},
                }
            )

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Документ сгенерирован!",
                        "done": True,
                        "hidden": False,
                    },
                }
            )

            # await __event_emitter__(
            #     {
            #         "type": "message",
            #         "data": {"content": markdown_tagged_text},
            #     }
            # )

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Отправляю документ пользователю...",
                        "done": False,
                        "hidden": False,
                    },
                }
            )

            docx_file_path = convert_latex_to_docx(latex_code)

            # model_name = __model__.get("id")
            # current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            # unique_name = f"Document_{model_name}_{current_time}.docx"

            # Получаем данные авторизации из запроса пользователя
            # Нужно, чтобы пользователь имел доступ к сгенерированному файлу
            auth_data = __request__.headers["authorization"]

            # Обрезаем название файла до заданного максимального
            # Иначе OpenWebUI может выдавать ошибку 400 на длинных названиях
            safe_title = re.sub(r'[<>:"/\\|?*:]', "", title)
            upload_response = await upload_document_to_server(
                filename=f"{safe_title[:self.valves.MAX_FILENAME_LEN]}.docx",
                file_path=docx_file_path,
                valves=self.valves,
                auth_data=auth_data,
            )

            # Удаляем .docx-файл после загрузки на сервер
            os.remove(docx_file_path)

            error = upload_response.get("error")

            if error:
                raw_response = upload_response.get("raw_response")
                response = f"Во время генерации документа произошла ошибка: {error}. Ваша задача заключается в том, чтобы уведомить пользователя об ошибке."
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Ошибка при генерации документа: {raw_response}",
                            "done": True,
                            "hidden": False,
                        },
                    }
                )
                return response

            document_url = upload_response.get("url")

            # В идеале использовать html-теги <a> для формирования ссылки, но по какой-то причине их не поддерживает OpenWebUI
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

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Документ отправлен!",
                        "done": True,
                        "hidden": False,
                    },
                }
            )
            response = "Документ был написан и отправлен пользователю. \
            Ваша задача заключается в том, чтобы уведомить пользователя о выполнении его запроса. \
            Сообщите, что скачать его можно по кнопке 'Скачать документ'"
            return response
        except Exception as e:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Ошибка при генерации документа: {e}",
                        "done": True,
                        "hidden": False,
                    },
                }
            )
