"""
title: Data Analyst
description: Анализирует пользовательские данные. Ключевые слова: аналитика данных, построение графиков
author: Sergei Vyaznikov
version: 0.1
requirements: requests, plotly, pandas
"""

from io import BytesIO
from pathlib import Path
import aiohttp
import zlib
import base64
import requests

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
from pydantic import Field, BaseModel
from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from plotly.graph_objects import Figure
from typing import Optional, Callable, Awaitable, Any


def get_context(df: pd.DataFrame) -> str:
    if df is None:
        raise Exception("No DataFrame loaded.")
    return df.head(10).to_string()


def execute_code(df: pd.DataFrame, code: str) -> Figure:
    local_vars = {"pd": pd, "df": df, "px": px, "plt": plt}
    exec(code, {}, local_vars)
    result = local_vars.get("result")
    if result is None:
        raise HTTPException(status_code=500, detail="Не удалось построить график")
    return result


def process_prompt_stream(llm: OllamaLLM, prompt: PromptTemplate) -> str:
    chunks = list()
    is_thinking = False
    for chunk in llm.stream(prompt):
        if chunk == "<think>":
            is_thinking = True

        if chunk == "</think>":
            is_thinking = False
            continue

        if is_thinking:
            continue

        chunks.append(chunk)

    response = "".join(chunks).strip()
    return response


async def get_file_content(auth_data: str, file_id: str, valves: dict) -> BytesIO:
    """Получить массив байт с содержимым файла

    :param auth_data: Данные авторизации пользователя
    :param file_id: Идентификатор файла в OpenWebUI
    :param valves: Вентили из OpenWebUI
    """
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
    """Получить jwt-токен текущего пользователя

    :param request: Объект __request__ из OpenWebUI
    """
    auth_data = request.headers.get("Authorization")
    return auth_data


async def make_df_from_file(
    auth_data: str, file_data: dict, valves: dict
) -> pd.DataFrame:
    """Создать pd.DataFrame из файла диалога

    :param auth_data: Данные авторизации пользователя
    :param file_data: Данные о файле из __files__
    :param valves: Вентили из OpenWebUI
    """
    file_id = file_data.get("id")
    file_name = file_data.get("meta").get("name")
    file_extension = Path(file_name).suffix.lower()

    file_content = await get_file_content(
        auth_data=auth_data, file_id=file_id, valves=valves
    )

    readers = {
        ".csv": pd.read_csv,
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
        ".parquet": pd.read_parquet,
        ".feather": pd.read_feather,
        ".json": pd.read_json,
    }

    pd_reader = readers.get(file_extension)
    if not pd_reader:
        raise HTTPException(status_code=400, detail="Неподдерживаемый формат файла")

    df = pd_reader(file_content)
    return df


class Tools:

    class Valves(BaseModel):

        OLLAMA_BASE_URL: str = Field(
            default="http://localhost:11434",
            description="Домен для доступа к Ollama (указывается изнутри контейнера)",
        )

        OLLAMA_MODEL_NAME: str = Field(
            default="qwen3:14b",
            description="Название модели для генерации графиков",
        )

        INTERNAL_URL: str = Field(
            default="http://127.0.0.1:8080",
            description="Домен для доступа к OpenWebUI изнутри контейнера",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def analyse_data(
        self,
        prompt: str,
        __request__=None,
        __files__=None,
        __messages__=None,
        __event_emitter__=None,
    ) -> str:
        """
        Инструмент для построения графика из пользовательских данных

        :param prompt: Запрос для построения графика
        :type prompt: str

        :return: Дальнейшие инструкции
        :rtype: str
        """
        if not __files__:
            raise HTTPException(status_code=400, detail="Файл не обнаружен")

        auth_data = get_user_auth_data(__request__)

        file_data = __files__[-1].get("file")

        df = await make_df_from_file(
            auth_data=auth_data, file_data=file_data, valves=self.valves
        )

        llm = OllamaLLM(
            model=self.valves.OLLAMA_MODEL_NAME,
            base_url=self.valves.OLLAMA_BASE_URL,
        )

        generate_graph_prompt_template = """/no_think
Ты — специалист по данным. Твоя задача — писать код на Python для визуализации графиков.

## Входные данные
    - Первые 10 строчек датафрейма `df`
    - Запрос пользователя 

## Состояние среды исполнения кода
    - Никакие дополнительные импорты не требуются.
    - Уже определена библиотека pandas (доступ по псевдониму `pd`).
    - Уже определена библиотека plotly.express (доступ по псевдониму `px`).
    - Уже определён pandas.DataFrame, который требуется визуализировать (доступ по псевдониму `df`).

## Выходные данные
    - Код на языке программирования Python для визуализации датафрема `df`.

## Требования к коду
    - Ни в коем случае не переопределять существующий датафрейм `df`; работать с определённым заранее.
    - Для визуализации графика использовать библиотеку plotly.express (`px`).
    - Результирующий график обязан быть сохранён в переменной `result` (эту переменную необходимо определить).


Правила формирования ответа:
1.  Твой ответ — это ИСКЛЮЧИТЕЛЬНО валидный Python-код, ничего более.
2.  Не пиши ничего больше. Не пиши никаких пометок или объяснений.
3.  Не используй никакого внешнего форматирования. Твой ответ должен без дополнительных преобразований запускаться в интерпретаторе Python.

Контекст, первые 10 строчек датафрейма:
<context>
{context}
</context>

Запрос пользователя: {prompt}
"""

        fix_error_prompt_template = """/no_think
Ты — специалист по программированию на Python. Твоя задача — исправить ошибку в предоставленном коде.

## Входные данные
    - Код, содержащий ошибку
    - Описание ошибки

## Состояние среды исполнения кода
    - Никакие дополнительные импорты не требуются, если они не указаны в исходном коде.
    - Все библиотеки, использованные в исходном коде, считаются доступными.

## Выходные данные
    - Исправленный код на языке программирования Python.

## Требования к коду
    - Код должен быть исправлен так, чтобы устранить указанную ошибку.
    - Сохранять оригинальную логику кода, насколько это возможно.
    - Не добавлять лишних изменений, не связанных с исправлением ошибки.

Правила формирования ответа:
1. Твой ответ — это ИСКЛЮЧИТЕЛЬНО валидный Python-код, ничего более.
2. Не пиши никаких пометок, комментариев или объяснений.
3. Не используй никакого внешнего форматирования. Твой ответ должен без дополнительных преобразований запускаться в интерпретаторе Python.

Код с ошибкой:
<code>
{code}
</code>

Ошибка:
<error>
{error}
</error>
"""

        context = get_context(df)

        generate_graph_prompt = PromptTemplate.from_template(
            generate_graph_prompt_template
        ).invoke({"context": context, "prompt": prompt})

        generate_graph_code = process_prompt_stream(llm, generate_graph_prompt)
        try:
            result = execute_code(df, generate_graph_code)
        except Exception as e:
            error = str(e)
            fix_error_prompt = PromptTemplate.from_template(
                fix_error_prompt_template
            ).invoke({"code": generate_graph_code, "error": error})
            fixed_generate_graph_code = process_prompt_stream(llm, fix_error_prompt)
            result = execute_code(df, fixed_generate_graph_code)

        html_content = result.to_html()

        headers = {"Content-Disposition": "inline"}

        return HTMLResponse(content=html_content, headers=headers)
