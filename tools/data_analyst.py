"""
title: Data Analyst
description: Анализирует пользовательские данные. Ключевые слова: аналитика данных, построение графиков
author: Sergei Vyaznikov
version: 0.3
"""

from io import BytesIO
from pathlib import Path
import aiohttp
import zlib
import base64
import requests

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

import pandas as pd
import plotly.express as px
from bs4 import BeautifulSoup
import logging
from matplotlib import pyplot as plt
from pydantic import Field, BaseModel
from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from plotly.graph_objects import Figure
from typing import Optional, Callable, Awaitable, Any, Literal


russian_locale_script = """var locale={moduleType:"locale",name:"ru",dictionary:{Autoscale:"\u0410\u0432\u0442\u043e\u043c\u0430\u0442\u0438\u0447\u0435\u0441\u043a\u043e\u0435 \u0448\u043a\u0430\u043b\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u0435","Box Select":"\u0412\u044b\u0434\u0435\u043b\u0435\u043d\u0438\u0435 \u043f\u0440\u044f\u043c\u043e\u0443\u0433\u043e\u043b\u044c\u043d\u043e\u0439 \u043e\u0431\u043b\u0430\u0441\u0442\u0438","Click to enter Colorscale title":"\u041d\u0430\u0436\u043c\u0438\u0442\u0435 \u0434\u043b\u044f \u0432\u0432\u043e\u0434\u0430 \u043d\u0430\u0437\u0432\u0430\u043d\u0438\u044f \u0446\u0432\u0435\u0442\u043e\u0432\u043e\u0439 \u0448\u043a\u0430\u043b\u044b","Click to enter Component A title":"\u041d\u0430\u0436\u043c\u0438\u0442\u0435 \u0434\u043b\u044f \u0432\u0432\u043e\u0434\u0430 \u043d\u0430\u0437\u0432\u0430\u043d\u0438\u044f \u043a\u043e\u043c\u043f\u043e\u043d\u0435\u043d\u0442\u0430 A","Click to enter Component B title":"\u041d\u0430\u0436\u043c\u0438\u0442\u0435 \u0434\u043b\u044f \u0432\u0432\u043e\u0434\u0430 \u043d\u0430\u0437\u0432\u0430\u043d\u0438\u044f \u043a\u043e\u043c\u043f\u043e\u043d\u0435\u043d\u0442\u0430 B","Click to enter Component C title":"\u041d\u0430\u0436\u043c\u0438\u0442\u0435 \u0434\u043b\u044f \u0432\u0432\u043e\u0434\u0430 \u043d\u0430\u0437\u0432\u0430\u043d\u0438\u044f \u043a\u043e\u043c\u043f\u043e\u043d\u0435\u043d\u0442\u0430 C","Click to enter Plot title":"\u041d\u0430\u0436\u043c\u0438\u0442\u0435 \u0434\u043b\u044f \u0432\u0432\u043e\u0434\u0430 \u043d\u0430\u0437\u0432\u0430\u043d\u0438\u044f \u0433\u0440\u0430\u0444\u0438\u043a\u0430","Click to enter X axis title":"\u041d\u0430\u0436\u043c\u0438\u0442\u0435 \u0434\u043b\u044f \u0432\u0432\u043e\u0434\u0430 \u043d\u0430\u0437\u0432\u0430\u043d\u0438\u044f \u043e\u0441\u0438 X","Click to enter Y axis title":"\u041d\u0430\u0436\u043c\u0438\u0442\u0435 \u0434\u043b\u044f \u0432\u0432\u043e\u0434\u0430 \u043d\u0430\u0437\u0432\u0430\u043d\u0438\u044f \u043e\u0441\u0438 Y","Click to enter radial axis title":"\u041d\u0430\u0436\u043c\u0438\u0442\u0435 \u0434\u043b\u044f \u0432\u0432\u043e\u0434\u0430 \u043d\u0430\u0437\u0432\u0430\u043d\u0438\u044f \u043f\u043e\u043b\u044f\u0440\u043d\u043e\u0439 \u043e\u0441\u0438","Compare data on hover":"\u041f\u0440\u0438 \u043d\u0430\u0432\u0435\u0434\u0435\u043d\u0438\u0438 \u043f\u043e\u043a\u0430\u0437\u044b\u0432\u0430\u0442\u044c \u0432\u0441\u0435 \u0434\u0430\u043d\u043d\u044b\u0435","Double-click on legend to isolate one trace":"\u0414\u0432\u0430\u0436\u0434\u044b \u0449\u0451\u043b\u043a\u043d\u0438\u0442\u0435 \u043f\u043e \u043b\u0435\u0433\u0435\u043d\u0434\u0435 \u0434\u043b\u044f \u0432\u044b\u0434\u0435\u043b\u0435\u043d\u0438\u044f \u043e\u0442\u0434\u0435\u043b\u044c\u043d\u044b\u0445 \u0434\u0430\u043d\u043d\u044b\u0445","Double-click to zoom back out":"\u0414\u043b\u044f \u0441\u0431\u0440\u043e\u0441\u0430 \u043c\u0430\u0441\u0448\u0442\u0430\u0431\u0430 \u043a \u0437\u043d\u0430\u0447\u0435\u043d\u0438\u044e \u043f\u043e \u0443\u043c\u043e\u043b\u0447\u0430\u043d\u0438\u044e \u0434\u0432\u0430\u0436\u0434\u044b \u0449\u0451\u043b\u043a\u043d\u0438\u0442\u0435 \u043c\u044b\u0448\u044c\u044e","Download plot":"\u0421\u043e\u0445\u0440\u0430\u043d\u0438\u0442\u044c \u0433\u0440\u0430\u0444\u0438\u043a","Download plot as a PNG":"\u0421\u043e\u0445\u0440\u0430\u043d\u0438\u0442\u044c \u0432 \u0444\u043e\u0440\u043c\u0430\u0442\u0435 PNG","Edit in Chart Studio":"\u0420\u0435\u0434\u0430\u043a\u0442\u0438\u0440\u043e\u0432\u0430\u0442\u044c \u0432 Chart Studio","IE only supports svg.  Changing format to svg.":"IE \u043f\u043e\u0434\u0434\u0435\u0440\u0436\u0438\u0432\u0430\u0435\u0442 \u0442\u043e\u043b\u044c\u043a\u043e svg. \u0424\u043e\u0440\u043c\u0430\u0442 \u0441\u043c\u0435\u043d\u044f\u0435\u0442\u0441\u044f \u043d\u0430 svg.","Lasso Select":"\u041b\u0430\u0441\u0441\u043e","Orbital rotation":"\u041e\u0440\u0431\u0438\u0442\u0430\u043b\u044c\u043d\u043e\u0435 \u0434\u0432\u0438\u0436\u0435\u043d\u0438\u0435",Pan:"\u0421\u0434\u0432\u0438\u0433","Produced with Plotly.js":"\u0421\u043e\u0437\u0434\u0430\u043d\u043e \u0441 \u043f\u043e\u043c\u043e\u0449\u044c\u044e Plotly.js",Reset:"\u0421\u0431\u0440\u043e\u0441\u0438\u0442\u044c \u043a \u0437\u043d\u0430\u0447\u0435\u043d\u0438\u044f\u043c \u043f\u043e \u0443\u043c\u043e\u043b\u0447\u0430\u043d\u0438\u044e","Reset axes":"\u0421\u0431\u0440\u043e\u0441\u0438\u0442\u044c \u043e\u0442\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u0435 \u043e\u0441\u0435\u0439 \u043a \u0437\u043d\u0430\u0447\u0435\u043d\u0438\u044f\u043c \u043f\u043e \u0443\u043c\u043e\u043b\u0447\u0430\u043d\u0438\u044e","Reset camera to default":"\u0421\u0431\u0440\u043e\u0441\u0438\u0442\u044c \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b \u043a\u0430\u043c\u0435\u0440\u044b \u043a \u0437\u043d\u0430\u0447\u0435\u043d\u0438\u044f\u043c \u043f\u043e \u0443\u043c\u043e\u043b\u0447\u0430\u043d\u0438\u044e","Reset camera to last save":"\u0421\u0431\u0440\u043e\u0441\u0438\u0442\u044c \u043a\u0430\u043c\u0435\u0440\u0443 \u043a \u043f\u043e\u0441\u043b\u0435\u0434\u043d\u0435\u043c\u0443 \u0441\u043e\u0445\u0440\u0430\u043d\u0451\u043d\u043d\u043e\u043c\u0443 \u0441\u043e\u0441\u0442\u043e\u044f\u043d\u0438\u044e","Reset view":"\u0421\u0431\u0440\u043e\u0441\u0438\u0442\u044c \u043e\u0442\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u0435 \u043a \u0437\u043d\u0430\u0447\u0435\u043d\u0438\u044f\u043c \u043f\u043e \u0443\u043c\u043e\u043b\u0447\u0430\u043d\u0438\u044e","Reset views":"\u0421\u0431\u0440\u043e\u0441\u0438\u0442\u044c \u043e\u0442\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u044f \u043a \u0437\u043d\u0430\u0447\u0435\u043d\u0438\u044f\u043c \u043f\u043e \u0443\u043c\u043e\u043b\u0447\u0430\u043d\u0438\u044e","Show closest data on hover":"\u041f\u0440\u0438 \u043d\u0430\u0432\u0435\u0434\u0435\u043d\u0438\u0438 \u043f\u043e\u043a\u0430\u0437\u044b\u0432\u0430\u0442\u044c \u0431\u043b\u0438\u0436\u0430\u0439\u0448\u0438\u0435 \u0434\u0430\u043d\u043d\u044b\u0435","Snapshot succeeded":"\u0421\u043d\u0438\u043c\u043e\u043a \u0443\u0441\u043f\u0435\u0448\u043d\u043e \u0441\u043e\u0437\u0434\u0430\u043d","Sorry, there was a problem downloading your snapshot!":"\u041a \u0441\u043e\u0436\u0430\u043b\u0435\u043d\u0438\u044e, \u0432\u043e\u0437\u043d\u0438\u043a\u043b\u0430 \u043f\u0440\u043e\u0431\u043b\u0435\u043c\u0430 \u043f\u0440\u0438 \u0441\u043e\u0445\u0440\u0430\u043d\u0435\u043d\u0438\u0438 \u0441\u043d\u0438\u043c\u043a\u0430","Taking snapshot - this may take a few seconds":"\u0414\u0435\u043b\u0430\u0435\u0442\u0441\u044f \u0441\u043d\u0438\u043c\u043e\u043a - \u044d\u0442\u043e \u043c\u043e\u0436\u0435\u0442 \u0437\u0430\u043d\u044f\u0442\u044c \u043d\u0435\u0441\u043a\u043e\u043b\u044c\u043a\u043e \u0441\u0435\u043a\u0443\u043d\u0434","Toggle Spike Lines":"\u0412\u043a\u043b\u044e\u0447\u0438\u0442\u044c/\u0432\u044b\u043a\u043b\u044e\u0447\u0438\u0442\u044c \u043e\u0442\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u0435 \u043b\u0438\u043d\u0438\u0439 \u043f\u0440\u043e\u0435\u043a\u0446\u0438\u0439 \u0442\u043e\u0447\u0435\u043a","Toggle show closest data on hover":"\u0412\u043a\u043b\u044e\u0447\u0438\u0442\u044c/\u0432\u044b\u043a\u043b\u044e\u0447\u0438\u0442\u044c \u043f\u043e\u043a\u0430\u0437 \u0431\u043b\u0438\u0436\u0430\u0439\u0448\u0438\u0445 \u0434\u0430\u043d\u043d\u044b\u0445 \u043f\u0440\u0438 \u043d\u0430\u0432\u0435\u0434\u0435\u043d\u0438\u0438","Turntable rotation":"\u0412\u0440\u0430\u0449\u0435\u043d\u0438\u0435 \u043d\u0430 \u043f\u043e\u0432\u043e\u0440\u043e\u0442\u043d\u043e\u043c \u0441\u0442\u043e\u043b\u0435",Zoom:"\u0417\u0443\u043c","Zoom in":"\u0423\u0432\u0435\u043b\u0438\u0447\u0438\u0442\u044c","Zoom out":"\u0423\u043c\u0435\u043d\u044c\u0448\u0438\u0442\u044c","close:":"\u0417\u0430\u043a\u0440\u044b\u0442\u0438\u0435:","concentration:":"\u041a\u043e\u043d\u0446\u0435\u043d\u0442\u0440\u0430\u0446\u0438\u044f:","high:":"\u041c\u0430\u043a\u0441\u0438\u043c\u0443\u043c:","incoming flow count:":"\u041a\u043e\u043b\u0438\u0447\u0435\u0441\u0442\u0432\u043e \u0432\u0445\u043e\u0434\u044f\u0449\u0438\u0445 \u0441\u0432\u044f\u0437\u0435\u0439:","kde:":"\u042f\u0434\u0435\u0440\u043d\u0430\u044f \u043e\u0446\u0435\u043d\u043a\u0430 \u043f\u043b\u043e\u0442\u043d\u043e\u0441\u0442\u0438:","lat:":"\u0428\u0438\u0440\u043e\u0442\u0430:","lon:":"\u0414\u043e\u043b\u0433\u043e\u0442\u0430:","low:":"\u041c\u0438\u043d\u0438\u043c\u0443\u043c:","lower fence:":"\u041d\u0438\u0436\u043d\u044f\u044f \u0433\u0440\u0430\u043d\u0438\u0446\u0430:","max:":"\u041c\u0430\u043a\u0441.:","mean \xb1 \u03c3:":"\u0421\u0440\u0435\u0434\u043d\u0435\u0435 \xb1 \u03c3:","mean:":"\u0421\u0440\u0435\u0434\u043d\u0435\u0435:","median:":"\u041c\u0435\u0434\u0438\u0430\u043d\u0430:","min:":"\u041c\u0438\u043d.:","new text":"\u041d\u043e\u0432\u044b\u0439 \u0442\u0435\u043a\u0441\u0442","open:":"\u041e\u0442\u043a\u0440\u044b\u0442\u0438\u0435:","outgoing flow count:":"\u041a\u043e\u043b\u0438\u0447\u0435\u0441\u0442\u0432\u043e \u0438\u0441\u0445\u043e\u0434\u044f\u0449\u0438\u0445 \u0441\u0432\u044f\u0437\u0435\u0439:","q1:":"q1:","q3:":"q3:","source:":"\u0418\u0441\u0442\u043e\u0447\u043d\u0438\u043a:","target:":"\u0426\u0435\u043b\u044c:",trace:"\u0420\u044f\u0434","upper fence:":"\u0412\u0435\u0440\u0445\u043d\u044f\u044f \u0433\u0440\u0430\u043d\u0438\u0446\u0430:"},format:{days:["\u0432\u043e\u0441\u043a\u0440\u0435\u0441\u0435\u043d\u044c\u0435","\u043f\u043e\u043d\u0435\u0434\u0435\u043b\u044c\u043d\u0438\u043a","\u0432\u0442\u043e\u0440\u043d\u0438\u043a","\u0441\u0440\u0435\u0434\u0430","\u0447\u0435\u0442\u0432\u0435\u0440\u0433","\u043f\u044f\u0442\u043d\u0438\u0446\u0430","\u0441\u0443\u0431\u0431\u043e\u0442\u0430"],shortDays:["\u0432\u0441","\u043f\u043d","\u0432\u0442","\u0441\u0440","\u0447\u0442","\u043f\u0442","\u0441\u0431"],months:["\u042f\u043d\u0432\u0430\u0440\u044c","\u0424\u0435\u0432\u0440\u0430\u043b\u044c","\u041c\u0430\u0440\u0442","\u0410\u043f\u0440\u0435\u043b\u044c","\u041c\u0430\u0439","\u0418\u044e\u043d\u044c","\u0418\u044e\u043b\u044c","\u0410\u0432\u0433\u0443\u0441\u0442","\u0421\u0435\u043d\u0442\u044f\u0431\u0440\u044c","\u041e\u043a\u0442\u044f\u0431\u0440\u044c","\u041d\u043e\u044f\u0431\u0440\u044c","\u0414\u0435\u043a\u0430\u0431\u0440\u044c"],shortMonths:["\u042f\u043d\u0432.","\u0424\u0435\u0432\u0440.","\u041c\u0430\u0440\u0442","\u0410\u043f\u0440.","\u041c\u0430\u0439","\u0418\u044e\u043d\u044c","\u0418\u044e\u043b\u044c","\u0410\u0432\u0433.","\u0421\u0435\u043d\u0442.","\u041e\u043a\u0442.","\u041d\u043e\u044f","\u0414\u0435\u043a."],date:"%d.%m.%Y",decimal:",",thousands:" "}};"undefined"==typeof Plotly?(window.PlotlyLocales=window.PlotlyLocales||[],window.PlotlyLocales.push(locale)):Plotly.register(locale);"""


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


async def process_prompt_stream(
    llm: ChatOllama | ChatOpenAI, prompt: PromptTemplate
) -> str:
    chunks = list()
    is_thinking = False
    async for chunk in llm.astream(prompt):
        chunk_text = chunk.content
        if chunk_text == "<think>":
            is_thinking = True

        if chunk_text == "</think>":
            is_thinking = False
            continue

        if is_thinking:
            continue

        chunks.append(chunk_text)

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
        try:

            if not __files__:
                raise HTTPException(status_code=400, detail="Файл не обнаружен")

            auth_data = get_user_auth_data(__request__)

            file_data = __files__[-1].get("file")

            df = await make_df_from_file(
                auth_data=auth_data, file_data=file_data, valves=self.valves
            )

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

            generate_graph_code = await process_prompt_stream(
                llm, generate_graph_prompt
            )
            try:
                result = execute_code(df, generate_graph_code)
            except Exception as e:
                error = str(e)
                fix_error_prompt = PromptTemplate.from_template(
                    fix_error_prompt_template
                ).invoke({"code": generate_graph_code, "error": error})
                fixed_generate_graph_code = await process_prompt_stream(
                    llm, fix_error_prompt
                )
                result = execute_code(df, fixed_generate_graph_code)

            config = {
                "locale": "ru",
                "displaylogo": False,
            }

            html_content = result.to_html(config=config)

            soup = BeautifulSoup(html_content, "lxml")
            custom_block = soup.new_tag("script", type="text/javascript")
            custom_block.string = russian_locale_script

            scripts = soup.find_all("script")
            if scripts:
                first_script = scripts[0]
                first_script.insert_after(custom_block)

            html_content = soup.prettify()

            await __event_emitter__(
                {
                    "type": "embeds",
                    "data": {
                        "embeds": [html_content],
                    },
                }
            )

            response = "График был сгенерирован и отправлен пользователю. \
            Ваша задача заключается в том, чтобы уведомить пользователя о выполнении его запроса."

            return response

        except Exception as e:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Ошибка при генерации графика: {e}",
                        "done": True,
                        "hidden": False,
                    },
                }
            )
