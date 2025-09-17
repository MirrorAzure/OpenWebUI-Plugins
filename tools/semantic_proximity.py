"""
title: Семантическая близость
description: Сопоставляет две таблицы (excel-файла) и возвращает результат сопоставления. Ключевые слова: сопоставление таблиц, спосоставление excel-файлов
author: Sergei Vyaznikov
version: 0.2
requirements: fastapi, aiohttp, pydantic, xlsxwriter, chromadb, pandas, sentence_transformers
"""

import os
import math
import aiohttp
import difflib
import tempfile
import xlsxwriter

import pandas as pd
from io import BytesIO
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional, Callable, Awaitable, Any, Union, Literal

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CHROMA_DB_MAX_BATCH_SIZE = 5461
collection_name = "similarity_temp"
device_name = "cuda"


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


async def make_df_from_excel_file(
    auth_data: str, file_data: dict, valves: dict
) -> pd.DataFrame:
    """Создать pd.DataFrame из файла диалога

    :param auth_data: Данные авторизации пользователя
    :param file_data: Данные о файле из __files__
    :param valves: Вентили из OpenWebUI
    """
    file_id = file_data.get("id")
    file_content = await get_file_content(
        auth_data=auth_data, file_id=file_id, valves=valves
    )
    df = pd.read_excel(file_content)
    return df


def get_embedding_model() -> SentenceTransformer:
    """Загружает модель эмбеддера из переменной среды RAG_EMBEDDING_MODEL"""
    # TODO: добавить выбор модели в вентили
    embedding_model_name = os.environ.get(
        "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedder = SentenceTransformer(embedding_model_name)
    # embedder.to(device_name)
    return embedder


def get_proximity_df(
    first_df: pd.DataFrame,
    second_df: pd.DataFrame,
    first_column: str,
    second_column: str,
    valves: dict,
) -> pd.DataFrame:
    """Сопоставлет два датафрейма по столбцам

    :param first_df: Первый датафрейм
    :param second_df: Второй датафрейм
    :param first_column: Название столбца из первого датафрейма
    :param second_column: Название столбца из второго датафрейма
    :param valves: Вентили из OpenWebUI

    :return: Объединённый датафрейм по строкам (один ко многим)
    """
    # Проверяем, есть ли вообще такие столбцы
    assert (
        first_column in first_df.columns
    ), f"Столбец {first_column} не обнаружен в первом файле. Пожалуйста, проверьте правильность написания"
    assert (
        second_column in second_df.columns
    ), f"Столбец {second_column} не обнаружен во втором файле. Пожалуйста, проверьте правильность написания"

    # Загрузка модели эмбеддера
    embedder_model = get_embedding_model()

    # Очистка датасета от пустых значений в целевых столбцах
    first_df.dropna(subset=[first_column], inplace=True)
    first_df.reset_index(drop=True, inplace=True)
    second_df.dropna(subset=[second_column], inplace=True)
    second_df.reset_index(drop=True, inplace=True)

    # Создаём векторы для наименований второго файла
    second_names_list = second_df[second_column].tolist()
    second_vectors = embedder_model.encode(
        second_names_list  # , device=device_name, show_progress_bar=True
    )

    # Инициируем коллекцию ChromaDB в оперативной памяти
    client = chromadb.Client(Settings())

    # Проверка существования коллекции и удаление, если существует
    if collection_name in [coll.name for coll in client.list_collections()]:
        client.delete_collection(collection_name=collection_name)

    # Создание новой коллекции
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": valves.DISTANCE_METRIC},  # Указываем метрику расстояния
    )

    # У ChromaDB есть ограничение на количество одновременно добавляемых точек
    # Разделяем все точки на батчи
    num_items = len(second_df)
    num_batches = math.ceil(num_items / CHROMA_DB_MAX_BATCH_SIZE)

    for idx in range(num_batches):
        start_idx = idx * CHROMA_DB_MAX_BATCH_SIZE
        end_idx = min((idx + 1) * CHROMA_DB_MAX_BATCH_SIZE, num_items)

        # Получаем текущий батч
        batch_ids = [str(idx) for idx in range(start_idx, end_idx)]
        batch_embeddings = second_vectors[start_idx:end_idx]

        # Добавляем батч в коллекцию
        try:
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
            )
        except Exception as e:
            print(f"Error adding batch {idx+1}: {e}")

    # Создаём векторы для наименований первого файла
    first_names_list = first_df[first_column].tolist()
    first_vectors = embedder_model.encode(
        first_names_list  # , device=device_name, show_progress_bar=True
    )

    # Сохраняем индекс ближайшего совпадения для каждой строки в первом файле
    similar_indices = []
    similarities = []
    for idx, vector in enumerate(first_vectors):
        res = collection.query(vector, n_results=1)
        original_index = int(res.get("ids")[0][0])
        similarity = 1 - res["distances"][0][0]
        similar_indices.append(original_index)
        similarities.append(similarity)

    # Формируем новый датафрейм из двух
    merged_df = first_df.copy()
    merged_df["second_df_index"] = similar_indices
    merged_df = merged_df.join(second_df, on="second_df_index")
    merged_df["Семантическая близость"] = similarities

    # Сортируем по семантической близости
    sorted_df = merged_df.sort_values(by="Семантическая близость", ascending=False)
    sorted_df.reset_index(inplace=True, drop=True)

    return sorted_df


def custom_excel_save(
    df: pd.DataFrame,
    excel_file: Union[str, Path],
    first_column: Union[int, str] = 0,
    second_column: Union[int, str] = 1,
    first_color: str = "green",
    second_color: str = "red",
    sheet_name: str = "Sheet1",
    has_header: bool = True,
) -> None:
    """Функция для кастомной записи датафрейма в excel-файл
    с отображением разницы между двумя выбранными столбцами

    :param df: Данные для записи
    :type df: pd.DataFrame

    :param excel_file: Путь до результирующего Excel-файла
    :type excel_file: Union[str, Path]

    :param first_column: Название или индекс первого столбца для сравнения
    :type first_column: Union[int, str]

    :param second_column: Название или индекс второго столбца для сравнения
    :type second_column: Union[int, str]

    :param first_color: Цвет для отображения разницы в первом столбце
    :type first_color: str

    :param second_color: Цвет для отображения разницы во втором столбце
    :type second_color: str

    :param sheet_name: Название страницы Excel, куда будет производиться запись
    :type sheet_name: str

    :param has_header: Записывать ли заголовок
    :type has_header: bool
    """

    workbook = xlsxwriter.Workbook(excel_file)
    worksheet = workbook.add_worksheet(sheet_name)

    first_format = workbook.add_format({"font_color": first_color})
    second_format = workbook.add_format({"font_color": second_color})
    bold_format = workbook.add_format({"bold": True})

    if type(first_column) is int:
        first_col_idx = first_column
    elif type(first_column) is str:
        first_col_idx = (df.columns.to_list()).index(first_column)
    else:
        raise Exception("first_column type must be in [int, str]")

    if type(second_column) is int:
        second_col_idx = first_column
    elif type(second_column) is str:
        second_col_idx = (df.columns.to_list()).index(second_column)
    else:
        raise Exception("second_column type must be in [int, str]")

    if has_header:
        for col_idx, column_name in enumerate(df.columns.to_list()):
            # print(col_idx, column_name)
            try:
                worksheet.write(0, col_idx, str(column_name), bold_format)
            except:
                pass

    for row_idx, row in df.iterrows():

        cur_row_idx = row_idx + has_header

        for col_idx, value in enumerate(row.to_list()):
            try:
                worksheet.write(cur_row_idx, col_idx, value)
            except:
                pass

        first_text = str(row.to_list()[first_col_idx])
        second_text = str(row.to_list()[second_col_idx])
        if not first_text or not second_text:
            continue
        first_args = []
        diff = difflib.ndiff(first_text, second_text)
        for change in diff:
            if change.startswith("  "):
                first_args.append(change[2:])
            elif change.startswith("- "):
                first_args.extend((first_format, change[2:]))
            elif change.startswith("+ "):
                continue  # Удаленные символы не добавляем

        worksheet.write_rich_string(cur_row_idx, first_col_idx, *first_args)

        second_args = []
        diff = difflib.ndiff(second_text, first_text)
        for change in diff:
            if change.startswith("  "):
                second_args.append(change[2:])
            elif change.startswith("- "):
                second_args.extend((second_format, change[2:]))
            elif change.startswith("+ "):
                continue  # Удаленные символы не добавляем

        worksheet.write_rich_string(cur_row_idx, second_col_idx, *second_args)

    workbook.close()


def remove_html_tags(text: str, tag: str = "think") -> str:
    """Убирает html-теги с заданным названием из ответа нейросети

    :param text: Сообщение для форматирования
    :param tag: Название тега
    :return: Сообщение без тегов
    """
    # TODO: добавить логику для полной очистки содержимого тегов
    cleaned_text = text.replace(f"</{tag}>", "").replace(f"<{tag}>", "")
    return cleaned_text


def add_code_tags(text: str, lang: str = "python") -> str:
    """Добавляет кодовые теги к сообщению, если их нет

    :param text: Сообщение для форматирования
    :param lang: Наименование языка
    :return: Сообщение с markdown-тегами для кода
    """
    if f"```{lang}" in text:
        return text
    else:
        return f"```{lang}\n{text}\n```"


def add_markdown_hidden_tags(text: str, summary: str = "Скрытое поле") -> str:
    """Добавляет теги Markdown для скрытия кода

    :param text: Сообщение для форматирования
    :type text: str

    :param summary: Краткий текст для отображения
    :type summary: str

    :return: Сообщение с markdown-тегами для скрытия кода
    :rtype: str
    """
    # Данное форматирование создаёт выпадающий элемент с заданным текстом
    markdown_tagged_text = f"""
<details>
<summary>{summary}</summary>
{text}
</details>
"""
    return markdown_tagged_text


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


def get_user_auth_data(request) -> str:
    """Получить jwt-токен текущего пользователя

    :param request: Объект __request__ из OpenWebUI
    """
    auth_data = request.headers.get("Authorization")
    return auth_data


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

        PUBLIC_DOWNLOAD_DOMAIN: str = Field(
            default="http://localhost:3000",
            description="Домен для доступа к OpenWebUI (без символа `/` в конце)",
        )

        INTERNAL_URL: str = Field(
            default="http://127.0.0.1:8080",
            description="Домен для доступа к OpenWebUI изнутри контейнера",
        )

        DISTANCE_METRIC: Literal["cosine", "l2", "ip"] = Field(
            default="cosine",
            description="Метрика для определения семантической близости",
        )

    def __init__(self):
        self.file_handler = True
        self.valves = self.Valves()

    async def compare_documents(
        self,
        first_column: str,
        second_column: str,
        __request__=None,
        __files__=None,
        __event_emitter__=None,
        __event_call__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> dict:
        """Производит сопоставление двух файлов по заданным колонкам.

        :param fisrt_column: Название первой колонки
        :type first_column: str
        :param second_column: Название второй колонки
        :type second_column: str
        :return: Дальнейшие указания
        """

        if not __files__:
            return "Файлы не найдены. Ваша задача заключается в том, чтобы уведомить пользователя об ошибке."

        if len(__files__) < 2:
            return "Для сопоставления требуется два файла. Ваша задача заключается в том, чтобы уведомить пользователя об ошибке."

        auth_data = get_user_auth_data(__request__)

        first_file = __files__[-2].get("file")
        second_file = __files__[-1].get("file")

        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": "Получение таблиц из файлов...",
                    "done": False,
                    "hidden": False,
                },
            }
        )

        first_df = await make_df_from_excel_file(
            auth_data=auth_data, file_data=first_file, valves=self.valves
        )

        second_df = await make_df_from_excel_file(
            auth_data=auth_data, file_data=second_file, valves=self.valves
        )

        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": "Таблицы из файлов получены!",
                    "done": True,
                    "hidden": False,
                },
            }
        )

        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": "Формирование смежной таблицы...",
                    "done": False,
                    "hidden": False,
                },
            }
        )
        proximity_df = get_proximity_df(
            first_df=first_df,
            second_df=second_df,
            first_column=first_column,
            second_column=second_column,
            valves=self.valves,
        )

        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": "Смежная таблица сформирована!",
                    "done": True,
                    "hidden": False,
                },
            }
        )

        # Обрабатываем возможные коллизии в именах
        first_column = (
            first_column
            if first_column in proximity_df.columns
            else f"{first_column}_x"
        )
        second_column = (
            second_column
            if second_column in proximity_df.columns
            else f"{second_column}_y"
        )

        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        unique_name = f"Сопоставление_таблиц_{current_time}.xlsx"

        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": "Формирование результирующего excel-файла...",
                    "done": False,
                    "hidden": False,
                },
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, unique_name)
            custom_excel_save(
                df=proximity_df,
                excel_file=temp_file_path,
                first_column=first_column,
                second_column=second_column,
            )

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Результирующий файл сформирован!",
                        "done": True,
                        "hidden": False,
                    },
                }
            )

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Формирование ссылки на скачивание...",
                        "done": False,
                        "hidden": False,
                    },
                }
            )
            url_response = upload_document_to_server(
                filename=unique_name,
                file_path=temp_file_path,
                valves=self.valves,
                auth_data=auth_data,
            )
            file_url = url_response.get("url")

        if not file_url:
            return "Не удалось загрузить файл на сервер. Ваша задача заключается в том, чтобы уведомить пользователя об ошибке."

        formatted_url = f"""
\n\n
[Скачать файл]({file_url})
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
                    "description": "Файл успешно отправлен!",
                    "done": True,
                    "hidden": False,
                },
            }
        )

        response = "Сопоставление было проведено и отправлено пользователю. \
        Ваша задача заключается в том, чтобы уведомить пользователя о выполнении его запроса. \
        Сообщите, что скачать сопоставленный файл можно по кнопке 'Скачать файл'"
        return response
