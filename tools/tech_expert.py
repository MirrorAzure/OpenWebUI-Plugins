"""
title: Data Analyst
description: Получает данные из системы ТехЭксперт. Ключевые слова: ТехЭксперт
author: Sergei Vyaznikov
version: 0.3
"""

import os
import uuid
import math
import json
import torch
import aiohttp
import logging
import datetime

import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings

from fastapi import HTTPException
from typing import Any, Literal, List, Tuple, Callable
from pydantic import BaseModel, Field, field_validator
from pydantic.types import conint

from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize

CHROMA_DB_MAX_BATCH_SIZE = 5461

logger = logging.getLogger(__name__)


class EventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Any] = None):
        self.event_emitter = event_emitter

    async def emit(self, description="Unknown State", status="in_progress", done=False):
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


def get_embedding_model(valves: dict) -> SentenceTransformer:
    embedding_model_name = os.environ.get(
        "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedder = SentenceTransformer(embedding_model_name)
    return embedder


async def search_techexpert_documents(
    valves: dict, query: str, has_annot: bool = True, max_docs: int = 10
):
    url = f"{valves.TECH_EXPERT_URL}/search"
    params = {
        "query": query,
        "has_annot": "true" if has_annot else "false",
        "max_docs": max_docs,
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return data
            else:
                error_content = await response.text()
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Запрос вернул ошибку: {error_content}",
                )


async def retrieve_techexpert_document(valves: dict, nd: int, format: str = "markdown"):
    url = f"{valves.TECH_EXPERT_URL}/document"
    params = {"nd": nd, "format": format}

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return data
            else:
                error_content = await response.text()
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Запрос вернул ошибку: {error_content}",
                )


def chunk_text(text: str, max_chunk_size: int = 1000, overlap: int = 0):
    if overlap < 0:
        raise ValueError("Overlap не может быть отрицательным.")

    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        # Если текущий чанк пуст, просто добавляем предложение
        if not current_chunk:
            current_chunk.append(sentence)
            current_length = sentence_length
        else:
            # Проверяем, помещается ли текущее предложение с учётом overlap
            if current_length + sentence_length <= max_chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                # Если не помещается, сохраняем текущий чанк
                chunks.append(" ".join(current_chunk))
                # Создаём новый чанк с перекрытием
                overlap_text = " ".join(current_chunk[-overlap:]) if overlap > 0 else ""
                current_chunk = [sentence] + sent_tokenize(overlap_text)
                current_length = len(" ".join(current_chunk))

    # Добавляем оставшийся чанк
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def create_collection_from_chunks(
    valves: dict,
    collection_name: str,
    chunk_data: list,
    embedding_model: SentenceTransformer,
):
    client = chromadb.Client(Settings())

    if collection_name in [coll.name for coll in client.list_collections()]:
        client.delete_collection(name=collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": valves.DISTANCE_METRIC},
    )

    chunk_content_list = [chunk.get("content", "") for chunk in chunk_data]
    chunk_metadata_list = [chunk.get("metadata", {}) for chunk in chunk_data]
    document_vectors = embedding_model.encode(chunk_content_list)

    num_items = len(chunk_data)
    num_batches = math.ceil(num_items / CHROMA_DB_MAX_BATCH_SIZE)

    for idx in range(num_batches):
        start_idx = idx * CHROMA_DB_MAX_BATCH_SIZE
        end_idx = min((idx + 1) * CHROMA_DB_MAX_BATCH_SIZE, num_items)

        batch_ids = [str(idx) for idx in range(start_idx, end_idx)]
        batch_documents = chunk_content_list[start_idx:end_idx]
        batch_embeddings = document_vectors[start_idx:end_idx]
        batch_metadatas = chunk_metadata_list[start_idx:end_idx]

        collection.add(
            ids=batch_ids,
            documents=batch_documents,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
        )
    return collection


class Tools:

    class Valves(BaseModel):

        MAX_DOCS: int = Field(
            default=10, description="Количество документов для поиска"
        )

        CHUNK_SIZE: conint(ge=500, le=2000) = Field(
            json_schema_extra={"format": "range", "minimum": 100, "maximum": 2000},
            default=1000,
            description="Размер чанков для выбора семантически близких документов",
        )

        CHUNK_OVERLAP: conint(ge=100, le=2000) = Field(
            json_schema_extra={"format": "range", "minimum": 100, "maximum": 2000},
            default=1000,
            description="Размер наслоения чанков",
        )

        TECH_EXPERT_URL: str = Field(
            default="http://mcp_techexpert:8000",
            description="Домен для доступа к API ТехЭксперта",
        )

        INTERNAL_URL: str = Field(
            default="http://127.0.0.1:8080",
            description="Домен для доступа к OpenWebUI изнутри контейнера",
        )

        DISTANCE_METRIC: Literal["cosine", "l2", "ip"] = Field(
            default="cosine",
            description="Метрика для определения семантической близости",
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

    async def search_techexpert(
        self,
        queries: list[str],
        __messages__=None,
        __event_emitter__=None,
    ) -> str:
        """
        Инструмент для получения документов из системы ТехЭксперт

        Инструмент возвращает список релевантных документов, вместе с ссылками на них

        :param queries: Запрос для поиска документов в системе
        :type queries: list[str]

        :return: Данные из системы ТехЭксперт со ссылками на документы
        :rtype: str
        """
        try:
            emitter = EventEmitter(__event_emitter__)

            logger.info(f"Поисковые запросы: {str(queries)}")

            await emitter.emit("Получение документов из системы ТехЭксперт...")

            collection_name = f"{uuid.uuid4()}_tech_expert"

            user_query = list(
                filter(lambda message: message.get("role") == "user", __messages__)
            )[-1].get("content")
            processed_nd = set()
            chunk_data = []
            for query in queries:
                response = await search_techexpert_documents(
                    valves=self.valves, query=query, max_docs=self.valves.MAX_DOCS
                )
                documents = response.get("documents")
                logger.info(
                    f"Найденные документы: {str(list(map(lambda x: x.get('data_nd', ''), documents)))}"
                )
                for document in documents:

                    document_nd = document.get("data_nd")
                    if document_nd in processed_nd:
                        continue
                    processed_nd.add(document_nd)

                    document_text = await retrieve_techexpert_document(
                        valves=self.valves, nd=document_nd
                    )

                    if not document_text:
                        continue
                    chunks = chunk_text(
                        document_text,
                        max_chunk_size=self.valves.CHUNK_SIZE,
                        overlap=self.valves.CHUNK_OVERLAP,
                    )
                    for chunk in chunks:
                        chunk_data.append(
                            {
                                "metadata": {
                                    "document_name": document.get("document_name"),
                                    "link": document.get("link"),
                                },
                                "content": chunk,
                            }
                        )

            embedding_model = get_embedding_model(valves=self.valves)

            document_collection = create_collection_from_chunks(
                valves=self.valves,
                collection_name=collection_name,
                chunk_data=chunk_data,
                embedding_model=embedding_model,
            )

            logger.info(f"Коллекция создана: {collection_name}")
            # return str(body)
            query_embedding = embedding_model.encode(user_query)

            results = document_collection.query(
                query_embeddings=[query_embedding], n_results=10
            )

            client = chromadb.Client(Settings())
            client.delete_collection(name=collection_name)

            logger.info(f"Коллекция удалена: {collection_name}")

            tool_response = list()

            logger.info(f"Сформированный ссылки:")
            for document, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                document_name = metadata.get("document_name")
                link = metadata.get("link")

                logger.info(f"Источник: {link}")
                logger.info(f"Название: {document_name}")

                await __event_emitter__(
                    {
                        "type": "citation",
                        "data": {
                            "document": [document],
                            "metadata": [
                                {
                                    "source": document_name,
                                    "url": link,
                                }
                            ],
                            "source": {"name": document_name, "url": link},
                        },
                    }
                )

                response_element = {
                    "document_link": link,
                    "document_name": document_name,
                    "distance": distance,
                    # "content": document,
                }
                tool_response.append(response_element)

            await emitter.emit(
                status="complete",
                description="Данные из системы ТехЭксперт были получены!",
                done=True,
            )

            return json.dumps(tool_response, ensure_ascii=False)

        except Exception as e:
            await emitter.emit(f"При работе системы возникла ошибка: {e}", done=True)