"""
title: Diagram Drawer
description: Рисует диаграмму по пользовательскому запросу. Ключевые слова: диаграмма, график, workflow, блок-схема, UML
author: Sergei Vyaznikov
version: 0.2
requirements: requests
"""

import zlib
import base64
import requests

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

from pydantic import Field, BaseModel
from typing import Optional, Callable, Awaitable, Any


def encode_plantuml(text):
    utf8_bytes = text.encode("utf-8")
    compressed = zlib.compress(utf8_bytes)[2:-4]  # Remove zlib header and checksum
    plantuml_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"

    result = []
    for i in range(0, len(compressed), 3):
        chunk = compressed[i : i + 3]
        num = 0
        for j, byte in enumerate(chunk):
            num += byte << (16 - 8 * j)

        for j in range(4):
            if i * 8 + j * 6 < len(compressed) * 8:  # Ensure we have enough bits
                six_bits = (num >> (18 - 6 * j)) & 0x3F
                result.append(plantuml_chars[six_bits])

    return "".join(result)


def get_plantuml_image(
    uml: str, valves: dict, plantuml_url: str = "http://plantuml-server:8080"
) -> str:
    try:
        encoded_uml = encode_plantuml(uml)
        image_url = f"{valves.PLANTUML_URL}/plantuml/png/{encoded_uml}"
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()

        if "image/png" not in response.headers.get("Content-Type", ""):
            raise ValueError("Response is not a PNG image")

        image_base64 = base64.b64encode(response.content).decode("utf-8")
        return image_base64

    except requests.RequestException as e:
        raise requests.RequestException(
            f"Failed to fetch image from PlantUML server: {str(e)}"
        )
    except Exception as e:
        raise ValueError(f"Failed to process image: {str(e)}")


class Tools:

    class Valves(BaseModel):

        OLLAMA_BASE_URL: str = Field(
            default="http://localhost:11434",
            description="Домен для доступа к Ollama (указывается изнутри контейнера)",
        )

        OLLAMA_MODEL_NAME: str = Field(
            default="qwen3:14b",
            description="Название модели для генерации диаграмм",
        )

        PLANTUML_URL: str = Field(
            default="http://plantuml-server:8080/",
            description="Домен для доступа к PlantUML (указывается изнутри контейнера)",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def draw_diagram(self) -> str:
        """
        Инструмент для рисования диаграмм

        :return: Дальнейшие инструкции
        :rtype: str
        """
        response = """
        Ваша задача заключается в том, чтобы составить диаграмму с помощью mermaid и отправить её пользователю.
        ВСЕ ПОДПИСИ следует экранировать с помощью кавычек ""
        Таким образом:
        A --> B[ЛЮБОЙ ТЕКСТ] >>> A --> B["ЛЮБОЙ ТЕКСТ"] 
        """
        return response

    async def draw_uml_diagram(
        self, title: str, __messages__=None, __event_emitter__=None
    ) -> str:
        """
        Инструмент для рисования UML-диаграмм

        Используется только если встречается ключевое слово UML

        :param title: Название диаграммы
        :type title: str

        :return: Дальнейшие инструкции
        :rtype: str
        """

        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": "Генерирую диаграмму...",
                    "done": False,
                    "hidden": False,
                },
            }
        )

        context = str(list(filter(lambda x: x["role"] == "user", __messages__)))
        generate_uml_prompt_template = """/no_think
Ты — составитель UML-диаграмм. Твоя задача — составлять UML-диаграммы на основе запроса пользователя.

Правила формирования ответа:
1.  Твой ответ — это ИСКЛЮЧИТЕЛЬНО валидный UML-код, ничего более.
2.  Не пиши ничего больше. Не пиши никаких пометок или объяснений.
3.  Не используй никакого внешнего форматирования. Твой ответ должен без дополнительных преобразований компилироваться в диаграмму.

Инструкций по шаблону:
- Начинай свой ответ со строчки `@startuml`
- Заканчивай свой ответ строчкой `@enduml`

Название диаграммы: {title}

Контекст сообщений пользователя:
<context>
{context}
</context>
"""

        generate_uml_prompt = PromptTemplate.from_template(generate_uml_prompt_template)
        generate_uml_prompt = generate_uml_prompt.invoke(
            {"title": title, "context": context}
        )

        llm = OllamaLLM(
            model=self.valves.OLLAMA_MODEL_NAME, base_url=self.valves.OLLAMA_BASE_URL
        )

        await __event_emitter__(
            {
                "type": "message",
                "data": {
                    "content": "<details>\n<summary>Исходный код диаграммы</summary>\n```uml\n"
                },
            }
        )

        chunks = []
        is_thinking = False
        batch = []
        batch_size = 20

        for chunk in llm.stream(generate_uml_prompt):
            if chunk == "<think>":
                is_thinking = True

            if chunk == "</think>":
                is_thinking = False
                continue

            if is_thinking:
                continue

            batch.append(chunk)
            chunks.append(chunk)
            if len(batch) == batch_size:
                await __event_emitter__(
                    {
                        "type": "message",
                        "data": {"content": "".join(batch).strip()},
                    }
                )
                batch = []

        if batch:
            await __event_emitter__(
                {
                    "type": "message",
                    "data": {"content": "".join(batch).strip()},
                }
            )

        uml_code = "".join(chunks).strip()

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
                    "description": "Диаграмма сгенерирована!",
                    "done": True,
                    "hidden": False,
                },
            }
        )

        image_base64 = get_plantuml_image(uml_code, valves=self.valves)

        formatted_image = f"\n\n![image](data:image/png;base64,{image_base64})\n\n"

        await __event_emitter__(
            {
                "type": "message",  # or simply "message"
                "data": {"content": formatted_image},
            }
        )

        response = "Диаграмма была сгенерирована и отправлена пользователю. \
        Ваша задача заключается в том, чтобы уведомить пользователя о выполнении его запроса. \
        Сообщите, что посмотреть исходный код диаграммы можно во вкладке 'Сгенерированный UML код'."

        return response
