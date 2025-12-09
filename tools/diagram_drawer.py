"""
title: Diagram Drawer
description: Рисует диаграмму по пользовательскому запросу. Ключевые слова: диаграмма, график, workflow, блок-схема, UML
author: Sergei Vyaznikov
version: 0.4
"""

import zlib
import base64
import requests

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from pydantic import Field, BaseModel
from typing import Optional, Callable, Awaitable, Any, Literal


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
    uml: str,
    valves: dict,
    data_type="png",
) -> str:
    try:
        encoded_uml = encode_plantuml(uml)
        image_url = f"{valves.PLANTUML_URL}/plantuml/{data_type}/{encoded_uml}"
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        if data_type == "png":
            if "image/png" not in response.headers.get("Content-Type", ""):
                raise ValueError("Response is not a PNG image")

            image_base64 = base64.b64encode(response.content).decode("utf-8")
            return image_base64
        elif data_type == "svg":
            svg_content = response.text
            return svg_content

    except requests.RequestException as e:
        raise requests.RequestException(
            f"Failed to fetch image from PlantUML server: {str(e)}"
        )
    except Exception as e:
        raise ValueError(f"Failed to process image: {str(e)}")


class Tools:

    class Valves(BaseModel):

        DEBUG_MODE: bool = Field(
            default=False, description="Режим отладки (Отправляет mock-диаграмму)"
        )

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

        # RETURN_FORMAT: Literal["png", "svg"] = Field(
        #     default="svg",
        #     description="Формат возвращаемой диаграммы",
        # )

        MODEL_NAME: str = Field(
            default="qwen3:14b",
            description="Название модели для генерации документов",
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
        try:
            emitter = EventEmitter(__event_emitter__)

            await emitter.status("Генерирую диаграмму...")

            if self.valves.DEBUG_MODE:
                mock_diagram = (
                    """```plantuml\n@startuml\nBob -> Alice : hello\n@enduml\n```\n"""
                )
                await emitter.message(mock_diagram)

                await emitter.status("Диаграмма сгенерирована!", done=True)

                response = "Инструмент запущен в режиме отладки. Тестовая диаграмма была отправлена пользователю. \
                Ваша задача заключается в том, чтобы уведомить пользователя, что инструмент работает."
                return response

            context = str(list(__messages__[-10:]))
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

Контекст сообщений в чате:
<context>
{context}
</context>
    """

            generate_uml_prompt = PromptTemplate.from_template(
                generate_uml_prompt_template
            )
            generate_uml_prompt = generate_uml_prompt.invoke(
                {"title": title, "context": context}
            )

            if self.valves.LLM_BACKEND == "ollama":
                llm = ChatOllama(
                    model=self.valves.MODEL_NAME, base_url=self.valves.OLLAMA_BASE_URL
                )
            elif self.valves.LLM_BACKEND == "OpenAI":
                llm = ChatOpenAI(
                    model=self.valves.MODEL_NAME,
                    openai_api_base=self.valves.OPEN_AI_BASE_URL,
                    openai_api_key="not-needed",
                    temperature=0.2,
                )

            chunks = []
            is_thinking = False
            batch = ["```plantuml\n"]
            batch_size = 20

            async for chunk in llm.astream(generate_uml_prompt):
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
                if len(batch) >= batch_size:
                    await emitter.message("".join(batch))
                    batch = []

            if batch:
                await emitter.message("".join(batch))

            uml_code = "".join(chunks).strip()

            await emitter.message("\n```\n")

            await emitter.status("Диаграмма сгенерирована!", done=True)

            response = "Диаграмма была сгенерирована и отправлена пользователю. \
            Ваша задача заключается в том, чтобы уведомить пользователя о выполнении его запроса."

            return response

        except Exception as e:
            await emitter.status(f"Ошибка при генерации диаграммы: {e}", done=True)
