"""
title: Markdown Table to Excel
author: Sergei Vyaznikov
description: Извлекает Markdown-таблицы и конвертирует их в .xlsx
icon_url:  data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAACXBIWXMAAAsSAAALEgHS3X78AAADxklEQVR4Xu1bT6hOQRS/j0eUt7GRP1HKSrJAOdnLgkJJJNnoRUoJzxQrdCIR0WMpG7382Vgoe7Nhg42FpJdii9Qjf85593m+O9/c783MvXPuvL47NfXevTP39zu/OXPm75dlKSaEtRnCdglqgxIgVgwExj5M+QLlxZYyI/TsSWx+8QVAWENGXKUs0qK+gtUjAMJcAj5I+SLlpb4kmizvJwDCSiJ7mfLeJknXiW0XAIFb8zxlNrip9FsCeICi7T0COiABliDGtjlESkTpBI1nSk/9YoCkFUoPRIVD+MPfZw/o69QK0NfN33aBLEs3CE4FqS4P7QyOZWXMSj0CahsD2hjQ5wr0fRdINwi6zARdyszg4TE9YIywT1NeT3koq4FsjN7q4wGXyIgzBRIIvGQ+ayE2TmWLewYISS66fATgPbqiAEqfo+V0twBK2/YR/BY3Sc4DEDYHuSHCRFA9gUq+MUBbOG0xnq22lJkvYEsQhK8AGbl8sY7SzwvISr834sTPIGZClfwFyLJfFm7/AtwzyzufOCNk9n+YusgtpE9OUOTfarT+12CLXIZNlzJR5gEI3w23/1GCsyhYAKGKIV2AqXGLm6n4LYQvQjZUggnvAgiPyeV3TaMrPbnJ2JGGKjFLch5QtGhnqYEInyoZL1g5tAvkFBF2l3BdImhDJahqAij9qAR9XyVWgpWrCHC3lKfS9wVtqAQVHgSVPlRARhimoHin49lx+vt6MDuXMd6lzAwEQj3glGE8L5JuF54pfSPYeMGKYQIofcXgaFskcZFjgrYEQYV0gT09hr8P1A1WTb9X+haNFDeDmCU7D1D6geH+7zr+t22E8LWZZJNvF9hksaS4/kdYYcQC25ZZMoL4CaD0C6P111ksGbc8G03GYoOIjwC22d0rJ8OUPupUroFCPkHwMwW0N8Tx4WRW+nWPYDhK748Y73mOMOxso8sY71JmBkC+JMUzOr4VFjvx5oj7CrEG43oa1MAVGXfjYzdFx/d9uoAgLYJKdh4gK0N0NJ9RIDqZJgBaAZpQPSXMdINgfqT+radYNQyVbRdIyR2b4JKyByyTECTlGPCWJkPRNUjZA6IbTwBjKXvACRoFrsVWgT1gXmyQwO+L8BoklfcTQc55yo+7eCNzVv38LVBky23x/Lir+8gL4SQ952txC0LBUqznHgPys4DieQACr/FZFD4FmpXJ7+6eq4kIy6konxRV+bnsCHVP/pFm1OTuAT40lP5IxXd0VUHYMCXMRp/PxSwbR4Ayxkq/pFfdZwsIfNmCD1LFf6n6F/QrwhzXYOQMAAAAAElFTkSuQmCC
version: 0.3
requirements: fastapi, pandas
"""

import re
import base64
from io import BytesIO
from datetime import datetime
from typing import Optional, Callable, Awaitable, Any
from fastapi import HTTPException
import pandas as pd


class Action:
    class Valves:
        pass

    def __init__(self):
        pass

    async def action(
        self,
        body: dict,
        __event_emitter__=None,
        __event_call__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ):
        if __event_emitter__:
            last_assistant_message = body["messages"][-1]
            model_name = body["model"]

            try:
                # Extract tables from the message
                message_content = last_assistant_message["content"]
                tables = self.get_tables_from_text(text=message_content)
                if not tables:
                    raise HTTPException(status_code=400, detail="Таблицы не найдены")

                bytes_object = self.save_tables_to_bytes(tables)
                current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

                unique_name = f"Таблица_{model_name}_{current_time}.xlsx"

                if __event_call__:
                    base64_blob = base64.b64encode(bytes_object.getbuffer()).decode(
                        "utf-8"
                    )
                    # Send the blob directly to the frontend
                    # Credit to brunthaler sebastian and Fu-Jie
                    await __event_call__(
                        {
                            "type": "execute",
                            "data": {
                                "code": f"""
                                try {{
                                    const base64Data = "{base64_blob}";
                                    const binaryData = atob(base64Data);
                                    const arrayBuffer = new Uint8Array(binaryData.length);
                                    for (let i = 0; i < binaryData.length; i++) {{
                                        arrayBuffer[i] = binaryData.charCodeAt(i);
                                    }}
                                    const blob = new Blob([arrayBuffer], {{ type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" }});
                                    const filename = "{unique_name}";

                                    const url = URL.createObjectURL(blob);
                                    const a = document.createElement("a");
                                    a.style.display = "none";
                                    a.href = url;
                                    a.download = filename;
                                    document.body.appendChild(a);
                                    a.click();
                                    URL.revokeObjectURL(url);
                                    document.body.removeChild(a);
                                }} catch (error) {{
                                    console.error('Error triggering download:', error);
                                }}
                                """
                            },
                        }
                    )

                return {"message": "Загрузка файла инициирована"}

            except Exception as e:
                await __event_emitter__(
                    {
                        "type": "notification",
                        "data": {
                            "type": "info",  # "success", "warning", "error"
                            "content": str(e),
                        },
                    }
                )
                print(f"Ошибка при обработке таблиц: {str(e)}")
                raise HTTPException(
                    status_code=500, detail=f"Ошибка при обработке таблиц: {str(e)}"
                )

    def save_tables_to_bytes(self, tables: list[pd.DataFrame]) -> BytesIO:
        excel_buffer = BytesIO()

        # Записываем DataFrame в буфер как Excel файл
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            for idx, table in enumerate(tables):
                table.to_excel(writer, sheet_name=f"таблица_{idx}", index=False)

        # Важно: переместить указатель в начало буфера
        excel_buffer.seek(0)

        return excel_buffer

    def get_tables_from_text(self, text: str) -> list[pd.DataFrame]:
        # Регулярное выражение для поиска таблиц
        # Ищет строки, начинающиеся и заканчивающиеся |, с разделителями | внутри
        table_pattern = r"(?m)(^\s*\|.*?\|\s*$\n?)+\s*"
        # Регулярное выражение для строки заголовков (дефисы или =, минимум 2)
        header_separator_pattern = r"^\s*\|(?:\s*-{2,}\s*\|)+\s*$"

        dataframes = []
        tables = re.finditer(table_pattern, text)

        for table_idx, table_match in enumerate(tables):
            table_text = table_match.group(0).strip().splitlines()
            if len(table_text) < 2:  # Пропускаем, если меньше 2 строк
                print(f"Таблица {table_idx+1} пропущена: слишком мало строк")
                continue

            # Проверяем наличие строки заголовков
            if not re.match(header_separator_pattern, table_text[1]):
                print(f"Таблица {table_idx+1} пропущена: некорректный заголовок")
                continue

            # Извлекаем заголовки (первая строка)
            headers = [h.strip() for h in table_text[0].strip("|").split("|")]
            headers = [h for h in headers if h]  # Удаляем пустые заголовки

            # Извлекаем строки данных (все, кроме первой и второй)
            rows = []
            for row in table_text[2:]:
                cells = [c.strip() for c in row.strip("|").split("|")]
                cells = [c for c in cells if c]  # Удаляем пустые ячейки
                if len(cells) == len(
                    headers
                ):  # Проверяем соответствие количества ячеек
                    rows.append(cells)

            # Если есть заголовки и строки, создаём DataFrame
            if headers and rows:
                df = pd.DataFrame(rows, columns=headers)
                dataframes.append(df)

        return dataframes
