"""
title: Markdown Table to Excel
author: Sergei Vyaznikov
description: Конвертирует LaTeX в .docx
icon_url: data:image/png;base64,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
version: 0.1
requirements: fastapi, tex2docx
"""

import os
import re
import base64
import tempfile
from tex2docx import LatexToWordConverter

from datetime import datetime
from typing import Optional, Callable, Awaitable, Any
from fastapi import HTTPException

# По умолчанию tex2docx использует pandoc-crossref
# Отключаем фильтры, чтобы LatexToWordConverter не жаловался
from tex2docx.constants import PandocOptions

PandocOptions.FILTER_OPTIONS = []


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
                message_content = last_assistant_message["content"]
                latex_code = self.get_latex_from_text(message_content)

                if not latex_code:
                    raise HTTPException(
                        status_code=400, detail="Не обнаружена разметка LaTeX"
                    )

                with tempfile.NamedTemporaryFile(
                    mode="w", delete=False, encoding="utf-8"
                ) as temp_tex_file:
                    temp_tex_file.write(latex_code)
                    tex_file_path = temp_tex_file.name

                with tempfile.NamedTemporaryFile(
                    mode="w", delete=False, encoding="utf-8"
                ) as temp_docx_file:
                    docx_file_path = temp_docx_file.name

                config = {
                    "input_texfile": tex_file_path,
                    "output_docxfile": docx_file_path,
                }
                converter = LatexToWordConverter(**config)
                converter.convert()

                with open(docx_file_path, "rb") as docx_file:
                    docx_data = docx_file.read()
                    base64_blob = base64.b64encode(docx_data).decode("utf-8")

                current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                unique_name = f"Документ_{model_name}_{current_time}.docx"

                if __event_call__:
                    # Send the blob directly to the frontend
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
                                    const blob = new Blob([arrayBuffer], {{ type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document" }});
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
                print(f"Ошибка при обработке документа: {str(e)}")
                raise HTTPException(
                    status_code=500, detail=f"Ошибка при обработке документа: {str(e)}"
                )

    def get_latex_from_text(self, text):
        pattern = re.compile(
            r"(?:.|\n)*```latex(?P<latex_code>(?:.|\n)+)```(?:.|\n)*", flags=0
        )
        match = pattern.search(text)
        if match:
            return match.groupdict().get("latex_code")
        else:
            return None
