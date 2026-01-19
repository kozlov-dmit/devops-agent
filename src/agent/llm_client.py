from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class LLMConfig:
    credentials: str
    scope: Optional[str] = None
    verify_ssl_certs: bool = True
    temperature: float = 0.1


def load_llm_config() -> LLMConfig:
    credentials = os.getenv("GIGACHAT_CREDENTIALS", "MGFlN2NiMjYtM2RmYy00NDkzLWI0OWUtNjgzOTdkNTdhYzUzOmI4NDUyYzdkLTg1YjYtNDM5OC1iMDM1LTFmMzVlNGZhNTgzMQ==").strip()
    logging.info("credentials: %s", credentials)
    if not credentials:
        raise ValueError("GIGACHAT_CREDENTIALS is not set")

    return LLMConfig(
        credentials=credentials,
        scope=os.getenv("GIGACHAT_SCOPE"),
        verify_ssl_certs=os.getenv("GIGACHAT_VERIFY_SSL_CERTS", "false").lower() == "true",
        temperature=float(os.getenv("GIGACHAT_TEMPERATURE", "0.1")),
    )


class LLMClient:
    """
    Корректная обёртка над официальным gigachat SDK.
    Без передачи неподдерживаемых kwargs.
    """

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        from gigachat import GigaChat  # импортируем лениво
        self._GigaChat = GigaChat

    def chat_text(self, *, system: str, user: str) -> str:
        prompt = f"SYSTEM:\n{system}\n\nUSER:\n{user}"

        with self._GigaChat(
            credentials=self.cfg.credentials,
            scope=self.cfg.scope,
            verify_ssl_certs=self.cfg.verify_ssl_certs,
        ) as client:
            response = client.chat(
                prompt,
                # temperature=self.cfg.temperature,
            )

        return response.choices[0].message.content
