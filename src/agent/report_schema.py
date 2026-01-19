from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Минимальный JSON-repair: пытаемся вытащить первый JSON-объект из ответа.
    """
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    # Найдём первую '{' и последнюю '}'
    i = text.find("{")
    j = text.rfind("}")
    if i >= 0 and j > i:
        return json.loads(text[i:j + 1])

    raise ValueError("No JSON object found in LLM response")


def validate_report(report: Dict[str, Any]) -> None:
    """
    MVP-валидация: проверяем ключевые поля, чтобы не получать мусор.
    """
    required_top = ["summary", "classification", "hypotheses", "hotspots", "checks"]
    for k in required_top:
        if k not in report:
            raise ValueError(f"Missing field: {k}")

    if not isinstance(report["hypotheses"], list):
        raise ValueError("hypotheses must be a list")
    if not isinstance(report["hotspots"], list):
        raise ValueError("hotspots must be a list")
    if not isinstance(report["checks"], list):
        raise ValueError("checks must be a list")
