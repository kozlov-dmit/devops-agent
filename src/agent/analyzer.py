from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .llm_client import LLMClient
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .report_schema import extract_json_from_text, validate_report


@dataclass(frozen=True)
class ContextItem:
    score: float
    base: float
    rr: float
    path: str
    start_line: int
    end_line: int
    language: str
    text: str


def analyze_incident_with_llm(
    *,
    llm: LLMClient,
    incident: Dict[str, Any],
    contexts: List[ContextItem],
    temperature: float = 0.1,
) -> Dict[str, Any]:
    # temperature передаём через env/config в GigaChat wrapper (в MVP — достаточно)
    user_prompt = build_user_prompt(
        incident=incident,
        contexts=[c.__dict__ for c in contexts],
    )

    content = llm.chat_text(system=SYSTEM_PROMPT, user=user_prompt)

    report = extract_json_from_text(content)
    validate_report(report)
    return report
