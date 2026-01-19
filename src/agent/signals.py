from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set


# Stacktrace
STACK_FRAME_RE = re.compile(r"\bat\s+([a-zA-Z_][\w$]*(?:\.[a-zA-Z_][\w$]*)+)\(([^)]*)\)")
# Endpoint-like: /api/payments/confirm
ENDPOINT_RE = re.compile(r"(/api/[a-zA-Z0-9/_\-\.]+)")
# Exception-like tokens: SomethingException / Error
EXC_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]+(?:Exception|Error))\b")
# Common infra/perf tokens
KEY_TOKENS_RE = re.compile(
    r"\b(HikariPool|Hikari|Timeout|timeout|timed out|Connection is not available|PSQLException|deadlock|lock|"
    r"retry|retries|circuit|throttle|rate limit|OOM|OutOfMemory|GC|pause|latency|p99|p95)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class IncidentSignals:
    endpoints: Set[str]
    exceptions: Set[str]
    frames: Set[str]          # fqcn.method or fqcn
    keywords: Set[str]


def extract_signals(text: str) -> IncidentSignals:
    endpoints = set(m.group(1) for m in ENDPOINT_RE.finditer(text))

    exceptions = set(m.group(1) for m in EXC_RE.finditer(text))

    frames: Set[str] = set()
    for m in STACK_FRAME_RE.finditer(text):
        fq = m.group(1)
        frames.add(fq)

    keywords = set(m.group(0) for m in KEY_TOKENS_RE.finditer(text))

    # normalize (lowercase keywords for matching)
    keywords_norm = set(k.lower() for k in keywords)
    return IncidentSignals(
        endpoints=endpoints,
        exceptions=exceptions,
        frames=frames,
        keywords=keywords_norm,
    )


def score_chunk_text(chunk_text: str, signals: IncidentSignals) -> float:
    """
    Lightweight reranker: add points if chunk contains important tokens.
    Keep it simple and deterministic.
    """
    text = chunk_text.lower()
    score = 0.0

    # keywords are usually strongest
    for kw in signals.keywords:
        if kw and kw in text:
            score += 1.0

    # exceptions
    for exc in signals.exceptions:
        if exc and exc.lower() in text:
            score += 1.5

    # stack frames (fqcn/method)
    for fr in signals.frames:
        # match by last segment to avoid missing due to imports / formatting
        last = fr.split(".")[-1].lower()
        if last and last in text:
            score += 0.75

    # endpoints
    for ep in signals.endpoints:
        if ep and ep.lower() in text:
            score += 1.25

    return score


def path_penalty(path: str) -> float:
    """
    Penalize low-value paths for performance root-cause analysis.
    We do NOT drop them completely; we just lower the ranking.
    """
    p = path.replace("\\", "/").lower()

    # tests are almost always noise
    if "/src/test/" in p or p.endswith("test.java") or p.endswith("tests.java") or p.endswith("it.java"):
        return -3.0

    # DTO/model noise
    if "/model/" in p or "/dto/" in p:
        return -1.0

    # Encourage config and service layers
    bonus = 0.0
    if "/config/" in p:
        bonus += 0.5
    if "/service/" in p or "/repository/" in p or "/client/" in p:
        bonus += 0.3
    if p.endswith("application.yml") or p.endswith("application.yaml") or p.endswith("application.properties"):
        bonus += 0.8

    return bonus
