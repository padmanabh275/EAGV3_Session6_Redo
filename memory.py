from __future__ import annotations

import json
from pathlib import Path

from schemas import MemoryInput, MemoryOutput

STATE_DIR = Path(__file__).parent / "state"
MEMORY_FILE = STATE_DIR / "memory.json"


def _ensure_state() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text(json.dumps({"facts": []}, indent=2), encoding="utf-8")


def _load_memory() -> list[str]:
    _ensure_state()
    try:
        raw = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raw = {"facts": []}
    facts = raw.get("facts", [])
    if not isinstance(facts, list):
        return []
    return [str(f) for f in facts]


def _save_memory(facts: list[str]) -> None:
    _ensure_state()
    MEMORY_FILE.write_text(json.dumps({"facts": facts}, indent=2), encoding="utf-8")


def recall(input_data: MemoryInput) -> MemoryOutput:
    facts = _load_memory()
    q = input_data.query.lower()
    scored: list[tuple[int, str]] = []
    for fact in facts:
        f = fact.lower()
        score = 0
        for token in q.split():
            if len(token) >= 4 and token in f:
                score += 1
        if score > 0:
            scored.append((score, fact))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = [fact for _, fact in scored[: input_data.max_results]]
    return MemoryOutput(recalled_facts=out, memory_hit=bool(out))


def store_fact(text: str) -> None:
    cleaned = text.strip()
    if not cleaned:
        return
    facts = _load_memory()
    if cleaned not in facts:
        facts.append(cleaned)
        _save_memory(facts)


def clear_state() -> None:
    _ensure_state()
    _save_memory([])
