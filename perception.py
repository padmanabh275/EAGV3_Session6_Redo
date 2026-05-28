from __future__ import annotations

import time

from llm_gatewayV3.client import LLM

from schemas import PerceptionInput, PerceptionOutput


SYSTEM_PROMPT = (
    "You are the Perception layer of a 4-layer cognitive architecture. "
    "Read the user query and context, then produce a compact structured analysis. "
    "If the user asks to remember/save/store a fact, set needs_memory_write=true and include the exact fact text. "
    "Output must be valid JSON only."
)


PROVIDER_FALLBACK_ORDER = [
    "openai",
    "ollama",
]
RETRY_ROUNDS = 1


def run_perception(llm: LLM, input_data: PerceptionInput) -> PerceptionOutput:
    schema = PerceptionOutput.model_json_schema()
    payload = (
        f"Iteration {input_data.iteration}/{input_data.max_iterations}\n"
        f"Query:\n{input_data.query}\n\n"
        f"Known memory facts:\n{input_data.memory_facts}\n\n"
        f"Observations so far:\n{input_data.observations}\n"
    )
    last_error: Exception | None = None
    for round_idx in range(RETRY_ROUNDS):
        for provider in PROVIDER_FALLBACK_ORDER:
            try:
                resp = llm.chat(
                    messages=[{"role": "user", "content": payload}],
                    system=SYSTEM_PROMPT,
                    provider=provider,
                    response_format={"type": "json_schema", "schema": schema, "name": "perception_out", "strict": True},
                )
                parsed = resp.get("parsed") or {}
                return PerceptionOutput.model_validate(parsed)
            except Exception as exc:  # fallback on gateway/provider failures (e.g. 502/503)
                last_error = exc
                continue
        if round_idx < RETRY_ROUNDS - 1:
            time.sleep(1.5 * (round_idx + 1))

    # Keep the run alive during temporary gateway outages.
    if input_data.observations:
        return PerceptionOutput(
            intent="degraded_mode",
            salient_entities=[],
            needs_tool=False,
            needs_memory_write=False,
            memory_write_text=None,
            confidence=0.0,
        )
    raise RuntimeError(f"Perception failed across providers: {last_error}")
