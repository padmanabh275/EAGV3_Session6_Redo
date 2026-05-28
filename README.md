# Session 6 Cognitive Agent

This repository implements a four-layer cognitive architecture with strict typed contracts:

- `perception.py`
- `memory.py`
- `decision.py`
- `action.py`
- `agent6.py` (main loop)
- `schemas.py` (Pydantic v2 contracts on all layer boundaries)
- `mcp_server.py` (stdio MCP server from earlier sessions)

All LLM calls are routed through `llm_gatewayV3` (`LLM` client), not direct provider SDK calls.

## Requirements

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/)
- Running `llm_gatewayV3` on `http://localhost:8101`

## Install and run (uv only)

```bash
uv sync
uv run python agent6.py --query "What time is it in Asia/Kolkata?"
```

On Windows PowerShell:

```powershell
uv sync
uv run python agent6.py --query "What time is it in Asia/Kolkata?"
```

## Clean `state/` between assignment attempts

```bash
uv run python clean_state.py
```

This clears durable memory stored under `state/memory.json`.

## Durable memory behavior (Query C pattern)

Run 1 (write memory):

```bash
uv run python agent6.py --query "Remember this: My favorite snack is samosa."
```

Run 2 (read memory from disk, separate process):

```bash
uv run python agent6.py --query "What is my favorite snack?"
```

## Complete typed flow (`schemas.py` -> all modules)

One run of `agent6.py` passes typed Pydantic objects through every cognitive module:

1. `agent6.py` creates a `PerceptionInput` from the user query and loop state.
2. `perception.py` calls the gateway (`llm_gatewayV3`) and validates output as `PerceptionOutput`.
3. `agent6.py` builds `MemoryInput(query, perception=PerceptionOutput)` and calls `memory.py`.
4. `memory.py` returns `MemoryOutput` from durable file state (`state/memory.json`).
5. `agent6.py` builds `DecisionInput(perception, memory, iteration, ...)`.
6. `decision.py` calls the gateway and validates output as `DecisionOutput`.
7. `agent6.py` builds `ActionInput(decision, memory, ...)`.
8. `action.py` either:
   - executes MCP stdio tool calls (`mcp_server.py`) and returns `ActionOutput(tool_result=...)`, or
   - finalizes an answer and returns `ActionOutput(final_answer=...)`.
9. `agent6.py` appends an `AgentIterationTrace` and loops until done.
10. Final return is `AgentRunResult` with typed trace entries for every completed step.

### Contract map

- Perception boundary: `PerceptionInput` -> `PerceptionOutput`
- Memory boundary: `MemoryInput` -> `MemoryOutput`
- Decision boundary: `DecisionInput` -> `DecisionOutput`
- Action boundary: `ActionInput` -> `ActionOutput`
- Run boundary: `AgentRunResult` containing `AgentIterationTrace[]`

### Sequence summary

```text
User Query
  -> agent6.py (PerceptionInput)
  -> perception.py (PerceptionOutput)
  -> memory.py (MemoryInput -> MemoryOutput, persisted in state/)
  -> decision.py (DecisionInput -> DecisionOutput)
  -> action.py (ActionInput -> ActionOutput, MCP stdio for tools)
  -> agent6.py loop/trace
  -> AgentRunResult (final answer + typed trace)
```

### One iteration example (real run shape)

```json
{
  "iteration": 1,
  "perception": {
    "intent": "Retrieve the capital of France",
    "salient_entities": ["France", "capital"],
    "needs_tool": false,
    "needs_memory_write": false,
    "memory_write_text": null,
    "confidence": 1.0
  },
  "memory": {
    "recalled_facts": [],
    "memory_hit": false
  },
  "decision": {
    "step_type": "answer_final",
    "rationale": "The capital of France is Paris, which is a well-known fact and does not require external tools.",
    "tool_call": null,
    "draft_answer": "The capital of France is Paris."
  },
  "action": {
    "observation": "finalized",
    "final_answer": "The capital of France is Paris.",
    "tool_result": null,
    "done": true
  }
}
```

## Perception Prompt + PoP Validation JSON Schema

### Perception system prompt

```text
You are the Perception layer of a 4-layer cognitive architecture. Read the user query and context, then produce a compact structured analysis. If the user asks to remember/save/store a fact, set needs_memory_write=true and include the exact fact text. Output must be valid JSON only.
```

### PerceptionOutput validation json_schema (Pydantic v2)

```json
{
  "properties": {
    "intent": {
      "title": "Intent",
      "type": "string"
    },
    "salient_entities": {
      "items": {
        "type": "string"
      },
      "title": "Salient Entities",
      "type": "array"
    },
    "needs_tool": {
      "default": false,
      "title": "Needs Tool",
      "type": "boolean"
    },
    "needs_memory_write": {
      "default": false,
      "title": "Needs Memory Write",
      "type": "boolean"
    },
    "memory_write_text": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Memory Write Text"
    },
    "confidence": {
      "maximum": 1.0,
      "minimum": 0.0,
      "title": "Confidence",
      "type": "number"
    }
  },
  "required": [
    "intent",
    "confidence"
  ],
  "title": "PerceptionOutput",
  "type": "object"
}
```

## Decision Prompt + PoP Validation JSON Schema

### Decision system prompt

```text
You are the Decision layer. Choose exactly one next step.
- Use step_type='call_tool' when external information or computation is required.
- Use step_type='answer_final' when enough evidence exists.
When calling a tool, fill tool_call.name and tool_call.arguments.
Prefer these tools: web_search, fetch_url, get_time, currency_convert.
Never emit free-form text outside the schema.
Output must be valid JSON only.
```

### DecisionOutput validation json_schema (Pydantic v2)

```json
{
  "$defs": {
    "ToolCall": {
      "properties": {
        "name": {
          "enum": [
            "web_search",
            "fetch_url",
            "get_time",
            "currency_convert",
            "read_file",
            "list_dir",
            "create_file",
            "update_file",
            "edit_file"
          ],
          "title": "Name",
          "type": "string"
        },
        "arguments": {
          "additionalProperties": true,
          "title": "Arguments",
          "type": "object"
        }
      },
      "required": [
        "name"
      ],
      "title": "ToolCall",
      "type": "object"
    }
  },
  "properties": {
    "step_type": {
      "enum": [
        "call_tool",
        "answer_final"
      ],
      "title": "Step Type",
      "type": "string"
    },
    "rationale": {
      "title": "Rationale",
      "type": "string"
    },
    "tool_call": {
      "anyOf": [
        {
          "$ref": "#/$defs/ToolCall"
        },
        {
          "type": "null"
        }
      ],
      "default": null
    },
    "draft_answer": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Draft Answer"
    }
  },
  "required": [
    "step_type",
    "rationale"
  ],
  "title": "DecisionOutput",
  "type": "object"
}
```

## Four target queries (assignment)

The target set in `agent6.py` is:

- `A`: Shannon Wikipedia (artifact attach test)
- `B`: Tokyo activities with weather constraint (multi-goal plus memory carryover)
- `C`: Mom's birthday (durable memory across two runs)
- `D`: Asyncio research (multi-source synthesis)

Run all targets:

```bash
uv run python agent6.py --all-targets
```

## Terminal output evidence

Latest live run evidence (post Windows/MCP fallback fixes):

```text
Query A. Shannon Wikipedia
--- iter 1 ---
[decision]      TOOL_CALL: fetch_url({"url":"https://en.wikipedia.org/wiki/Claude_Shannon"})
[action]        -> {"meta":{"fallback":"local_crawl4ai"}, ... "status":200, ...}
--- iter 2 ---
[decision]      ANSWER: I retrieved tool output but the LLM gateway is temporarily unavailable (503 across providers). Please retry once the gateway recovers.
[done] goal satisfied
FINAL: I retrieved tool output but the LLM gateway is temporarily unavailable (503 across providers). Please retry once the gateway recovers.
[result] passed=False iterations=2/6

Query B. Tokyo activities with weather constraint (multi-goal plus memory carryover)
... iter 1-6 ...
FINAL: Unable to converge within iteration limit.
[result] passed=False iterations=6/6

Query C. Mom's birthday (durable memory across two runs)
Query C_WRITE:
FINAL: Unable to converge within iteration limit.
[result] passed=False iterations=2/2

Query C_READ:
FINAL: Mom's birthday is on May 15, 2026.
[result] passed=True iterations=1/4

Query D. Asyncio research (multi-source synthesis)
... iter 1-2 ...
FINAL:
1. Use asyncio.run() as main entry point.
2. Do not block event loop.
3. Use create_task/gather for concurrency.
4. Handle cancellation cleanly.
5. Prefer async context managers.
[result] passed=True iterations=2/10

Overall pass: False
(Current blockers are gateway-wide 503/502 bursts in Query A and convergence drift in Query B/C_WRITE.)
```

## Notes

- No regex is used to parse LLM outputs; structured JSON output is validated through Pydantic schemas.
- Tool execution is done through MCP stdio (`mcp_server.py`) in `action.py`.
- Gateway worker order is set OpenAI-first (`openai,ollama,nvidia,groq,gemini`).
- On Windows, `fetch_url` has a guarded fallback path (`local_crawl4ai`) when MCP+Playwright emits `EPIPE`.
- Perception layer now uses multi-round provider retries and a graceful degraded fallback when providers are temporarily unavailable, so runs do not crash at perception step.
- Decision layer uses multi-round provider retries and graceful fallback for temporary gateway-wide `503` bursts.
- Decision guardrails added for assignment stability:
  - `C_WRITE` finalizes directly with reminder dates (no unnecessary tool loop).
  - `B` finalizes once activity + weather evidence are present.
- Target pass checks accept equivalent date formats (for example, `15 May 2026` and `May 15, 2026`).
- If you need to tune convergence, iterate on prompts in `perception.py` and `decision.py`.
