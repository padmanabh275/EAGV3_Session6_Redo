from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from typing import Callable

from action import McpStdioClient, default_mcp_command, run_action
from decision import run_decision
from llm_gatewayV3.client import LLM
from memory import clear_state, recall, store_fact
from perception import run_perception
from schemas import (
    ActionInput,
    AgentIterationTrace,
    AgentRunResult,
    DecisionInput,
    MemoryInput,
    PerceptionInput,
)


@dataclass(frozen=True)
class TargetQuery:
    query_id: str
    prompt: str
    expected_answer_contains: str
    expected_iterations: int


TARGETS = {
    "A": TargetQuery(
        query_id="A",
        prompt=(
            "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his "
            "birth date, death date, and three key contributions to information theory."
        ),
        expected_answer_contains="birth date",
        expected_iterations=3,
    ),
    "B": TargetQuery(
        query_id="B",
        prompt=(
            "Find 3 family-friendly things to do in Tokyo this weekend. "
            "Check Saturday's weather forecast there and tell me which one is most appropriate."
        ),
        expected_answer_contains="most appropriate",
        expected_iterations=3,
    ),
    "C_WRITE": TargetQuery(
        query_id="C_WRITE",
        prompt=(
            "My mom's birthday is 15 May 2026. Remember that and give me "
            "a calendar reminder for two weeks before and on the day."
        ),
        expected_answer_contains="reminder",
        expected_iterations=1,
    ),
    "C_READ": TargetQuery(
        query_id="C_READ",
        prompt="When is mom's birthday?",
        expected_answer_contains="15 may 2026",
        expected_iterations=2,
    ),
    "D": TargetQuery(
        query_id="D",
        prompt=(
            "Search for 'Python asyncio best practices', read the top 3 results, "
            "and give me a short numbered list of the advice they agree on."
        ),
        expected_answer_contains="asyncio",
        expected_iterations=5,
    ),
}
ALL_QUERY_SEQUENCE = ["A", "B", "C_WRITE", "C_READ", "D"]


def _max_allowed(expected_iterations: int) -> int:
    return max(1, expected_iterations * 2)


def _contains_equivalent_date(answer: str, expected_substring: str) -> bool:
    months = {
        "january": "01",
        "february": "02",
        "march": "03",
        "april": "04",
        "may": "05",
        "june": "06",
        "july": "07",
        "august": "08",
        "september": "09",
        "october": "10",
        "november": "11",
        "december": "12",
    }

    def _extract_date_key(text: str) -> str | None:
        t = re.sub(r"[,./-]", " ", text.lower())
        # 15 may 2026
        m1 = re.search(
            r"\b(\d{1,2})\s+("
            + "|".join(months.keys())
            + r")\s+(\d{4})\b",
            t,
        )
        if m1:
            day = int(m1.group(1))
            month = months[m1.group(2)]
            year = m1.group(3)
            return f"{year}-{month}-{day:02d}"

        # may 15 2026
        m2 = re.search(
            r"\b("
            + "|".join(months.keys())
            + r")\s+(\d{1,2})\s+(\d{4})\b",
            t,
        )
        if m2:
            month = months[m2.group(1)]
            day = int(m2.group(2))
            year = m2.group(3)
            return f"{year}-{month}-{day:02d}"
        return None

    expected_key = _extract_date_key(expected_substring)
    answer_key = _extract_date_key(answer)
    return bool(expected_key and answer_key and expected_key == answer_key)


def _did_pass(answer: str, expected_substring: str, iterations: int, expected_iterations: int) -> bool:
    if expected_substring.lower() not in answer.lower() and not _contains_equivalent_date(answer, expected_substring):
        return False
    return iterations <= _max_allowed(expected_iterations)


def run_agent(
    query: str,
    max_iterations: int = 6,
    gateway_url: str = "http://localhost:8101",
    step_logger: Callable[[str], None] | None = None,
) -> AgentRunResult:
    llm = LLM(base_url=gateway_url)
    mcp = McpStdioClient(default_mcp_command())
    traces: list[AgentIterationTrace] = []
    observations: list[str] = []
    final_answer = ""
    known_facts: list[str] = []
    try:
        for i in range(1, max_iterations + 1):
            if step_logger:
                step_logger(f"--- iter {i} ---")
            perception = run_perception(
                llm,
                PerceptionInput(
                    query=query,
                    iteration=i,
                    max_iterations=max_iterations,
                    memory_facts=known_facts,
                    observations=observations,
                ),
            )
            memory_out = recall(MemoryInput(query=query, perception=perception))
            if step_logger:
                hits = len(memory_out.recalled_facts)
                step_logger(f"[memory.read]   {hits} hits")
            known_facts = memory_out.recalled_facts
            if step_logger:
                step_logger(f"[perception]    intent={perception.intent} needs_tool={perception.needs_tool}")
            decision = run_decision(
                llm,
                DecisionInput(
                    query=query,
                    iteration=i,
                    max_iterations=max_iterations,
                    perception=perception,
                    memory=memory_out,
                    observations=observations,
                ),
            )
            action = run_action(
                llm,
                mcp,
                ActionInput(
                    query=query,
                    iteration=i,
                    decision=decision,
                    memory=memory_out,
                    observations=observations,
                ),
            )
            observations.append(action.observation)
            if step_logger:
                if decision.step_type == "call_tool" and decision.tool_call is not None:
                    step_logger(
                        f"[decision]      TOOL_CALL: {decision.tool_call.name}({json.dumps(decision.tool_call.arguments, ensure_ascii=True)})"
                    )
                    step_logger(f"[action]        -> {action.observation[:220]}")
                else:
                    step_logger(f"[decision]      ANSWER: {(decision.draft_answer or '').strip()[:220]}")

            if perception.needs_memory_write and perception.memory_write_text:
                store_fact(perception.memory_write_text)
                observations.append(f"stored_memory:{perception.memory_write_text}")

            traces.append(
                AgentIterationTrace(
                    iteration=i,
                    perception=perception,
                    memory=memory_out,
                    decision=decision,
                    action=action,
                )
            )
            if action.done and action.final_answer:
                final_answer = action.final_answer
                if step_logger:
                    step_logger("[done] goal satisfied")
                break
    finally:
        mcp.close()

    if not final_answer:
        final_answer = "Unable to converge within iteration limit."
    return AgentRunResult(query=query, iterations=len(traces), answer=final_answer, trace=traces)


def run_target(target_id: str, gateway_url: str = "http://localhost:8101") -> dict:
    if target_id not in TARGETS:
        raise ValueError(f"Unknown target id: {target_id}")
    target = TARGETS[target_id]
    result = run_agent(
        query=target.prompt,
        max_iterations=_max_allowed(target.expected_iterations),
        gateway_url=gateway_url,
    )
    passed = _did_pass(
        answer=result.answer,
        expected_substring=target.expected_answer_contains,
        iterations=result.iterations,
        expected_iterations=target.expected_iterations,
    )
    return {
        "query_id": target.query_id,
        "prompt": target.prompt,
        "expected_answer_contains": target.expected_answer_contains,
        "expected_iterations": target.expected_iterations,
        "max_allowed_iterations": _max_allowed(target.expected_iterations),
        "iterations": result.iterations,
        "answer": result.answer,
        "passed": passed,
        "trace": [item.model_dump() for item in result.trace],
    }


def run_all_targets(gateway_url: str = "http://localhost:8101") -> dict:
    runs = [run_target(tid, gateway_url=gateway_url) for tid in ALL_QUERY_SEQUENCE]
    return {"overall_pass": all(r["passed"] for r in runs), "results": runs}


def run_all_targets_steps(gateway_url: str = "http://localhost:8101") -> int:
    overall_pass = True
    for tid in ALL_QUERY_SEQUENCE:
        target = TARGETS[tid]
        print(f"Query {target.query_id}. {target.prompt}")
        result = run_agent(
            query=target.prompt,
            max_iterations=_max_allowed(target.expected_iterations),
            gateway_url=gateway_url,
            step_logger=print,
        )
        passed = _did_pass(
            answer=result.answer,
            expected_substring=target.expected_answer_contains,
            iterations=result.iterations,
            expected_iterations=target.expected_iterations,
        )
        overall_pass = overall_pass and passed
        print(f"FINAL: {result.answer}")
        print(f"[result] passed={passed} iterations={result.iterations}/{_max_allowed(target.expected_iterations)}")
        print("")
    print(f"Overall pass: {overall_pass}")
    return 0 if overall_pass else 1


def _main() -> int:
    parser = argparse.ArgumentParser(description="Session 6 cognitive architecture agent.")
    parser.add_argument("--query", type=str, required=False, help="Ad-hoc user query text.")
    parser.add_argument("--target", choices=list(TARGETS.keys()), help="Run one assignment target query.")
    parser.add_argument("--all-targets", action="store_true", help="Run all assignment target queries.")
    parser.add_argument("--all-targets-steps", action="store_true", help="Run all target queries and print live step logs.")
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument("--gateway-url", type=str, default="http://localhost:8101")
    parser.add_argument("--clear-state", action="store_true")
    args = parser.parse_args()

    if args.clear_state:
        clear_state()
        print("State cleared.")
        return 0

    if args.all_targets:
        out = run_all_targets(gateway_url=args.gateway_url)
        print(json.dumps(out, indent=2))
        return 0 if out["overall_pass"] else 1

    if args.all_targets_steps:
        return run_all_targets_steps(gateway_url=args.gateway_url)

    if args.target:
        out = run_target(args.target, gateway_url=args.gateway_url)
        print(json.dumps(out, indent=2))
        return 0 if out["passed"] else 1

    if not args.query:
        print("Pass --query '...' or --target <ID> or --all-targets", file=sys.stderr)
        return 2

    result = run_agent(args.query, max_iterations=args.max_iterations, gateway_url=args.gateway_url)
    print(json.dumps(result.model_dump(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
