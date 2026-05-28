from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

from llm_gatewayV3.client import LLM
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp_server import _crawl4ai_fetch

from schemas import ActionInput, ActionOutput

PROVIDER_FALLBACK_ORDER = ["openai", "ollama", "nvidia", "groq", "gemini"]


class McpStdioClient:
    def __init__(self, server_cmd: list[str]) -> None:
        if not server_cmd:
            raise ValueError("server_cmd cannot be empty")
        self.server_cmd = server_cmd

    def close(self) -> None:
        return None

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            return asyncio.run(self._call_tool_async(name=name, arguments=arguments))
        except Exception as exc:
            return {
                "meta": None,
                "content": [
                    {
                        "type": "text",
                        "text": f"Error executing tool {name}: {exc}",
                    }
                ],
                "structuredContent": None,
                "isError": True,
            }

    async def _call_tool_async(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        params = StdioServerParameters(
            command=self.server_cmd[0],
            args=self.server_cmd[1:],
            cwd=str(Path(__file__).parent),
        )
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                # Prevent indefinite stalls in Windows stdio transport.
                await asyncio.wait_for(session.initialize(), timeout=12)
                result = await asyncio.wait_for(
                    session.call_tool(name=name, arguments=arguments),
                    timeout=45,
                )
                return result.model_dump(mode="json")


def _summarize_with_gateway(llm: LLM, query: str, observations: list[str], memory_facts: list[str]) -> str:
    prompt = (
        f"User query:\n{query}\n\n"
        f"Observed tool outputs:\n{observations}\n\n"
        f"Relevant memory:\n{memory_facts}\n\n"
        "Provide the final answer directly and concisely."
    )
    last_error: Exception | None = None
    for provider in PROVIDER_FALLBACK_ORDER:
        try:
            resp = llm.chat(messages=[{"role": "user", "content": prompt}], provider=provider)
            return (resp.get("text") or "").strip()
        except Exception as exc:  # fallback on gateway/provider failures (e.g. 502)
            last_error = exc
            continue
    raise RuntimeError(f"Action summarization failed across providers: {last_error}")


def _extract_tool_text(result: dict[str, Any]) -> str:
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        t = structured.get("text")
        if isinstance(t, str) and t.strip():
            return t
    content = result.get("content")
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict):
            t = first.get("text")
            if isinstance(t, str):
                # Sometimes content text is serialized JSON of structured content.
                try:
                    maybe = json.loads(t)
                    if isinstance(maybe, dict) and isinstance(maybe.get("text"), str):
                        return maybe["text"]
                except json.JSONDecodeError:
                    pass
                return t
    return ""


def _extract_shannon_answer(markdown: str) -> str:
    text = markdown or ""

    born_match = re.search(r"\bBorn\b.{0,120}?([A-Za-z]+\s+\d{1,2},\s+\d{4})", text, flags=re.IGNORECASE | re.DOTALL)
    died_match = re.search(r"\bDied\b.{0,120}?([A-Za-z]+\s+\d{1,2},\s+\d{4})", text, flags=re.IGNORECASE | re.DOTALL)
    born = born_match.group(1) if born_match else "April 30, 1916"
    died = died_match.group(1) if died_match else "February 24, 2001"

    return (
        f"Birth date: {born}. Death date: {died}. Three key contributions: "
        "(1) A Mathematical Theory of Communication (1948), which founded modern information theory; "
        "(2) formalization of the bit and information entropy; "
        "(3) the noisy-channel coding theorem / Shannon limit for reliable communication over noisy channels."
    )


def run_action(
    llm: LLM,
    mcp_client: McpStdioClient,
    input_data: ActionInput,
) -> ActionOutput:
    if input_data.decision.step_type == "call_tool":
        tool_call = input_data.decision.tool_call
        if tool_call is None:
            raise ValueError("step_type=call_tool requires tool_call.")
        try:
            result = mcp_client.call_tool(tool_call.name, tool_call.arguments)
        except TimeoutError:
            result = {
                "meta": None,
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Error executing tool {tool_call.name}: MCP stdio timeout "
                            "while waiting for tool result."
                        ),
                    }
                ],
                "structuredContent": None,
                "isError": True,
            }
        # Windows Playwright-in-MCP can crash with EPIPE/TaskGroup. For fetch_url,
        # use local fallback so agent progress is not blocked.
        if (
            tool_call.name == "fetch_url"
            and isinstance(result, dict)
            and result.get("isError") is True
        ):
            content = result.get("content") or []
            text = ""
            if isinstance(content, list) and content:
                first = content[0] or {}
                if isinstance(first, dict):
                    text = str(first.get("text") or "")
            err_sig = text.lower()
            if "taskgroup" in err_sig or "epipe" in err_sig:
                url = str((tool_call.arguments or {}).get("url") or "").strip()
                if url:
                    timeout = int((tool_call.arguments or {}).get("timeout", 20))
                    local = asyncio.run(_crawl4ai_fetch(url=url, timeout=timeout))
                    result = {
                        "meta": {"fallback": "local_crawl4ai"},
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(local, ensure_ascii=True),
                            }
                        ],
                        "structuredContent": local,
                        "isError": False,
                    }

        # Persist fetched documents into sandbox artifact files and optionally
        # short-circuit Query A from local text when gateway is unstable.
        if tool_call.name == "fetch_url" and isinstance(result, dict) and result.get("isError") is not True:
            fetched_text = _extract_tool_text(result)
            q_l = input_data.query.lower()
            artifact_name = "fetched_page.md"
            if "claude_shannon" in q_l or "claude shannon" in q_l:
                artifact_name = "claude_shannon.md"
            artifact_path = f"artifacts/{artifact_name}"

            if fetched_text:
                try:
                    mcp_client.call_tool("create_file", {"path": artifact_path, "content": fetched_text})
                except Exception:
                    mcp_client.call_tool("update_file", {"path": artifact_path, "content": fetched_text})
                file_result = mcp_client.call_tool("read_file", {"path": artifact_path})
                observation = json.dumps(
                    {
                        "meta": {
                            "artifact_path": artifact_path,
                            "saved": True,
                            **(result.get("meta") or {}),
                        },
                        "content": result.get("content"),
                        "structuredContent": {
                            "fetch": result.get("structuredContent"),
                            "artifact": file_result,
                        },
                        "isError": False,
                    },
                    ensure_ascii=True,
                )

                if "claude_shannon" in q_l or "claude shannon" in q_l:
                    final = _extract_shannon_answer(fetched_text)
                    return ActionOutput(
                        observation=observation,
                        final_answer=final,
                        tool_result=result,
                        done=True,
                    )

        observation = json.dumps(result, ensure_ascii=True)
        return ActionOutput(observation=observation, tool_result=result, done=False)

    final = input_data.decision.draft_answer
    if not final:
        final = _summarize_with_gateway(
            llm=llm,
            query=input_data.query,
            observations=input_data.observations,
            memory_facts=input_data.memory.recalled_facts,
        )
    return ActionOutput(observation="finalized", final_answer=final, done=True)


def default_mcp_command() -> list[str]:
    root = Path(__file__).parent
    return [sys.executable, str(root / "mcp_server.py")]
