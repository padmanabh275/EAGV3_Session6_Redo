from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


RoleName = Literal["perception", "memory", "decision", "action"]
ToolName = Literal[
    "web_search",
    "fetch_url",
    "get_time",
    "currency_convert",
    "read_file",
    "list_dir",
    "create_file",
    "update_file",
    "edit_file",
]


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class PerceptionInput(BaseModel):
    query: str
    iteration: int
    max_iterations: int
    memory_facts: list[str] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)


class PerceptionOutput(BaseModel):
    intent: str
    salient_entities: list[str] = Field(default_factory=list)
    needs_tool: bool = False
    needs_memory_write: bool = False
    memory_write_text: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class MemoryInput(BaseModel):
    query: str
    perception: PerceptionOutput
    max_results: int = 5


class MemoryOutput(BaseModel):
    recalled_facts: list[str] = Field(default_factory=list)
    memory_hit: bool = False


class DecisionInput(BaseModel):
    query: str
    iteration: int
    max_iterations: int
    perception: PerceptionOutput
    memory: MemoryOutput
    observations: list[str] = Field(default_factory=list)


class ToolCall(BaseModel):
    name: ToolName
    arguments: dict[str, Any] = Field(default_factory=dict)


class DecisionOutput(BaseModel):
    step_type: Literal["call_tool", "answer_final"]
    rationale: str
    tool_call: ToolCall | None = None
    draft_answer: str | None = None


class ActionInput(BaseModel):
    query: str
    iteration: int
    decision: DecisionOutput
    memory: MemoryOutput
    observations: list[str] = Field(default_factory=list)


class ActionOutput(BaseModel):
    observation: str
    final_answer: str | None = None
    tool_result: dict[str, Any] | None = None
    done: bool = False


class AgentIterationTrace(BaseModel):
    iteration: int
    perception: PerceptionOutput
    memory: MemoryOutput
    decision: DecisionOutput
    action: ActionOutput


class AgentRunResult(BaseModel):
    query: str
    iterations: int
    answer: str
    trace: list[AgentIterationTrace] = Field(default_factory=list)
