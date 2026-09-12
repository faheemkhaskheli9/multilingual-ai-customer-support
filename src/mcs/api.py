"""FastAPI chat surface.

``POST /chat`` runs one agent turn. Durable conversation persistence
(PostgreSQL, issue #3/#4) is still out of scope — this in-process
``_sessions`` map is deliberately just working memory (see the
knowledge-base "working-memory" pattern): it lets a follow-up like "what
about the tax on that?" resolve within a session's lifetime, but it is not a
store — it is empty again on process restart, and unbounded growth here is a
known Phase-1 limitation to be replaced by real persistence in a later issue.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .agent import SupportAgent

app = FastAPI(title="Multilingual Customer Support AI", version="0.1.0")
_agent = SupportAgent()
_sessions: dict[str, tuple[dict[str, object], ...]] = {}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(
        default=None,
        description="Opaque id to thread multi-turn context. Omit for a stateless, one-off request.",
    )


class ChatResponse(BaseModel):
    reply: str
    stop_reason: str
    iterations: int
    tool_calls: list[str]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    history = list(_sessions.get(request.session_id, ())) if request.session_id else None
    result = _agent.run(request.message, history=history)
    if request.session_id:
        _sessions[request.session_id] = result.history
    return ChatResponse(
        reply=result.reply,
        stop_reason=result.stop_reason,
        iterations=result.iterations,
        tool_calls=list(result.tool_calls),
    )
