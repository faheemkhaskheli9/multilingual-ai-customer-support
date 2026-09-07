"""FastAPI chat surface.

``POST /chat`` runs one agent turn. Conversation persistence (issue #4) is out
of scope for Phase 1's issue #1 — each request is stateless.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .agent import SupportAgent

app = FastAPI(title="Multilingual Customer Support AI", version="0.1.0")
_agent = SupportAgent()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


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
    result = _agent.run(request.message)
    return ChatResponse(
        reply=result.reply,
        stop_reason=result.stop_reason,
        iterations=result.iterations,
        tool_calls=list(result.tool_calls),
    )
