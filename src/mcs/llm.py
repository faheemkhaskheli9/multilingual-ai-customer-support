"""LLM client abstraction and a dependency-free mock implementation.

The agent loop talks to an :class:`LLMClient`. Phase 1 ships
:class:`MockLLMClient`, a deterministic rule-based stand-in so the project runs
and tests pass with no API key and no network. Issue #4 adds a real
LangChain/OpenAI-backed client implementing the same protocol.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# --- message / response value objects ---------------------------------------


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class LLMResponse:
    """Exactly one of ``tool_calls`` (a pause) or ``final_text`` (done) is set."""

    tool_calls: tuple[ToolCall, ...] = ()
    final_text: str | None = None

    def __post_init__(self) -> None:
        if bool(self.tool_calls) == (self.final_text is not None):
            raise ValueError("LLMResponse must carry either tool_calls or final_text, not both/neither")


@runtime_checkable
class LLMClient(Protocol):
    def respond(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> LLMResponse: ...


# --- deterministic mock ----------------------------------------------------

_ORDER_INTENT = re.compile(
    r"\b(order|package|parcel|shipment|delivery|tracking)\b", re.IGNORECASE
)
# An order id: optional '#', then 3+ digits, or an alphanumeric token that
# contains at least one digit (e.g. "A1234"). Anchored to a word boundary.
_ORDER_ID = re.compile(r"#?\b(?=[A-Za-z0-9]*\d)([A-Za-z0-9]{3,})\b")
_ID_STOPWORDS = {"order", "orders", "status", "where", "number"}


def _extract_order_id(text: str) -> str | None:
    for match in _ORDER_ID.finditer(text):
        token = match.group(1)
        if token.lower() in _ID_STOPWORDS:
            continue
        return token
    return None


@dataclass
class MockLLMClient:
    """Rule-based LLM stand-in.

    Behaviour:
    * If the latest turn is a tool result, phrase it as a natural-language reply.
    * Else if the user asks about an order and names an id, emit a
      ``get_order_status`` tool call.
    * Else if they ask about an order without an id, ask for the id.
    * Else return a generic capability message.
    """

    _counter: int = field(default=0, repr=False)

    def respond(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> LLMResponse:
        last = messages[-1]
        if last.get("role") == "tool":
            return LLMResponse(final_text=self._phrase_tool_result(last))

        user_text = self._latest_user_text(messages)
        if user_text is None:
            return LLMResponse(final_text="How can I help you today?")

        if _ORDER_INTENT.search(user_text):
            order_id = _extract_order_id(user_text)
            if order_id is None:
                return LLMResponse(
                    final_text="I can check your order status — what is the order number?"
                )
            self._counter += 1
            return LLMResponse(
                tool_calls=(
                    ToolCall(
                        id=f"call-{self._counter}",
                        name="get_order_status",
                        arguments={"order_id": order_id},
                    ),
                )
            )

        return LLMResponse(
            final_text=(
                "I can help with order status and invoice questions. "
                "Ask me about an order and include its number."
            )
        )

    @staticmethod
    def _latest_user_text(messages: list[dict[str, object]]) -> str | None:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content")
                return content if isinstance(content, str) else None
        return None

    @staticmethod
    def _phrase_tool_result(tool_msg: dict[str, object]) -> str:
        payload = tool_msg.get("content")
        if not isinstance(payload, dict):
            return "Sorry, I could not read the backend response."
        if not payload.get("found"):
            oid = payload.get("order_id", "that order")
            return (
                f"I couldn't find an order matching {oid}. "
                "Please double-check the order number."
            )
        status = payload.get("status")
        oid = payload.get("order_id")
        carrier = payload.get("carrier")
        eta = payload.get("eta")
        parts = [f"Order {oid} is currently '{status}'."]
        if carrier and eta:
            parts.append(f"It is with {carrier}, estimated to arrive {eta}.")
        elif carrier:
            parts.append(f"It is with {carrier}.")
        return " ".join(parts)
