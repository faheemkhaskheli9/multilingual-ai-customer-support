"""Bounded tool-calling support agent.

Loop shape follows the knowledge-base pattern "Stateful agentic run lifecycle":
a bounded request -> tool-call pause -> resume cycle that branches on an
explicit stop reason instead of only handling the success path.

Phase 1 wires ``get_order_status`` and ``get_invoice``. Issue #4 generalises
the registry and swaps :class:`~mcs.llm.MockLLMClient` for a real provider;
until then, multi-turn context is threaded explicitly via `run(history=...)`
rather than persisted server-side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .backends.invoices import InvoiceNotFoundError, get_invoice
from .backends.orders import OrderNotFoundError, get_order_status
from .llm import LLMClient, MockLLMClient, ToolCall

MAX_ITERATIONS = 4
"""Hard backstop. Phase 1 needs 2 turns (call tool, phrase result); the cap is
a runaway-cost guard, not a tuning knob."""

ToolFn = Callable[[dict[str, object]], dict[str, object]]


def _tool_get_order_status(arguments: dict[str, object]) -> dict[str, object]:
    order_id = arguments.get("order_id")
    if not isinstance(order_id, str) or not order_id.strip():
        return {"found": False, "order_id": order_id, "error": "missing order_id"}
    try:
        return get_order_status(order_id)
    except OrderNotFoundError:
        # Structured "not found" rather than an exception bubbling up, so the
        # LLM phrases a safe reply instead of the caller guessing.
        return {"found": False, "order_id": order_id}


def _tool_get_invoice(arguments: dict[str, object]) -> dict[str, object]:
    order_id = arguments.get("order_id")
    if not isinstance(order_id, str) or not order_id.strip():
        return {"found": False, "order_id": order_id, "error": "missing order_id"}
    try:
        return get_invoice(order_id)
    except InvoiceNotFoundError:
        return {"found": False, "order_id": order_id}


DEFAULT_TOOLS: dict[str, ToolFn] = {
    "get_order_status": _tool_get_order_status,
    "get_invoice": _tool_get_invoice,
}

_TOOL_SPECS = [
    {
        "name": "get_order_status",
        "description": "Look up the delivery status of an order by its order number.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "get_invoice",
        "description": "Look up invoice amount, date, and line items for an order/account number.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
]


@dataclass(frozen=True)
class AgentResult:
    reply: str
    stop_reason: str  # "completed" | "max_iterations"
    iterations: int
    tool_calls: tuple[str, ...]
    # Full message thread including this turn, so a caller can pass it back
    # as `history` on the next `run()` call and get multi-turn follow-ups
    # ("what about the tax on that?") resolved from context instead of a
    # fresh lookup. See knowledge-base "working-memory": the application
    # owns this state explicitly rather than the LLM client accumulating it.
    history: tuple[dict[str, object], ...]


class SupportAgent:
    def __init__(
        self,
        llm: LLMClient | None = None,
        tools: dict[str, ToolFn] | None = None,
        max_iterations: int = MAX_ITERATIONS,
    ) -> None:
        self.llm: LLMClient = llm or MockLLMClient()
        self.tools = dict(DEFAULT_TOOLS if tools is None else tools)
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self.max_iterations = max_iterations

    def run(
        self,
        user_message: str,
        history: list[dict[str, object]] | None = None,
    ) -> AgentResult:
        if not user_message or not user_message.strip():
            raise ValueError("user_message must be non-empty")

        messages: list[dict[str, object]] = list(history or [])
        messages.append({"role": "user", "content": user_message})
        invoked: list[str] = []

        for iteration in range(1, self.max_iterations + 1):
            response = self.llm.respond(messages, _TOOL_SPECS)

            if response.final_text is not None:
                messages.append({"role": "assistant", "content": response.final_text})
                return AgentResult(
                    reply=response.final_text,
                    stop_reason="completed",
                    iterations=iteration,
                    tool_calls=tuple(invoked),
                    history=tuple(messages),
                )

            for call in response.tool_calls:
                invoked.append(call.name)
                result = self._execute_tool(call)
                messages.append(
                    {"role": "assistant", "tool_calls": [call.name], "content": None}
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": result,
                    }
                )

        # Deliberate termination: the model never settled on a final answer.
        reply = (
            "I'm having trouble completing that request right now. "
            "Please try rephrasing or contact a human agent."
        )
        messages.append({"role": "assistant", "content": reply})
        return AgentResult(
            reply=reply,
            stop_reason="max_iterations",
            iterations=self.max_iterations,
            tool_calls=tuple(invoked),
            history=tuple(messages),
        )

    def _execute_tool(self, call: ToolCall) -> dict[str, object]:
        fn = self.tools.get(call.name)
        if fn is None:
            return {"error": f"unknown tool {call.name!r}"}
        return fn(call.arguments)
