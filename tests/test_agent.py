import pytest

from mcs.agent import MAX_ITERATIONS, SupportAgent
from mcs.llm import LLMResponse, ToolCall


def test_order_status_intent_calls_tool_and_reports_status():
    result = SupportAgent().run("where is my order #1234?")
    assert result.stop_reason == "completed"
    assert result.tool_calls == ("get_order_status",)
    assert "1234" in result.reply
    assert "shipped" in result.reply


def test_unknown_order_gets_graceful_not_found_not_a_hallucination():
    result = SupportAgent().run("status of order 0000 please")
    assert result.tool_calls == ("get_order_status",)
    assert "couldn't find" in result.reply.lower()
    # must not invent a status
    for invented in ("shipped", "processing", "delivered", "cancelled"):
        assert invented not in result.reply.lower()


def test_order_question_without_id_asks_for_it():
    result = SupportAgent().run("where is my order?")
    assert result.tool_calls == ()
    assert "order number" in result.reply.lower()


def test_non_order_message_gets_capability_reply():
    result = SupportAgent().run("hello there")
    assert result.stop_reason == "completed"
    assert result.tool_calls == ()


def test_empty_message_rejected():
    with pytest.raises(ValueError):
        SupportAgent().run("   ")


class _LoopingLLM:
    """Always calls a tool -> forces the max-iteration backstop."""

    def respond(self, messages, tools):
        return LLMResponse(
            tool_calls=(ToolCall(id="x", name="get_order_status", arguments={"order_id": "1234"}),)
        )


def test_max_iteration_backstop_terminates_deliberately():
    agent = SupportAgent(llm=_LoopingLLM())
    result = agent.run("loop forever")
    assert result.stop_reason == "max_iterations"
    assert result.iterations == MAX_ITERATIONS
    assert len(result.tool_calls) == MAX_ITERATIONS


def test_max_iterations_must_be_positive():
    with pytest.raises(ValueError):
        SupportAgent(max_iterations=0)


def test_invoice_intent_calls_tool_and_reports_line_items():
    result = SupportAgent().run("can I get the invoice for order 1234?")
    assert result.stop_reason == "completed"
    assert result.tool_calls == ("get_invoice",)
    assert "Wireless mouse" in result.reply
    assert "1234" in result.reply


def test_unknown_invoice_gets_graceful_not_found():
    result = SupportAgent().run("invoice for order 0000 please")
    assert result.tool_calls == ("get_invoice",)
    assert "couldn't find an invoice" in result.reply.lower()


def test_invoice_question_without_id_asks_for_it():
    result = SupportAgent().run("I need my invoice")
    assert result.tool_calls == ()
    assert "order or account number" in result.reply.lower()


def test_tax_followup_resolves_from_history_not_a_fresh_lookup():
    agent = SupportAgent()
    first = agent.run("what's the invoice for order 1234?")
    assert first.tool_calls == ("get_invoice",)

    followup = agent.run("what about the tax on that?", history=list(first.history))
    # No second backend call — answered purely from the thread's own context.
    assert followup.tool_calls == ()
    assert "2.24" in followup.reply
    assert "1234" in followup.reply


def test_tax_followup_without_prior_invoice_asks_for_id():
    result = SupportAgent().run("what about the tax on that?")
    assert result.tool_calls == ()
    assert "order or account number" in result.reply.lower()


def test_run_returns_history_for_threading_next_turn():
    result = SupportAgent().run("hello there")
    assert result.history[0] == {"role": "user", "content": "hello there"}
    assert result.history[-1] == {"role": "assistant", "content": result.reply}
