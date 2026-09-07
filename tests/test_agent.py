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
