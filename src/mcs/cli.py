"""Interactive terminal chat: ``python -m mcs.cli``.

A dependency-light entrypoint for driving the Phase 1 agent by hand. Reads a
line, prints the agent reply, repeats until EOF/``exit``. A single message can
also be passed as an argument for one-shot use.
"""

from __future__ import annotations

import sys

from .agent import SupportAgent


def _run_once(agent: SupportAgent, message: str) -> None:
    result = agent.run(message)
    tools = ", ".join(result.tool_calls) or "none"
    print(f"agent: {result.reply}")
    print(f"       [stop={result.stop_reason} iterations={result.iterations} tools={tools}]")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    agent = SupportAgent()

    if argv:
        _run_once(agent, " ".join(argv))
        return 0

    print("Multilingual Customer Support AI — Phase 1 CLI. Ctrl-D or 'exit' to quit.")
    while True:
        try:
            line = input("you: ").strip()
        except EOFError:
            print()
            return 0
        if line.lower() in {"exit", "quit"}:
            return 0
        if not line:
            continue
        _run_once(agent, line)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
