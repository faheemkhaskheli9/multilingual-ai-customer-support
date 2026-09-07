# Multilingual Customer Support AI

> LLM, RAG & Agentic AI portfolio project — independent open-source implementation.
> This is an original, from-scratch build. It is not affiliated with, and does not
> contain any code, prompts, data, or business logic from, any employer or client.

![status](https://img.shields.io/badge/status-phase%201%20in%20progress-yellow)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

## 1. Problem

Customer support teams need an assistant that can handle orders, invoices, product questions, and FAQs across multiple languages, through both text and voice.

## 2. Architecture

```text
User Message (text/voice, any language) -> Language Detection/Translation -> Intent + Tool Calling -> RAG/FAQ or Backend Action -> Localized Response
```

## 3. Technology Stack

- Python
- LangChain
- OpenAI API (chat + translation)
- PostgreSQL
- FastAPI
- React

## 4. Feature List

- Order status handling
- Invoice lookups
- Product Q&A
- FAQ retrieval (RAG)
- Tool calling for backend actions
- Text chat interface
- Voice message handling
- Multilingual query understanding
- Automatic translation
- Persistent conversation history

## 5. Implementation Plan

1. Phase 1: Core chat + tool-calling framework with mock order/invoice backend
2. Phase 2: RAG layer for FAQ and product knowledge
3. Phase 3: Multilingual detection and translation layer
4. Phase 4: Voice message support (STT/TTS)

## Task Tracking

Work is broken into phase-tagged user stories tracked as GitHub Issues, not in this file. To see what's open:

    gh issue list --repo faheemkhaskheli9/multilingual-ai-customer-support --state open --label type:user-story

Implement Phase 1 issues first (later phases depend on it). When you start one, add label `status:in-progress`. When you finish, close it referencing the commit (e.g. `git commit -m "... Closes #4"`) and push.

## 6. Repository Structure

```text
multilingual-ai-customer-support/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── .env.example
├── docker/
├── docs/
│   ├── architecture.md
│   └── evaluation.md
├── src/
├── tests/
├── configs/
├── scripts/
├── notebooks/
├── examples/
├── assets/
└── .github/
    └── workflows/
```

## 7. Setup

```bash
git clone <this-repo-url>
cd multilingual-ai-customer-support
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # or: pip install -e .
cp .env.example .env              # fill in API keys / config
```

## 8. Dataset

Document which public dataset(s) or synthetic data generators are used here.
No proprietary, employer-owned, or client-identifiable data is used in this project.

## 9. Training / Execution

Phase 1 ships a bounded tool-calling chat agent over a mock order backend. It
runs with no API key (a deterministic rule-based LLM stand-in); a real provider
drops in behind `mcs.llm.LLMClient` in a later phase.

```bash
pip install -r requirements.txt

# One-shot CLI
PYTHONPATH=src python -m mcs.cli "where is my order #1234?"

# Interactive CLI
PYTHONPATH=src python -m mcs.cli

# HTTP API
PYTHONPATH=src uvicorn mcs.api:app --reload
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message": "where is my order #1234?"}'
```

VS Code: use the **MCS: FastAPI (uvicorn)**, **MCS: CLI chat**, or
**MCS: pytest** run configurations in `.vscode/launch.json`.

## 10. Evaluation

Document evaluation metrics and how to reproduce them here (see `docs/evaluation.md`).

## 11. Results

_To be filled in as the implementation progresses — screenshots, metrics tables, and
sample outputs go here._

## 12. API

_If this project exposes an API, document the main endpoints here (or link to
auto-generated OpenAPI docs, e.g. `/docs` for FastAPI)._

## 13. Docker

```bash
docker build -t multilingual-ai-customer-support .
docker run -p 8000:8000 multilingual-ai-customer-support
```

## 14. Tests

```bash
pytest tests/
```

## 15. Limitations

- This is a from-scratch, independent recreation built for portfolio purposes.
- Performance numbers, once added, are based on public datasets and are not
  representative of any production system's real-world results.

## 16. Future Work

- Expand evaluation coverage and add CI-based regression checks.
- Add more configuration presets and deployment targets.
- Track open items as GitHub Issues.

## 17. Disclosure

This repository is an **independent open-source recreation inspired by the kind of
production systems I have worked on professionally**. It contains no employer or
client source code, prompts, datasets, credentials, architecture diagrams, or
business logic. All code, data, and documentation here are original or built on
publicly available datasets and open-source tools.

---
_Last updated: 2026-08-18_
