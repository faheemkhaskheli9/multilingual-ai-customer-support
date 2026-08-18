# Architecture Notes: Multilingual Customer Support AI

## Pipeline

```text
User Message (text/voice, any language) -> Language Detection/Translation -> Intent + Tool Calling -> RAG/FAQ or Backend Action -> Localized Response
```

## Components

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

## Design Notes

- Keep provider/model choices swappable behind interfaces (see `multi-llm-router`
  and similar projects in this portfolio for the general pattern).
- Prefer configuration-driven pipelines (YAML/JSON in `configs/`) over hardcoded
  parameters so experiments are reproducible.
