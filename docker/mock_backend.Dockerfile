FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/

ENV PYTHONPATH=/app/src

EXPOSE 8100

CMD ["uvicorn", "mcs.mock_backend.app:app", "--host", "0.0.0.0", "--port", "8100"]
