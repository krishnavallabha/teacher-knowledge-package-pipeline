# Docker deploy is required (rather than Render's default Python buildpack)
# specifically because the OCR path (app/services/document_intelligence.py)
# needs the tesseract-ocr system binary, which pytesseract only wraps --
# a plain `pip install -r requirements.txt` on a buildpack has no way to
# install it. On Render: choose "Docker" as the environment when creating
# the web service and point it at this file; no other build config needed.

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render injects $PORT; default to 8000 for local `docker run`.
ENV PORT=8000
EXPOSE 8000

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
