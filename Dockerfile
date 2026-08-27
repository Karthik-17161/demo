# KLCAP-2026-00167 Vision-Language Representation Lab
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /lab

COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY src/ ./src/
COPY apps/api/ ./apps/api/
COPY configs/ ./configs/

# Non-root operator
RUN useradd --create-home labuser && chown -R labuser /lab
USER labuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
