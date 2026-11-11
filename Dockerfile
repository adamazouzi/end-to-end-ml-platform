FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN groupadd --system app && useradd --system --gid app --home-dir /app app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY streamlit_app.py ./
COPY .streamlit ./.streamlit
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-editable

RUN mkdir -p /app/data/raw /app/reports /app/models && chown -R app:app /app
USER app

EXPOSE 8000 8501
CMD ["uvicorn", "churn_platform.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
