# Serve image: FastAPI + MCP server + agent graph. Benchmarks and evals are not included.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/app/.venv PYTHONUNBUFFERED=1

# Hugging Face Spaces runs containers as uid 1000.
RUN useradd -m -u 1000 app
WORKDIR /app

# Dependencies first, for layer caching (tau2 is a path dependency, so it comes along).
COPY pyproject.toml uv.lock README.md ./
COPY vendor/tau2-bench vendor/tau2-bench
RUN uv sync --frozen --no-dev --no-install-project

COPY src src
COPY scripts/migrate.py scripts/migrate.py
RUN uv sync --frozen --no-dev && chown -R app:app /app

USER app
ENV PATH="/app/.venv/bin:$PATH" PORT=7860
EXPOSE 7860
# One process: migrate (idempotent), then serve. MCP is called over this same server.
CMD ["sh", "-c", "python scripts/migrate.py && MCP_URL=${MCP_URL:-http://127.0.0.1:$PORT/mcp/} exec uvicorn api.main:app --app-dir src --host 0.0.0.0 --port $PORT --proxy-headers --forwarded-allow-ips='*'"]
