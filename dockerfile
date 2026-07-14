FROM ghcr.io/astral-sh/uv:debian

WORKDIR /opt/app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-cache --no-install-project

COPY . .

RUN uv run python -c "import sys; sys.exit()"

CMD ["uv", "run", "python", "main.py"]
