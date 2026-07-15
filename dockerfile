FROM ghcr.io/astral-sh/uv:debian

WORKDIR /opt/app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-cache --no-install-project --group ml

COPY . .

RUN uv run python scr/pipeline.py

CMD ["uv", "run", "python", "main.py"]
