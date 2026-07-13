FROM ghcr.io/astral-sh/uv:debian

WORKDIR /opt/app

RUN apt-get update && apt-get upgrade -y

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project

COPY . .

CMD uv run python main.py
