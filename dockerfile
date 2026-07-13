FROM ubuntu:latest

WORKDIR /opt/app

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    curl \
    python3 \
    python3-venv \

RUN curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh

ENV UV_COMPILE_BYTECODE=1

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project

COPY . .

CMD uv run python main.py
