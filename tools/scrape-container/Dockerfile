FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DISPLAY=:99

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        fonts-liberation \
        gnupg \
        python3 \
        wget \
        xvfb \
    && curl -fsSL https://dl.google.com/linux/linux_signing_key.pub \
        | gpg --dearmor -o /usr/share/keyrings/google-linux.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-linux.gpg] http://dl.google.com/linux/chrome/deb/ stable main" \
        > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ARG BOOK_TRAIN_REV=unknown
ARG HARNESS_REV=unknown
LABEL book-train.rev="${BOOK_TRAIN_REV}" \
      scrape-harness.rev="${HARNESS_REV}"
ENV BOOK_TRAIN_REV="${BOOK_TRAIN_REV}" \
    HARNESS_REV="${HARNESS_REV}"

WORKDIR /app
COPY . /app/book-train
COPY --from=scrape-harness . /app/scrape-harness

RUN cd /app/book-train && uv sync --only-group scrape \
    && cd /app/scrape-harness && uv sync

COPY tools/scrape-container/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /app/book-train
ENTRYPOINT ["/entrypoint.sh"]
