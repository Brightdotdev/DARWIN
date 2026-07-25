# ── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

RUN pip install --no-cache-dir --prefix=/install .

# ── Runtime stage ───────────────────────────────────────────────────────────
FROM python:3.11-slim

ENV DARWIN_BASE_DIR=/data
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

RUN groupadd -r darwin && useradd -r -g darwin -d /app -s /sbin/nologin darwin

WORKDIR /app
RUN mkdir -p /data && chown darwin:darwin /data

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER darwin

ENTRYPOINT ["/entrypoint.sh"]
CMD ["darwin", "mcp", "--sse", "--port", "8080"]
