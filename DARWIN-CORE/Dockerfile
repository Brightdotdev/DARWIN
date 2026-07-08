FROM python:3.11-slim

# Set default (will be overridden by user)
ENV DARWIN_BASE_DIR=/app/data
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps (for sentence-transformers, faiss, etc.)
RUN apt-get update && apt-get install -y \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install your package
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# Copy entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Ensure the data directory exists inside container (but user must mount it)
RUN mkdir -p /app/data

ENTRYPOINT ["/entrypoint.sh"]
CMD ["darwin", "--help"]