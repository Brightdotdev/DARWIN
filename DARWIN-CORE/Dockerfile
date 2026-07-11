FROM python:3.11-slim

# Runtime configuration
ENV DARWIN_BASE_DIR=/data
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (gcc/g++ needed for some pip wheels)
RUN apt-get update && apt-get install -y \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Copy package definition first for dependency caching
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install the package (non-editable for production)
RUN pip install --no-cache-dir .

# Copy entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Ensure data directory exists
RUN mkdir -p /data

ENTRYPOINT ["/entrypoint.sh"]
CMD ["darwin", "mcp", "--sse", "--port", "8080"]
