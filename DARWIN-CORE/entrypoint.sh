#!/bin/bash
set -e

# Aggressive Docker volume enforcement
if [ -z "$DARWIN_BASE_DIR" ]; then
    echo "🚨 FATAL: DARWIN_BASE_DIR environment variable is not set."
    echo "   Docker containers are EPHEMERAL. You MUST mount a volume."
    echo "   Example: -e DARWIN_BASE_DIR=/data -v my_volume:/data"
    exit 1
fi

# Check if the directory is writable
if [ ! -w "$DARWIN_BASE_DIR" ]; then
    echo "🚨 FATAL: DARWIN_BASE_DIR='$DARWIN_BASE_DIR' is not writable."
    echo "   Did you forget to mount a volume or set proper permissions?"
    exit 1
fi

echo "✅ Persistent volume detected at $DARWIN_BASE_DIR"

# Pass control to the CLI
exec "$@"