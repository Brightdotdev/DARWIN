#!/bin/bash
set -e

DARWIN_BASE_DIR="${DARWIN_BASE_DIR:-/data}"

# Enforce persistent volume
if [ ! -d "$DARWIN_BASE_DIR" ]; then
    echo "FATAL: DARWIN_BASE_DIR=$DARWIN_BASE_DIR does not exist."
    echo "  Docker containers are EPHEMERAL. You MUST mount a volume."
    echo "  Example: -e DARWIN_BASE_DIR=/data -v my_volume:/data"
    exit 1
fi

if [ ! -w "$DARWIN_BASE_DIR" ]; then
    echo "FATAL: DARWIN_BASE_DIR=$DARWIN_BASE_DIR is not writable."
    echo "  Did you forget to mount a volume or set proper permissions?"
    exit 1
fi

echo "Persistent volume detected at $DARWIN_BASE_DIR"

# Auto-setup on first run
CONFIG_FILE="$DARWIN_BASE_DIR/config/darwin.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "First run detected -- running setup with 'server' preset..."
    darwin setup preset server --yes --base-dir "$DARWIN_BASE_DIR"
    echo "Setup complete."
fi

# Pass control to the CLI
exec "$@"
