"""Port availability checking and fallback logic."""

import socket


def is_port_available(host: str, port: int) -> bool:
    """Return True if the given host:port can be bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def find_available_port(host: str, start_port: int, max_tries: int = 10) -> int | None:
    """Return the first available port in [start_port, start_port+max_tries), or None."""
    for port in range(start_port, start_port + max_tries):
        if is_port_available(host, port):
            return port
    return None
