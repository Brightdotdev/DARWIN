# src/utils/internet_checker.py

import socket
import urllib.request
import time
from typing import Optional
from src.core.Logger.DarwinLogger import darwin,LogType

# Configuration
TEST_HOSTS = [
    ("1.1.1.1", 53, 2.0, "socket"),       # Cloudflare DNS
    ("8.8.8.8", 53, 2.0, "socket"),       # Google DNS
    ("https://www.gstatic.com/generate_204", 0, 3.0, "http"),
    ("https://www.google.com", 0, 3.0, "http"),
    ("https://www.cloudflare.com", 0, 3.0, "http"),
]

def check_internet_robust(verbose: bool = False) -> bool:
    """
    Returns True if at least one connectivity check succeeds.
    If logger is provided, it's called with (message, level) or just message.
    """
    for host, port, timeout, method in TEST_HOSTS:
        try:
            if method == "socket":
                if _check_socket(host, port, timeout, verbose):
                    return True
            elif method == "http":
                if _check_http(host, timeout, verbose):
                    return True
        except Exception as e:
            if verbose:
                darwin.log(f"Check failed for {host}: {e}")
            continue
    return False

def _check_socket(host: str, port: int, timeout: float, verbose: bool) -> bool:
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        if verbose:
            darwin.log(f"Socket OK: {host}:{port}")
        return True
    except OSError as e:
        if verbose:
            darwin.log(f"Socket failed: {host}:{port} - {e}",LogType.ERROR)
        return False

def _check_http(url: str, timeout: float, verbose: bool) -> bool:
    try:
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if 200 <= resp.status < 400:
                if verbose:
                    darwin.log(f"HTTP OK: {url} (Status: {resp.status})")
                return True
            else:
                if verbose:
                    darwin.log(f"HTTP bad status: {url} (Status: {resp.status})",LogType.WARNING)
                return False
    except Exception as e:
        if verbose:
            darwin.log(f"HTTP failed: {url} - {e}")
        return False

def wait_for_internet(
    pause: int = 5,
    max_wait: Optional[int] = None,
    verbose: bool = True,
) -> bool:
    """
    Blocks until internet is available or max_wait seconds elapsed.
    Logs progress messages.
    
    Returns:
        True if connected, False if timed out.
    """
    start = time.time()
    attempt = 0

    while True:
        attempt += 1
        elapsed = int(time.time() - start)
        if max_wait:
            remaining = max_wait - elapsed
            darwin.log(f"Internet check attempt {attempt} (elapsed {elapsed}s, remaining {remaining}s)...")
        else:
            darwin.log(f"Internet check attempt {attempt}...")

        if check_internet_robust(verbose=False):
            darwin.log("✓ Internet connection established.")
            return True

        if max_wait and elapsed >= max_wait:
            darwin.log(f"✗ No internet after waiting {max_wait} seconds. Check your network.",LogType.ERROR)
            return False

        darwin.log(f"Waiting {pause} seconds before retry...")
        time.sleep(pause)

