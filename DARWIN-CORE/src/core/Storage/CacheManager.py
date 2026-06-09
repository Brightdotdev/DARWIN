from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Optional


class CacheManager:
    """Thread-safe bounded LRU cache keyed by hash, not filename."""

    def __init__(self, maxsize: int = 256):
        self._maxsize = maxsize
        self._cache: dict[str, Any] = {}
        self._order: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                self._order.move_to_end(key)
                return self._cache[key]
            return None

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                self._cache[key] = value
                self._order.move_to_end(key)
                return
            if len(self._cache) >= self._maxsize:
                lru_key, _ = self._order.popitem(last=False)
                del self._cache[lru_key]
            self._cache[key] = value
            self._order[key] = None
            self._order.move_to_end(key)

    def invalidate(self, prefix: Optional[str] = None) -> None:
        with self._lock:
            if prefix is None:
                self._cache.clear()
                self._order.clear()
            else:
                to_del = [k for k in self._cache if k.startswith(prefix)]
                for k in to_del:
                    del self._cache[k]
                    del self._order[k]