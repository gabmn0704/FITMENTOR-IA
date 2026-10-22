from collections import OrderedDict
from threading import Lock
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    """Caché LRU basada en OrderedDict. Protegida con un lock para uso concurrente."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._capacity = capacity
        self._items: OrderedDict[K, V] = OrderedDict()
        self._lock = Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def get(self, key: K) -> V | None:
        with self._lock:
            if key not in self._items:
                return None
            self._items.move_to_end(key)
            return self._items[key]

    def set(self, key: K, value: V) -> None:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            if len(self._items) > self._capacity:
                self._items.popitem(last=False)

    def delete(self, key: K) -> bool:
        with self._lock:
            return self._items.pop(key, None) is not None
