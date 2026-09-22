from collections import OrderedDict


class EmbeddingCache:
    """Bounded LRU; access is serialized by the owning model registry entry."""

    def __init__(self, capacity):
        self.capacity = capacity
        self._values = OrderedDict()

    def get(self, key):
        value = self._values.get(key)
        if value is not None:
            self._values.move_to_end(key)
        return value

    def put(self, key, value):
        if self.capacity:
            self._values[key] = value
            self._values.move_to_end(key)
            while len(self._values) > self.capacity:
                self._values.popitem(last=False)

    def clear(self):
        self._values.clear()
