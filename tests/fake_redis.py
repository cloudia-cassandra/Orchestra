"""Minimal in-memory stand-in for redis.Redis, covering only what WorkingMemory calls."""

import fnmatch


class FakeRedis:
    def __init__(self):
        self.strings: dict[str, str] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.lists: dict[str, list[str]] = {}
        self.expiries: dict[str, int] = {}

    def get(self, key):
        return self.strings.get(key)

    def set(self, key, value):
        self.strings[key] = value
        return True

    def hset(self, key, field, value):
        self.hashes.setdefault(key, {})[field] = value
        return 1

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    def rpush(self, key, value):
        self.lists.setdefault(key, []).append(value)
        return len(self.lists[key])

    def lrange(self, key, start, end):
        values = self.lists.get(key, [])
        if end == -1:
            return values[start:]
        return values[start : end + 1]

    def scan_iter(self, match=None):
        all_keys = set(self.strings) | set(self.hashes) | set(self.lists)
        for key in all_keys:
            if match is None or fnmatch.fnmatch(key, match):
                yield key

    def delete(self, *keys):
        removed = 0
        for key in keys:
            for store in (self.strings, self.hashes, self.lists):
                if key in store:
                    del store[key]
                    removed += 1
            self.expiries.pop(key, None)
        return removed

    def expire(self, key, seconds):
        self.expiries[key] = seconds
        return True
