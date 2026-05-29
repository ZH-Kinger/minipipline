"""Persistent SQLite cache for DashScope Qwen-VL responses.

Keyed by (model, prompt, generation params, sorted frame content hashes) so
identical inputs short-circuit network calls. Survives `--force` reruns,
process restarts, and dev iteration.

Cache lives at ``<output_root>/.annotation_cache.db`` by default — alongside
the data it caches; ``rm output/`` wipes the cache with it.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    key BLOB PRIMARY KEY,
    model TEXT NOT NULL,
    response_text TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


def make_cache_key(
    model: str,
    prompt: str,
    frame_bytes_list: list[bytes],
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> bytes:
    """Deterministic 16-byte key. Changes in any input → new key → miss."""
    h = hashlib.blake2b(digest_size=16)
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(prompt.encode("utf-8"))
    h.update(b"\x00")
    h.update(f"{temperature:.4f}|{top_p:.4f}|{max_tokens}".encode("ascii"))
    h.update(b"\x00")
    # Sort by content hash so frame order doesn't change the key (VLM
    # treats them as a set conceptually here — they're all one clip).
    frame_hashes = sorted(hashlib.sha256(b).digest() for b in frame_bytes_list)
    for fh in frame_hashes:
        h.update(fh)
    return h.digest()


class AnnotationCache:
    """Thread-safe SQLite-backed cache for VLM response text.

    Multiple threads (clip-parallel ThreadPool) share one instance.
    Multiple processes (ProcessPool batch) each open their own connection;
    WAL mode + a per-instance write lock keep concurrent access sane.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,  # autocommit
            timeout=10.0,
        )
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
        self.hits = 0
        self.misses = 0

    def get(
        self,
        *,
        model: str,
        prompt: str,
        frame_bytes_list: list[bytes],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> str | None:
        key = make_cache_key(model, prompt, frame_bytes_list, temperature, top_p, max_tokens)
        with self._lock:
            row = self._conn.execute(
                "SELECT response_text FROM cache WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return row[0]

    def put(
        self,
        *,
        model: str,
        prompt: str,
        frame_bytes_list: list[bytes],
        temperature: float,
        top_p: float,
        max_tokens: int,
        response_text: str,
    ) -> None:
        key = make_cache_key(model, prompt, frame_bytes_list, temperature, top_p, max_tokens)
        # Retry a few times on transient "database is locked" under heavy
        # ProcessPool concurrency. WAL mostly avoids this, but be safe.
        for attempt in range(3):
            try:
                with self._lock:
                    self._conn.execute(
                        "INSERT OR REPLACE INTO cache (key, model, response_text, created_at) "
                        "VALUES (?, ?, ?, ?)",
                        (key, model, response_text, time.time()),
                    )
                return
            except sqlite3.OperationalError as exc:
                if "locked" in str(exc).lower() and attempt < 2:
                    time.sleep(0.05 * (attempt + 1))
                    continue
                raise

    def stats(self) -> dict[str, float]:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()
            n_rows = int(row[0]) if row else 0
        size_mb = self.db_path.stat().st_size / 1024 / 1024 if self.db_path.exists() else 0.0
        total = self.hits + self.misses
        return {
            "rows": n_rows,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
            "size_mb": round(size_mb, 3),
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()


__all__ = ["AnnotationCache", "make_cache_key"]