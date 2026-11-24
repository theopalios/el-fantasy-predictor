"""Polite HTTP client: rate limiting, retries with backoff, optional on-disk JSON cache."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

import requests

log = logging.getLogger(__name__)

RETRY_STATUSES = {429, 500, 502, 503, 504}


class HttpClient:
    def __init__(
        self,
        cache_dir: Path | None = None,
        min_interval: float = 0.35,
        timeout: float = 30.0,
        max_retries: int = 5,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.min_interval = min_interval
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json", "User-Agent": "elfantasy/0.1"})
        if headers:
            self.session.headers.update(headers)
        self._last_request = 0.0
        self._lock = threading.Lock()  # spaces request starts even when called from threads

    # ------------------------------------------------------------------ cache
    def _cache_path(self, url: str, params: dict[str, Any] | None) -> Path | None:
        if self.cache_dir is None:
            return None
        full = url + ("?" + urlencode(sorted(params.items()), doseq=True) if params else "")
        digest = hashlib.sha1(full.encode()).hexdigest()
        host = urlsplit(url).netloc.replace(":", "_")
        return self.cache_dir / host / digest[:2] / f"{digest}.json"

    # ---------------------------------------------------------------- request
    def _throttle(self) -> None:
        with self._lock:
            wait = self.min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()

    def get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        cache: bool = False,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """GET a JSON resource. With cache=True, a cached copy is returned if present and
        a successful response is stored. Only cache resources that no longer change."""
        path = self._cache_path(url, params) if cache else None
        if path is not None and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))

        delay = 2.0
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout, headers=headers)
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt == self.max_retries:
                    raise
                log.warning("network error %s on %s, retrying in %.0fs", exc, url, delay)
                time.sleep(delay)
                delay *= 2
                continue

            if resp.status_code in RETRY_STATUSES and attempt < self.max_retries:
                retry_after = resp.headers.get("Retry-After")
                sleep_for = float(retry_after) if retry_after and retry_after.isdigit() else delay
                log.warning("HTTP %s on %s, retrying in %.0fs", resp.status_code, url, sleep_for)
                time.sleep(sleep_for)
                delay *= 2
                continue

            resp.raise_for_status()
            data = resp.json()
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data), encoding="utf-8")
            return data

        raise RuntimeError(f"unreachable: retries exhausted for {url}")
