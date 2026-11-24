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
MAX_RATE_LIMIT_RETRIES = 12   # the EuroLeague API blocks for ~5 min after a burst; be patient
MIN_RATE_LIMIT_WAIT = 30.0    # a "Retry-After: 0" must not turn into instant retries


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

    def is_cached(self, url: str, params: dict[str, Any] | None = None) -> bool:
        path = self._cache_path(url, params)
        return path is not None and path.exists()

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
        attempt = 0
        rate_limited = 0
        while True:
            attempt += 1
            self._throttle()
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout, headers=headers)
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt > self.max_retries:
                    raise
                log.warning("network error %s on %s, retrying in %.0fs", exc, url, delay)
                time.sleep(delay)
                delay *= 2
                continue

            if resp.status_code == 429 and rate_limited < MAX_RATE_LIMIT_RETRIES:
                rate_limited += 1
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else 0.0
                sleep_for = max(wait, MIN_RATE_LIMIT_WAIT)
                log.warning("HTTP 429 (rate limit) on %s, waiting %.0fs (attempt %d/%d)",
                            url, sleep_for, rate_limited, MAX_RATE_LIMIT_RETRIES)
                time.sleep(sleep_for)
                continue
            if resp.status_code in RETRY_STATUSES and resp.status_code != 429 and attempt <= self.max_retries:
                log.warning("HTTP %s on %s, retrying in %.0fs", resp.status_code, url, delay)
                time.sleep(delay)
                delay *= 2
                continue

            resp.raise_for_status()
            data = resp.json()
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data), encoding="utf-8")
            return data
