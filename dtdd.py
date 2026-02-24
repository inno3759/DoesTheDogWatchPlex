"""
DoesTheDogDie.com API client with local JSON file caching.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import requests

DTDD_BASE = "https://www.doesthedogdie.com"
CACHE_DIR = Path(__file__).parent / ".cache"

logger = logging.getLogger(__name__)

_MAX_RETRIES = 5


class DTDDClient:
    def __init__(self, api_key: str, cache_ttl: int = 604800, api_delay: float = 1.0):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "X-API-KEY": api_key,
        })
        self.cache_ttl = cache_ttl
        self.api_delay = api_delay
        self._last_request_time = 0.0
        CACHE_DIR.mkdir(exist_ok=True)

    def _rate_limit(self):
        """Enforce minimum delay between API calls."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.api_delay:
            time.sleep(self.api_delay - elapsed)
        self._last_request_time = time.time()

    def _get_cache(self, key: str, ttl: int | None = None) -> dict | None:
        """Read from local JSON cache if still fresh."""
        path = CACHE_DIR / f"{key}.json"
        if not path.exists():
            return None
        effective_ttl = ttl if ttl is not None else self.cache_ttl
        try:
            data = json.loads(path.read_text())
            if time.time() - data.get("_cached_at", 0) < effective_ttl:
                return data.get("_payload")
        except (json.JSONDecodeError, KeyError):
            pass
        return None

    def _set_cache(self, key: str, payload: dict):
        """Write to local JSON cache."""
        path = CACHE_DIR / f"{key}.json"
        path.write_text(json.dumps({
            "_cached_at": time.time(),
            "_payload": payload,
        }, indent=2))

    def _get_json(self, url: str, params: dict | None = None) -> dict:
        """GET a URL with rate limiting and exponential backoff on 429."""
        for attempt in range(_MAX_RETRIES):
            self._rate_limit()
            resp = self.session.get(url, params=params)
            if resp.status_code == 429:
                wait = 2 ** attempt  # 1, 2, 4, 8, 16 s
                logger.warning(f"Rate limited (429). Retrying in {wait}s ({attempt + 1}/{_MAX_RETRIES})…")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"DTDD API rate limit exceeded after {_MAX_RETRIES} retries: {url}")

    def search(self, query: str) -> list[dict]:
        """Search DTDD by title string. Returns list of matching items."""
        cache_key = f"search_{query.lower().replace(' ', '_')}"
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached

        items = self._get_json(f"{DTDD_BASE}/dddsearch", params={"q": query}).get("items", [])
        self._set_cache(cache_key, items)
        return items

    def search_by_imdb(self, imdb_id: str) -> list[dict]:
        """Search DTDD by IMDB ID (e.g., 'tt1234567'). Returns list of matching items."""
        cache_key = f"imdb_{imdb_id}"
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached

        items = self._get_json(f"{DTDD_BASE}/dddsearch", params={"imdb": imdb_id}).get("items", [])
        self._set_cache(cache_key, items)
        return items

    def get_media(
        self,
        item_id: int,
        index1: int | None = None,
        index2: int | None = None,
        cache_ttl_override: int | None = None,
    ) -> dict:
        """Get full trigger/warning data for a DTDD media item.

        For TV shows, pass index1/index2 to filter by season/episode:
          - index1=-1, index2=-1  →  all episodes (use a short cache_ttl_override here
                                       so new episodes are discovered on the next run)
          - index1=1, index2=3    →  S01E03 specifically (cached for full cache_ttl)
        For movies, omit both params.

        cache_ttl_override: if set, overrides self.cache_ttl for this specific call.
        """
        if index1 is not None and index2 is not None:
            cache_key = f"media_{item_id}_{index1}_{index2}"
        else:
            cache_key = f"media_{item_id}"

        cached = self._get_cache(cache_key, ttl=cache_ttl_override)
        if cached is not None:
            return cached

        params = {}
        if index1 is not None:
            params["index1"] = index1
        if index2 is not None:
            params["index2"] = index2

        data = self._get_json(f"{DTDD_BASE}/media/{item_id}", params=params or None)
        self._set_cache(cache_key, data)
        return data

    def clear_cache(self):
        """Remove all cached files."""
        if CACHE_DIR.exists():
            for f in CACHE_DIR.glob("*.json"):
                f.unlink()
            print(f"Cache cleared ({CACHE_DIR})")
