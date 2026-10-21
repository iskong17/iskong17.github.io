"""Small clients for the Jolpica F1 API (Ergast-compatible) and OpenF1.

Both are free and need no key. Requests are cached in memory and retried with
backoff on rate limits, so a long backfill can run unattended.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

JOLPICA = "https://api.jolpi.ca/ergast/f1"
OPENF1 = "https://api.openf1.org/v1"
UA = {"User-Agent": "iskong17.github.io F1 pages"}

_cache = {}


def _fetch(url, pause, retries=12):
    if url in _cache:
        return _cache[url]
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                data = json.load(r)
            time.sleep(pause)
            _cache[url] = data
            return data
        except (urllib.error.URLError, TimeoutError) as e:
            code = getattr(e, "code", None)
            if code and 400 <= code < 500 and code != 429:
                raise  # a client error will not fix itself
            if attempt == retries - 1:
                raise
            wait = min(60 * (attempt + 1), 300) if code == 429 else 3 * (attempt + 1)
            print(f"retry in {wait}s ({e}): {url}", file=sys.stderr)
            time.sleep(wait)


def jolpica(path, limit=100, offset=0):
    """MRData for an Ergast-style path, e.g. jolpica('2026/17/results')."""
    url = f"{JOLPICA}/{path}/?format=json&limit={limit}&offset={offset}"
    # JOLPICA_PAUSE slows long backfills down to the API's sustained limit (about 500 requests/hour).
    return _fetch(url, pause=float(os.environ.get("JOLPICA_PAUSE", "0.6")))["MRData"]


def jolpica_all(path, table, key):
    """Follow pagination and return every item of MRData[table][key]."""
    items, offset = [], 0
    while True:
        data = jolpica(path, limit=100, offset=offset)
        page = data[table][key]
        items += page
        offset += 100
        if offset >= int(data["total"]) or not page:
            return items


def openf1(endpoint, **params):
    """OpenF1 list endpoint. Use keys like date__gt for 'date>' filters."""
    parts = []
    for k, v in params.items():
        op = "="
        for suffix, sym in (("__gt", ">"), ("__lt", "<")):
            if k.endswith(suffix):
                k, op = k[: -len(suffix)], sym
        parts.append(f"{k}{op}{urllib.parse.quote(str(v), safe='')}")
    url = f"{OPENF1}/{endpoint}?" + "&".join(parts)
    try:
        return _fetch(url, pause=1.0)
    except urllib.error.HTTPError as e:
        if e.code == 404:  # OpenF1 answers 404 when a query matches nothing
            return []
        raise


def merge_races(races):
    """Jolpica splits one race across pages when results are paginated; merge them."""
    merged = {}
    for r in races:
        key = (r["season"], r["round"])
        if key in merged:
            merged[key]["Results"] += r.get("Results", [])
        else:
            merged[key] = dict(r, Results=list(r.get("Results", [])))
    return list(merged.values())
