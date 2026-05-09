"""
Paper API wrappers with SQLite caching.
Every response is cached — re-running Phase A on the same paper
costs zero API calls after the first run.
"""

import time
import requests
from core.config import (
    S2_BASE,
    S2_FIELDS,
    S2_REF_FIELDS,
    MAX_REFS_PER_PAPER,
    UNPAYWALL_EMAIL,
)
from utils.cache import get_cached, set_cached

_HEADERS = {"User-Agent": "KGMS-Research/1.0"}


def _s2_get(url: str, params: dict = None, retries: int = 3) -> dict | None:
    """Rate-limit-aware GET for Semantic Scholar (100 req/5min unauthenticated)."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=_HEADERS, timeout=15)
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == retries - 1:
                print(f"[API] S2 error for {url}: {e}")
                return None
            time.sleep(2)
    return None


# ── Paper fetch ────────────────────────────────────────────────────────────


def fetch_paper(paper_id: str) -> dict | None:
    """
    Fetch a single paper by any ID: S2 paper ID, DOI, or arXiv ID.
    Accepted formats:
      - S2 ID:   "649def34f8be52c8b66281af98ae884c09aef38b"
      - DOI:     "DOI:10.1109/..."
      - arXiv:   "arXiv:2404.16130"
    """
    cache_key = f"paper:{paper_id}"
    cached = get_cached(cache_key)
    if cached:
        return cached

    data = _s2_get(f"{S2_BASE}/paper/{paper_id}", params={"fields": S2_FIELDS})
    if data and data.get("paperId"):
        set_cached(cache_key, data)
    return data


def fetch_references(paper_id: str) -> list[dict]:
    """
    Fetch direct references (Level 1) of a paper.
    Returns a flat list of paper dicts (may include nulls — filter downstream).
    Capped at MAX_REFS_PER_PAPER.
    """
    cache_key = f"refs:{paper_id}"
    cached = get_cached(cache_key)
    if cached is not None:
        return cached

    data = _s2_get(
        f"{S2_BASE}/paper/{paper_id}/references",
        params={"fields": S2_REF_FIELDS, "limit": MAX_REFS_PER_PAPER},
    )
    if not data:
        return []

    refs = []
    for entry in data.get("data", []):
        cp = entry.get("citedPaper")
        if cp and cp.get("paperId") and cp.get("abstract"):
            refs.append(cp)

    set_cached(cache_key, refs)
    return refs


def search_papers(query: str, limit: int = 10) -> list[dict]:
    """Search Semantic Scholar by keyword query (used for user-specified gaps)."""
    cache_key = f"search:{query}:{limit}"
    cached = get_cached(cache_key)
    if cached is not None:
        return cached

    data = _s2_get(
        f"{S2_BASE}/paper/search",
        params={"query": query, "fields": S2_FIELDS, "limit": limit},
    )
    results = []
    if data:
        for p in data.get("data", []):
            if p.get("paperId") and p.get("abstract"):
                results.append(p)

    set_cached(cache_key, results)
    return results


# ── PDF resolution ─────────────────────────────────────────────────────────


def resolve_pdf_url(paper_data: dict) -> str | None:
    """
    Try three sources in priority order:
    1. Semantic Scholar openAccessPdf
    2. Unpaywall by DOI
    3. arXiv API

    Returns a direct PDF URL or None.
    """
    pid = paper_data.get("paperId", "")
    cache_key = f"pdf_url:{pid}"
    cached = get_cached(cache_key)
    if cached is not None:
        return cached or None

    url = None

    # 1. S2 open access
    oap = paper_data.get("openAccessPdf")
    if oap and oap.get("url"):
        url = oap["url"]

    # 2. Unpaywall
    if not url:
        doi = paper_data.get("externalIds", {}).get("DOI")
        if doi:
            try:
                r = requests.get(
                    f"https://api.unpaywall.org/v2/{doi}",
                    params={"email": UNPAYWALL_EMAIL},
                    timeout=8,
                )
                if r.ok:
                    best = r.json().get("best_oa_location") or {}
                    url = best.get("url_for_pdf") or best.get("url")
            except Exception:
                pass

    # 3. arXiv
    if not url:
        arxiv_id = paper_data.get("externalIds", {}).get("ArXiv")
        if arxiv_id:
            url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    set_cached(cache_key, url or "")
    return url


def download_pdf(url: str, dest_path: str) -> bool:
    """Download a PDF to dest_path. Returns True on success."""
    try:
        r = requests.get(
            url,
            timeout=20,
            headers={**_HEADERS, "Accept": "application/pdf"},
            stream=True,
        )
        content_type = r.headers.get("Content-Type", "")
        if r.ok and ("pdf" in content_type or url.endswith(".pdf")):
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            return True
    except Exception as e:
        print(f"[PDF] Download failed for {url}: {e}")
    return False
