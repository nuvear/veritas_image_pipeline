"""
image_search.py — Stage 3: Image search via SerpAPI Google Images + download.

Uses SerpAPI (Google Images) for reliable, high-quality results.
Downloads candidates with resolution filtering (min 400px on each side).
Blocks stock photo domains.
"""

import os
import time
import hashlib
import logging
import requests
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SERP_API_KEY = os.environ.get("SERP_API_KEY", "")
SERP_ENDPOINT = "https://serpapi.com/search"

BLOCKED_DOMAINS = {
    "shutterstock.com", "gettyimages.com", "istockphoto.com",
    "alamy.com", "dreamstime.com", "depositphotos.com",
    "123rf.com", "bigstockphoto.com", "stockfood.com",
    "fotolia.com", "adobe.com/stock", "vectorstock.com",
    "freepik.com", "flaticon.com", "vecteezy.com",
    "canstockphoto.com", "pond5.com", "dissolve.com",
    "offset.com", "superstock.com", "agefotostock.com",
}

MIN_DIMENSION = 400      # Minimum pixel dimension (width or height)
DOWNLOAD_TIMEOUT = 12    # Seconds per image download
MAX_CANDIDATES = 12      # Max images to download per query
REQUEST_DELAY = 1.0      # Seconds between SerpAPI calls


# ---------------------------------------------------------------------------
# Domain filtering
# ---------------------------------------------------------------------------

def _is_blocked(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
        return any(blocked in host for blocked in BLOCKED_DOMAINS)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# SerpAPI search
# ---------------------------------------------------------------------------

def search_images(query: str, num: int = 20) -> list:
    """
    Search Google Images via SerpAPI.
    Returns list of image URLs (up to `num`).
    """
    if not SERP_API_KEY:
        logger.error("[SERP] SERP_API_KEY not set")
        return []

    params = {
        "engine": "google_images",
        "q": query,
        "num": num,
        "api_key": SERP_API_KEY,
        "safe": "active",
        "ijn": "0",
    }

    try:
        resp = requests.get(SERP_ENDPOINT, params=params, timeout=20)
        logger.info(f"[SERP] {query[:60]} → {resp.status_code}")
        if resp.status_code != 200:
            logger.warning(f"[SERP] Non-200: {resp.status_code} for query: {query}")
            return []

        data = resp.json()
        results = data.get("images_results", [])
        urls = []
        for r in results:
            url = r.get("original") or r.get("thumbnail")
            if url and not _is_blocked(url):
                urls.append(url)
        logger.info(f"[SERP] Got {len(urls)} URLs for '{query[:50]}'")
        time.sleep(REQUEST_DELAY)
        return urls[:num]

    except Exception as e:
        logger.warning(f"[SERP] Search failed for '{query}': {e}")
        return []


# ---------------------------------------------------------------------------
# Image download
# ---------------------------------------------------------------------------

def _url_to_filename(url: str, idx: int) -> str:
    h = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"cand_{idx:03d}_{h}.jpg"


def download_images(urls: list, dest_dir: Path, max_count: int = MAX_CANDIDATES) -> list:
    """
    Download images from URLs to dest_dir.
    Returns list of local file paths that passed resolution filter.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
        "Referer": "https://www.google.com/",
    }

    for idx, url in enumerate(urls):
        if len(downloaded) >= max_count:
            break
        if _is_blocked(url):
            continue
        try:
            resp = requests.get(url, headers=headers, timeout=DOWNLOAD_TIMEOUT, stream=True)
            if resp.status_code != 200:
                continue
            content_type = resp.headers.get("content-type", "")
            if "image" not in content_type and "jpeg" not in content_type and "png" not in content_type:
                continue

            # Write to temp file
            fname = _url_to_filename(url, idx)
            fpath = dest_dir / fname
            with open(fpath, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)

            # Check dimensions
            try:
                from PIL import Image as PILImage
                with PILImage.open(fpath) as img:
                    w, h = img.size
                    if w < MIN_DIMENSION or h < MIN_DIMENSION:
                        fpath.unlink(missing_ok=True)
                        continue
                    # Convert to RGB JPEG
                    if img.mode not in ("RGB", "L"):
                        img = img.convert("RGB")
                    img.save(fpath, "JPEG", quality=90)
            except Exception:
                fpath.unlink(missing_ok=True)
                continue

            downloaded.append(str(fpath))

        except Exception as e:
            logger.debug(f"[DL] Failed {url[:60]}: {e}")
            continue

    logger.info(f"[DL] Downloaded {len(downloaded)} candidates to {dest_dir}")
    return downloaded
