"""Local page extraction for Hermes ``web_extract`` — no external service.

Why: DATA's Mac has 8 GB. The self-hosted Firecrawl stack held 1.7 GB resident
for ~one extraction a day and starved the FaceTime voice pipeline (2026-09-12
measurements: TTS 3-6 s/sentence with it running, 0.3-0.9 s without). Cloud
extractors would send every page DATA reads through a third party. This
fetches the page in-process and reduces it to readable text with the stdlib
parser: zero memory when idle, same privacy as self-hosting.

Limits: no JavaScript rendering (use the ``browser`` tool for SPA pages);
no PDF. Result shape follows agent/web_search_provider.py.
"""
from __future__ import annotations

import asyncio
import html
import logging
import re
from html.parser import HTMLParser
from typing import Any, Dict, List

import httpx

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

MAX_BYTES = 5_000_000
TIMEOUT_S = 20.0
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 "
      "(KHTML, like Gecko) Version/17.0 Safari/605.1.15")
_SKIP = {"script", "style", "noscript", "svg", "canvas", "iframe", "template",
         "nav", "header", "footer", "aside", "form", "button", "select", "option"}
_BLOCK = {"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6",
          "tr", "table", "section", "article", "main", "blockquote", "pre", "hr", "dd", "dt"}


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False
        self._main_seen = False

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in ("main", "article"):
            self._main_seen = True
        if tag in _BLOCK:
            self.parts.append("\n")
        if tag.startswith("h") and len(tag) == 2 and tag[1].isdigit():
            self.parts.append("\n" + "#" * int(tag[1]) + " ")

    def handle_endtag(self, tag):
        if tag in _SKIP and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False
        if tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def html_to_text(raw: str) -> tuple[str, str]:
    p = _Text()
    try:
        p.feed(raw)
        p.close()
    except Exception:  # noqa: BLE001 — malformed markup: keep what we have
        pass
    text = "".join(p.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return html.unescape(p.title).strip(), text


def _fetch_one(client: httpx.Client, url: str) -> Dict[str, Any]:
    try:
        with client.stream("GET", url) as r:
            ctype = r.headers.get("content-type", "")
            if r.status_code >= 400:
                return {"url": url, "title": "", "error": f"HTTP {r.status_code}"}
            buf = bytearray()
            for chunk in r.iter_bytes():
                buf += chunk
                if len(buf) > MAX_BYTES:
                    break
            final_url = str(r.url)
        body = bytes(buf).decode(r.encoding or "utf-8", errors="replace")
        if "html" in ctype or body.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
            title, text = html_to_text(body)
        else:
            title, text = "", body.strip()
        if not text:
            return {"url": url, "title": title, "error": "no extractable text (JavaScript-rendered page? try the browser tool)"}
        return {"url": final_url, "title": title, "content": text, "raw_content": text,
                "metadata": {"content_type": ctype, "bytes": len(buf), "extractor": "local"}}
    except httpx.HTTPError as exc:
        return {"url": url, "title": "", "error": f"fetch failed: {exc.__class__.__name__}: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"url": url, "title": "", "error": f"extract failed: {exc}"}


class LocalExtractProvider(WebSearchProvider):
    name = "local"
    display_name = "Local (in-process, no service)"

    def is_available(self) -> bool:
        return True

    def supports_search(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return True

    async def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        def _run() -> List[Dict[str, Any]]:
            with httpx.Client(headers={"User-Agent": UA, "Accept": "text/html,application/xhtml+xml,*/*;q=0.8"},
                              follow_redirects=True, timeout=TIMEOUT_S) as client:
                return [_fetch_one(client, u) for u in urls]
        return await asyncio.to_thread(_run)

    def get_setup_schema(self) -> Dict[str, Any]:
        return {"name": "Local extract", "badge": "local", "tag": "no service, no key", "env_vars": []}
