"""
Substack RSS/HTML Parsing Script

Fetches a Substack RSS feed, cleans HTML into stable text, and writes Markdown
files that can be ingested with src/ingest.py.

Usage:
    python src/parse_substack.py --feed-url https://example.substack.com/feed
    python src/parse_substack.py --publication-base-url https://example.substack.com
    python src/parse_substack.py --dry-run
"""

import argparse
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode

import feedparser
import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from dotenv import load_dotenv

# Add src/ to Python path for imports when running from repo root
sys.path.insert(0, str(Path(__file__).parent))

from config import SUBSTACK_FEED_URL, SUBSTACK_PUBLICATION_NAME

load_dotenv()

# ============================================================================
# Parsing Configuration
# ============================================================================

TRACKING_PARAM_PREFIXES = ("utm_",)
TRACKING_PARAM_NAMES = {"gclid", "fbclid", "mc_cid", "mc_eid", "source", "s"}

BOILERPLATE_PHRASES = (
    "subscribe",
    "share",
    "comments",
    "leave a comment",
    "get the app",
    "upgrade to paid",
    "paid subscriber",
    "sign in",
    "sign up",
)

PAYWALL_PHRASES = (
    "this post is for paid subscribers",
    "to keep reading",
    "upgrade to paid",
    "paid subscribers",
    "subscribe to read",
    "paid subscription",
)

# ============================================================================
# Feed Helpers
# ============================================================================


def resolve_feed_url(feed_url: Optional[str]) -> str:
    """Resolve the feed URL from CLI args or environment configuration."""
    if feed_url:
        return feed_url
    if SUBSTACK_FEED_URL:
        return SUBSTACK_FEED_URL
    if SUBSTACK_PUBLICATION_NAME:
        return f"https://{SUBSTACK_PUBLICATION_NAME}.substack.com/feed"
    raise ValueError(
        "❌ No feed URL provided\n"
        "💡 Pass --feed-url or set SUBSTACK_FEED_URL / SUBSTACK_PUBLICATION_NAME in .env"
    )


def load_feed_entries(feed_url: str) -> List[Dict[str, Any]]:
    """Parse RSS feed and return entries list."""
    parsed = feedparser.parse(feed_url)
    if parsed.bozo:
        raise ValueError(f"❌ Failed to parse RSS feed: {parsed.bozo_exception}")
    return list(parsed.entries)


def parse_entry_date(entry: Dict[str, Any]) -> Optional[datetime]:
    """Parse entry date from standard RSS fields."""
    if entry.get("published_parsed"):
        return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
    if entry.get("updated_parsed"):
        return datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
    if entry.get("published"):
        try:
            parsed = parsedate_to_datetime(entry.published)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except Exception:
            return None
    return None


def parse_since_date(since_days: Optional[int], since_date: Optional[str]) -> Optional[datetime]:
    """Parse CLI time filter options into a datetime."""
    if since_date:
        parsed = datetime.fromisoformat(since_date)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    if since_days is not None:
        return datetime.now(timezone.utc) - timedelta(days=since_days)
    return None


# ============================================================================
# URL Helpers
# ============================================================================


def normalize_url(url: str, publication_base_url: Optional[str] = None) -> str:
    """Normalize Substack URLs and remove tracking parameters."""
    if not url:
        return ""
    if publication_base_url:
        url = urljoin(publication_base_url, url)
    parsed = urlparse(url)
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    if path.endswith("/") and len(path) > 1:
        path = path[:-1]

    filtered_params = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        key_lower = key.lower()
        if key_lower.startswith(TRACKING_PARAM_PREFIXES) or key_lower in TRACKING_PARAM_NAMES:
            continue
        filtered_params.append((key, value))

    cleaned_query = urlencode(filtered_params, doseq=True)
    cleaned = parsed._replace(scheme=scheme, netloc=netloc, path=path, query=cleaned_query, fragment="")
    return urlunparse(cleaned)


# ============================================================================
# Content Extraction
# ============================================================================


def extract_entry_html(entry: Dict[str, Any]) -> Tuple[str, str]:
    """Prefer full HTML content, fall back to summary."""
    if entry.get("content"):
        for item in entry.content:
            value = item.get("value")
            if value:
                return "content", value
    summary_detail = entry.get("summary_detail") or {}
    if summary_detail.get("value"):
        return "summary", summary_detail["value"]
    return "summary", entry.get("summary", "")


def fetch_full_html(url: str) -> Optional[str]:
    """Fetch full HTML from the canonical URL."""
    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            return None
        return response.text
    except Exception:
        return None


def looks_truncated(text: str) -> bool:
    """Heuristic to detect truncated RSS excerpts."""
    if not text:
        return True
    lowered = text.lower()
    if "continue reading" in lowered or "read more" in lowered:
        return True
    if text.strip().endswith(("...", "…")):
        return True
    return len(text.strip()) < 800


def looks_paywalled(raw_html: str, cleaned_text: str) -> bool:
    """Heuristic to detect paywalled content."""
    haystack = f"{raw_html}\n{cleaned_text}".lower()
    return any(phrase in haystack for phrase in PAYWALL_PHRASES)


# ============================================================================
# HTML Cleaning
# ============================================================================


def normalize_whitespace(text: str) -> str:
    text = unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_boilerplate_elements(soup: BeautifulSoup) -> None:
    """Remove common boilerplate or non-content elements."""
    for tag_name in ("script", "style", "noscript", "svg", "form", "button", "input"):
        for element in soup.find_all(tag_name):
            element.decompose()

    for element in soup.find_all(True):
        if element.name in {"nav", "footer", "header", "aside"}:
            element.decompose()
            continue
        if getattr(element, "attrs", None) is None:
            continue
        if element.get("aria-hidden") == "true":
            element.decompose()
            continue
        style_attr = (element.get("style") or "").lower().replace(" ", "")
        if "display:none" in style_attr or "visibility:hidden" in style_attr:
            element.decompose()
            continue
        attr_text = " ".join(
            filter(None, [element.get("class") and " ".join(element.get("class")), element.get("id")])
        ).lower()
        if any(keyword in attr_text for keyword in ("subscribe", "signup", "footer", "nav", "comment", "share")):
            element.decompose()
            continue
        text = element.get_text(" ", strip=True).lower()
        if any(phrase in text for phrase in BOILERPLATE_PHRASES) and len(text) < 120:
            element.decompose()


def extract_article_root(soup: BeautifulSoup) -> Tag:
    """Pick a likely article container before walking the tree."""
    article = soup.find("article")
    if article:
        return article
    for selector in ("div.post", "div.post-content", "div.pencraft", "div.body", "div.post-body"):
        match = soup.select_one(selector)
        if match:
            return match
    return soup.body or soup


def element_text(element: Tag) -> str:
    return unescape(element.get_text(" ", strip=True))


def html_to_structured_text(html: str) -> str:
    """Convert HTML to stable, markdown-friendly plain text."""
    soup = BeautifulSoup(html, "html.parser")
    remove_boilerplate_elements(soup)
    root = extract_article_root(soup)

    blocks: List[str] = []

    def add_block(text: str) -> None:
        cleaned = normalize_whitespace(text)
        if cleaned:
            blocks.append(cleaned)

    for child in root.descendants:
        if isinstance(child, NavigableString):
            continue
        if not isinstance(child, Tag):
            continue
        if child.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(child.name[1])
            heading = element_text(child)
            if heading:
                add_block(f"{'#' * level} {heading}")
        elif child.name == "p":
            add_block(element_text(child))
        elif child.name in {"ul", "ol"}:
            for li in child.find_all("li", recursive=False):
                item = element_text(li)
                if item:
                    add_block(f"- {item}")
        elif child.name == "blockquote":
            quoted = element_text(child)
            if quoted:
                add_block("\n".join([f"> {line}" for line in quoted.splitlines() if line.strip()]))
        elif child.name == "pre":
            code_text = child.get_text("\n", strip=True)
            if code_text:
                add_block(f"```\n{code_text}\n```")

    return "\n\n".join(blocks)


def build_clean_text(
    raw_html: str,
    canonical_url: str,
    fetch_full: bool,
) -> Tuple[str, bool]:
    """Clean RSS HTML, optionally fetching full HTML when truncated."""
    cleaned_text = html_to_structured_text(raw_html)
    truncated = looks_truncated(cleaned_text)

    if truncated and fetch_full and canonical_url:
        full_html = fetch_full_html(canonical_url)
        if full_html:
            full_text = html_to_structured_text(full_html)
            if full_text and len(full_text) > len(cleaned_text):
                return full_text, False
    return cleaned_text, truncated


def log_entry_details(title: str, url: str, field_used: str, truncated: bool, cleaned_len: int) -> None:
    truncated_flag = "yes" if truncated else "no"
    print(f"🧾 {title}")
    print(f"   url: {url}")
    print(f"   field: {field_used}")
    print(f"   truncated: {truncated_flag}")
    print(f"   cleaned_length: {cleaned_len}")


# ============================================================================
# Markdown Output
# ============================================================================


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug or "substack-post"


def filename_from_url_or_title(canonical_url: str, title: str) -> str:
    if canonical_url:
        path = urlparse(canonical_url).path.strip("/")
        if path:
            last_segment = path.split("/")[-1]
            if last_segment:
                return slugify(last_segment)
    return slugify(title)


def build_markdown(title: str, cleaned_text: str) -> str:
    if cleaned_text.lstrip().startswith("# "):
        return cleaned_text.strip() + "\n"
    return f"# {title}\n\n{cleaned_text.strip()}\n"


def write_markdown_file(output_dir: Path, filename: str, content: str, overwrite: bool) -> Optional[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{filename}.md"
    if file_path.exists() and not overwrite:
        return None
    file_path.write_text(content, encoding="utf-8")
    return file_path


# ============================================================================
# Entry Processing
# ============================================================================


def parse_entry(
    entry: Dict[str, Any],
    publication_base_url: Optional[str],
    fetch_full_html_flag: bool,
    output_dir: Optional[Path],
    dry_run: bool,
    overwrite: bool,
    skip_paid: bool,
) -> str:
    field_used, raw_html = extract_entry_html(entry)
    source_url = entry.get("link") or entry.get("id") or ""
    canonical_url = normalize_url(source_url, publication_base_url)
    title = entry.get("title", "Untitled").strip() or "Untitled"

    if not raw_html:
        print(f"⚠️ Skipping empty entry: {title}")
        return "skipped"
    if not canonical_url:
        print(f"⚠️ Skipping entry with no URL: {title}")
        return "skipped"

    cleaned_text, truncated = build_clean_text(raw_html, canonical_url, fetch_full_html_flag)
    log_entry_details(title, canonical_url, field_used, truncated, len(cleaned_text))

    if skip_paid and looks_paywalled(raw_html, cleaned_text):
        print("   ⏭️ Paywalled post detected, skipping")
        return "skipped_paid"

    if dry_run:
        sample = cleaned_text[:500] + ("..." if len(cleaned_text) > 500 else "")
        print("   sample:")
        print(f"   {sample.replace(os.linesep, ' ')}")
        return "dry_run"

    if not cleaned_text.strip():
        print(f"⚠️ Skipping entry with no clean text: {title}")
        return "skipped"

    markdown = build_markdown(title, cleaned_text)
    filename = filename_from_url_or_title(canonical_url, title)
    file_path = write_markdown_file(output_dir, filename, markdown, overwrite)
    if not file_path:
        print("   ⏭️ File exists, skipping")
        return "skipped"
    print(f"   ✅ Wrote: {file_path}")
    return "written"


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse Substack RSS feed into Markdown files.")
    parser.add_argument("--feed-url", help="Substack RSS feed URL")
    parser.add_argument("--since-days", type=int, help="Only parse posts from the last N days")
    parser.add_argument("--since-date", help="Only parse posts after this date (YYYY-MM-DD)")
    parser.add_argument("--publication-base-url", help="Base URL for canonicalization")
    parser.add_argument("--dry-run", action="store_true", help="Print 1-3 cleaned samples without writing files")
    parser.add_argument("--fetch-full-html", action="store_true", help="Fetch full HTML when RSS looks truncated")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of entries to process")
    parser.add_argument("--output-dir", default="./data/substack", help="Output directory for Markdown files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--skip-paid", action="store_true", help="Skip paywalled posts and log them")
    args = parser.parse_args()

    feed_url = resolve_feed_url(args.feed_url)
    since_date = parse_since_date(args.since_days, args.since_date)
    publication_base_url = args.publication_base_url or ""
    output_dir = Path(args.output_dir)

    entries = load_feed_entries(feed_url)
    print(f"🔗 Feed: {feed_url}")
    print(f"🧾 Entries: {len(entries)}")
    if since_date:
        print(f"📆 Since: {since_date.isoformat()}")
    if args.fetch_full_html:
        print("🌐 Fetch full HTML: enabled")
    if not args.dry_run:
        print(f"📁 Output dir: {output_dir}")

    processed = 0
    successes = 0
    skipped_paid = 0
    seen_urls: set[str] = set()

    for entry in entries:
        if args.limit and processed >= args.limit:
            break
        entry_date = parse_entry_date(entry)
        if since_date and entry_date and entry_date < since_date:
            continue

        canonical_url = normalize_url(entry.get("link", ""), publication_base_url)
        if not canonical_url:
            print("⚠️ Skipping entry with no URL")
            continue
        if canonical_url in seen_urls:
            print(f"⏭️ Duplicate entry detected, skipping: {canonical_url}")
            continue
        seen_urls.add(canonical_url)

        status = parse_entry(
            entry,
            publication_base_url,
            args.fetch_full_html,
            output_dir,
            args.dry_run,
            args.overwrite,
            args.skip_paid,
        )
        if status == "written":
            successes += 1
        elif status == "skipped_paid":
            skipped_paid += 1
        processed += 1
        print()
        if args.dry_run and processed >= 3:
            break

    print("=" * 60)
    print("📊 SUBSTACK PARSE COMPLETE")
    print("=" * 60)
    print(f"✅ Successful: {successes}")
    print(f"📄 Processed: {processed}")
    if skipped_paid:
        print(f"🔒 Paywalled skipped: {skipped_paid}")
    if not args.dry_run:
        print("➡️ Run: python src/ingest.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Parsing interrupted by user")
        sys.exit(0)
    except ValueError as e:
        print(f"\n{e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}\n")
        print("💡 Check the error message above and your configuration")
        sys.exit(1)
