"""Bright Data Reddit harvest for phoenix2pytest dataset expansion.

Modes:
  --mode subreddit  : discover posts within a list of LLM-focused subreddits
                      (community-precise, content needs post-filter)
  --mode keyword    : discover posts across all Reddit by failure-related keyword
                      (content-precise, community is noisy)

Workflow:
  1. trigger_collection(...) -> snapshot_id
  2. poll_snapshot(snapshot_id) until "ready"
  3. download_snapshot(snapshot_id) -> list of raw BD Reddit Post records
  4. filter_llm_failure_posts(...) keeps records whose title or body
     mention LLM-failure cues (hallucination, fabricated, wrong answer, ...)
  5. normalise_posts(...) maps raw records to phoenix2pytest demo_dataset schema

Usage:
  Pilot (cheap, ~5 records):
    python scripts/bd_harvest.py --pilot

  Single subreddit harvest:
    python scripts/bd_harvest.py --mode subreddit \
        --subreddit ChatGPT --limit 20 \
        --output tests/data/bd_samples/r-chatgpt-hot.json

  Multi-subreddit harvest with content filter:
    python scripts/bd_harvest.py --mode subreddit \
        --subreddit ChatGPT --subreddit ClaudeAI --subreddit OpenAI \
        --subreddit LocalLLaMA --subreddit GeminiAI \
        --limit 30 --filter-failure \
        --output tests/data/bd_samples/llm-failures-batch.json

  Keyword harvest (across-Reddit):
    python scripts/bd_harvest.py --mode keyword \
        --keyword "ChatGPT hallucination" --keyword "GPT-4 made up" \
        --limit 20 --filter-failure \
        --output tests/data/bd_samples/keyword-failures.json

Env:
  BRIGHTDATA_API_KEY  required, loaded from repo-root .env
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

REDDIT_POSTS_DATASET_ID = "gd_lvz8ah06191smkebj4"
BD_BASE = "https://api.brightdata.com"
POLL_INTERVAL_SEC = 10
DEFAULT_TIMEOUT_SEC = 600

FAILURE_KEYWORDS = [
    # Curated failure cues. Tuned for precision over recall: dropped
    # generic terms like "json", "incorrect", "regress", "malformed"
    # because they match technical posts that are not failure narratives.
    "hallucinat",  # hallucinate, hallucination, hallucinated
    "fabricat",  # fabricated, fabrication
    "made up",
    "made-up",
    "wrong answer",
    "gave wrong",
    "lied",
    "confidently wrong",
    "false citation",
    "fake citation",
    "fake source",
    "wrong code",
    "broken code",
    "refused to answer",
    "refused to help",
    "got it wrong",
    "completely wrong",
    "invented",
    "fictional source",
    "gaslit",
]

DEFAULT_LLM_SUBREDDITS = [
    "ChatGPT",
    "OpenAI",
    "ClaudeAI",
    "LocalLLaMA",
    "LangChain",
    "GeminiAI",
    "MachineLearning",
    "OpenAIDev",
    "PromptEngineering",
    "Agent_AI",
    "ChatGPTCoding",
    "ArtificialInteligence",
]


def load_api_key() -> str:
    load_dotenv()
    key = os.getenv("BRIGHTDATA_API_KEY", "").strip()
    if not key:
        raise RuntimeError("BRIGHTDATA_API_KEY not set in .env or environment")
    return key


def build_subreddit_inputs(subreddits: list[str], sort_by: str = "Hot") -> list[dict[str, str]]:
    """Build trigger input array for discover_by=subreddit_url mode."""
    return [{"url": f"https://www.reddit.com/r/{name}", "sort_by": sort_by} for name in subreddits]


def build_keyword_inputs(
    keywords: list[str], date_filter: str = "Past month"
) -> list[dict[str, str]]:
    """Build trigger input array for discover_by=keyword mode."""
    return [{"keyword": kw, "sort_by": "New", "date": date_filter} for kw in keywords]


def trigger_collection(
    api_key: str,
    *,
    discover_by: str,
    inputs: list[dict[str, Any]],
    limit_per_input: int = 20,
    dataset_id: str = REDDIT_POSTS_DATASET_ID,
) -> str:
    url = f"{BD_BASE}/datasets/v3/trigger"
    params = {
        "dataset_id": dataset_id,
        "type": "discover_new",
        "discover_by": discover_by,
        "limit_per_input": str(limit_per_input),
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, params=params, headers=headers, json=inputs, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    snapshot_id = payload.get("snapshot_id")
    if not snapshot_id:
        raise RuntimeError(f"No snapshot_id in trigger response: {payload}")
    return snapshot_id


def poll_snapshot(
    api_key: str,
    snapshot_id: str,
    *,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    interval_sec: int = POLL_INTERVAL_SEC,
) -> dict[str, Any]:
    url = f"{BD_BASE}/datasets/v3/progress/{snapshot_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    deadline = time.time() + timeout_sec
    last: dict[str, Any] = {}
    while time.time() < deadline:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        last = resp.json()
        status = (last.get("status") or "").lower()
        if status == "ready":
            return last
        if status in {"failed", "error"}:
            raise RuntimeError(f"Snapshot {snapshot_id} failed: {last}")
        time.sleep(interval_sec)
    raise TimeoutError(f"Snapshot {snapshot_id} did not finish in {timeout_sec}s: {last}")


def download_snapshot(api_key: str, snapshot_id: str) -> list[dict[str, Any]]:
    url = f"{BD_BASE}/datasets/v3/snapshot/{snapshot_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"format": "json"}
    resp = requests.get(url, headers=headers, params=params, timeout=60)
    resp.raise_for_status()
    body = resp.text.strip()
    if not body:
        return []
    if body.startswith("["):
        return json.loads(body)
    return [json.loads(line) for line in body.splitlines() if line.strip()]


_KW_RE = re.compile("|".join(re.escape(k) for k in FAILURE_KEYWORDS), re.IGNORECASE)


def looks_like_failure_post(post: dict[str, Any]) -> bool:
    """True when title or body mentions failure-related keywords."""
    title = post.get("title") or ""
    body = post.get("description") or post.get("description_markdown") or ""
    haystack = f"{title}\n{body}"
    if len(body) < 80:
        return False
    return bool(_KW_RE.search(haystack))


def filter_llm_failure_posts(
    posts: list[dict[str, Any]],
    *,
    community_allowlist: set[str] | None = None,
) -> list[dict[str, Any]]:
    kept = []
    for p in posts:
        if p.get("title") is None:
            continue
        if community_allowlist and (p.get("community_name") or "") not in community_allowlist:
            continue
        if not looks_like_failure_post(p):
            continue
        kept.append(p)
    return kept


def normalise_post(post: dict[str, Any]) -> dict[str, Any] | None:
    title = (post.get("title") or "").strip()
    body = (post.get("description") or post.get("description_markdown") or "").strip()
    if not body or len(body) < 40:
        return None
    url = post.get("url") or ""
    post_id = post.get("post_id") or url
    return {
        "id": f"reddit-{post_id}",
        "failure_mode": None,
        "source": "reddit",
        "community": post.get("community_name"),
        "user_prompt": title,
        "llm_output": body,
        "ideal_behavior": None,
        "demo_featured": False,
        "raw_url": url,
        "upvotes": post.get("num_upvotes"),
        "num_comments": post.get("num_comments"),
        "date_posted": post.get("date_posted"),
    }


def normalise_posts(posts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [n for p in posts if (n := normalise_post(p)) is not None]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harvest Reddit posts via Bright Data.")
    parser.add_argument("--mode", choices=["subreddit", "keyword"], default="subreddit")
    parser.add_argument(
        "--subreddit",
        action="append",
        default=None,
        help="Subreddit name (repeatable). Defaults to LLM allowlist.",
    )
    parser.add_argument(
        "--keyword",
        action="append",
        default=None,
        help="Search keyword (repeatable, only in --mode keyword).",
    )
    parser.add_argument(
        "--sort-by", default="Hot", choices=["Hot", "Top", "New"], help="Subreddit listing sort."
    )
    parser.add_argument(
        "--date-filter",
        default="Past month",
        help="Keyword mode date filter, default 'Past month'.",
    )
    parser.add_argument("--limit", type=int, default=20, help="limit_per_input for BD trigger.")
    parser.add_argument(
        "--filter-failure",
        action="store_true",
        help="Keep only posts matching failure keywords in title/body.",
    )
    parser.add_argument(
        "--filter-allowlist",
        action="store_true",
        help="Also restrict to default LLM subreddit allowlist when filtering.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=None,
        help="Optional path to save raw BD records before any filter.",
    )
    parser.add_argument(
        "--pilot", action="store_true", help="Tiny test: 1 subreddit (ChatGPT), limit 5."
    )
    parser.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:  # pragma: no cover - integration
    api_key = load_api_key()

    if args.pilot:
        discover_by = "subreddit_url"
        inputs = build_subreddit_inputs(["ChatGPT"], sort_by=args.sort_by)
        limit = 5
        print(f"[pilot] subreddit=ChatGPT sort_by={args.sort_by} limit={limit}")
    elif args.mode == "subreddit":
        discover_by = "subreddit_url"
        subreddits = args.subreddit or DEFAULT_LLM_SUBREDDITS
        inputs = build_subreddit_inputs(subreddits, sort_by=args.sort_by)
        limit = args.limit
        print(
            f"[trigger] mode=subreddit n_subs={len(subreddits)} "
            f"sort_by={args.sort_by} limit_per_input={limit}"
        )
    else:
        discover_by = "keyword"
        keywords = args.keyword or ["ChatGPT hallucination"]
        inputs = build_keyword_inputs(keywords, date_filter=args.date_filter)
        limit = args.limit
        print(
            f"[trigger] mode=keyword n_kw={len(keywords)} "
            f"date={args.date_filter} limit_per_input={limit}"
        )

    snapshot_id = trigger_collection(
        api_key,
        discover_by=discover_by,
        inputs=inputs,
        limit_per_input=limit,
    )
    print(f"[snapshot] id={snapshot_id}")

    print(f"[poll] every {POLL_INTERVAL_SEC}s, timeout {args.timeout_sec}s")
    final = poll_snapshot(api_key, snapshot_id, timeout_sec=args.timeout_sec)
    rows = final.get("records", "?")
    errs = final.get("errors", "?")
    print(f"[poll] ready records={rows} errors={errs}")

    records = download_snapshot(api_key, snapshot_id)
    print(f"[download] {len(records)} raw records")

    if args.raw_output:
        args.raw_output.parent.mkdir(parents=True, exist_ok=True)
        args.raw_output.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[write] raw -> {args.raw_output}")

    if args.filter_failure:
        allow = set(DEFAULT_LLM_SUBREDDITS) if args.filter_allowlist else None
        kept = filter_llm_failure_posts(records, community_allowlist=allow)
        print(f"[filter] failure-keyword kept {len(kept)} of {len(records)}")
        records = kept

    normalised = normalise_posts(records)
    print(f"[normalise] {len(normalised)} of {len(records)} have usable body")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(normalised, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[write] {args.output}")

    if args.pilot and normalised:
        sample = normalised[0]
        preview = {
            k: sample.get(k) for k in ("id", "community", "user_prompt", "upvotes", "num_comments")
        }
        print(f"[pilot] sample: {json.dumps(preview, ensure_ascii=False)}")

    return 0


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - integration
    args = parse_args(argv)
    try:
        return run(args)
    except (RuntimeError, requests.HTTPError, TimeoutError) as exc:
        print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
