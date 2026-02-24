"""
Gemini LLM-based timestamp extractor for DoesTheDogDie comments.

Sends user comments for a trigger topic to Gemini and asks it to identify
any timestamps mentioned for the start/end of the triggering scene.

Results are cached locally so the same topic+comments are never sent to
Gemini twice (saves cost and time on re-runs).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path

CACHE_DIR = Path(__file__).parent / ".cache"
logger = logging.getLogger(__name__)


def _cache_key(topic_item_id: int | None, comments: list[dict]) -> str:
    """Stable cache key based on the DTDD topicItemId + comment text fingerprint."""
    comment_blob = "|".join(sorted(c.get("comment", "") for c in comments))
    fingerprint = hashlib.md5(comment_blob.encode()).hexdigest()[:12]
    prefix = f"tid{topic_item_id}" if topic_item_id else "tidX"
    return f"gemini_{prefix}_{fingerprint}"


def _read_cache(key: str, ttl: int) -> dict | None:
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if time.time() - data.get("_cached_at", 0) < ttl:
            return data.get("_payload")
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def _write_cache(key: str, payload: dict):
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps({
        "_cached_at": time.time(),
        "_payload": payload,
    }, indent=2))


def extract_timestamps(
    comments: list[dict],
    topic_name: str,
    media_title: str,
    api_key: str,
    cache_ttl: int = 604800,
    topic_item_id: int | None = None,
    model: str = "gemini-3.1-pro-preview",
) -> dict:
    """Send DTDD comments to Gemini to extract scene timestamps.

    Results are cached locally. Pass topic_item_id (from DTDD stat) for a
    stable cache key; falls back to a hash of comment content.

    Args:
        comments: List of comment dicts from DTDD (each has a "comment" key).
        topic_name: The trigger topic name (e.g. "an animal dies").
        media_title: The media title for context (e.g. "House of the Dragon S01E02").
        api_key: Gemini API key.
        cache_ttl: How long to cache results (seconds). Default: 7 days.
        topic_item_id: DTDD topicItemId for stable cache keying.

    Returns:
        {
            "found": bool,
            "scenes": [
                {
                    "start_time_ms": int,
                    "end_time_ms": int,
                    "description": str,
                    "trigger_topic": str
                }
            ]
        }
    """
    if not comments:
        return {"found": False, "scenes": []}

    comment_texts = [c.get("comment", "").strip() for c in comments if c.get("comment", "").strip()]
    if not comment_texts:
        return {"found": False, "scenes": []}

    # Check cache before calling the API
    key = _cache_key(topic_item_id, comments)
    cached = _read_cache(key, cache_ttl)
    if cached is not None:
        logger.debug(f"Gemini cache hit for topic '{topic_name}'")
        return cached

    try:
        from google import genai
    except ImportError:
        logger.error("google-genai not installed. Run: pip install google-genai")
        return {"found": False, "scenes": []}

    client = genai.Client(api_key=api_key)
    comment_block = "\n".join(f"- {t}" for t in comment_texts)

    prompt = f"""You are analyzing user comments from DoesTheDogDie.com about the trigger topic \
"{topic_name}" in "{media_title}".

Your task: identify any specific timestamps that commenters mentioned for when this \
triggering scene starts and/or ends.

Comments:
{comment_block}

Rules:
- Accept any time format: 18:43, ~18:43, @18:43, 18min, 18 minutes, 18m43s, etc.
- Convert all times to milliseconds (e.g. 18:43 → 18*60000 + 43*1000 = 1123000).
- If only a start time is mentioned, estimate end from context clues \
(e.g. "lasts ~1 min" → end = start + 60000).
- Include a scene only when you have reasonable confidence in the timestamp.
- Multiple scenes are possible if multiple commenters mention different timestamps.

Respond ONLY with valid JSON (no markdown fences, no extra text):
{{
  "found": true,
  "scenes": [
    {{
      "start_time_ms": <integer>,
      "end_time_ms": <integer>,
      "description": "<brief description>",
      "trigger_topic": "{topic_name}"
    }}
  ]
}}

If no timestamps are found, respond with exactly:
{{"found": false, "scenes": []}}"""

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        text = response.text.strip()

        # Strip markdown code fences if the model adds them anyway
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        result = json.loads(text)
        scene_count = len(result.get("scenes", []))
        if result.get("found") and scene_count:
            logger.debug(f"Gemini found {scene_count} scene(s) for topic '{topic_name}'")
        else:
            logger.debug(f"Gemini: no timestamps for topic '{topic_name}'")

        _write_cache(key, result)
        return result

    except json.JSONDecodeError as e:
        logger.warning(f"Gemini returned non-JSON for topic '{topic_name}': {e}")
        return {"found": False, "scenes": []}
    except Exception as e:
        logger.error(f"Gemini API error for topic '{topic_name}': {e}")
        return {"found": False, "scenes": []}
