"""
Plex media marker creation module.

Creates timed markers on Plex media items via the Plex HTTP API.
Markers allow users to skip triggering scenes during playback.
"""
from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)

# Valid marker type strings accepted by the Plex HTTP API.
# NOTE: Only "bookmark" can be created via the HTTP API by external clients.
# "intro", "commercial", "credits", and "resume" are reserved for Plex's
# internal AI analysis engine and always return HTTP 400 when POSTed externally.
# Use MARKER_TYPE=bookmark in your config for HTTP-based marker creation.
VALID_MARKER_TYPES: set[str] = {"intro", "commercial", "bookmark", "resume", "credits"}

# Normalise aliases to canonical names.
_MARKER_TYPE_ALIASES: dict[str, str] = {
    "credit": "credits",
}


def _ms_to_timecode(ms: int) -> str:
    """Convert milliseconds to MM:SS."""
    total_s = ms // 1000
    return f"{total_s // 60:02d}:{total_s % 60:02d}"


def get_existing_markers(plex_url: str, plex_token: str, metadata_id: str) -> list[dict]:
    """Fetch all existing markers for a Plex media item.

    Uses the metadata endpoint with includeMarkers=1 — the /markers sub-path
    returns 404 on most Plex server versions.
    """
    url = f"{plex_url}/library/metadata/{metadata_id}"
    try:
        resp = requests.get(
            url,
            params={"X-Plex-Token": plex_token, "includeMarkers": 1},
            headers={"Accept": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        items = resp.json().get("MediaContainer", {}).get("Metadata", [])
        return items[0].get("Marker", []) if items else []
    except Exception as e:
        logger.debug(f"Could not fetch existing markers for item {metadata_id}: {e}")
        return []


def _marker_already_exists(markers: list[dict], start_ms: int, end_ms: int, tolerance_ms: int = 5000) -> bool:
    """Return True if a marker already exists within `tolerance_ms` of the given range."""
    for m in markers:
        if (
            abs(m.get("startTimeOffset", -999999) - start_ms) <= tolerance_ms
            and abs(m.get("endTimeOffset", -999999) - end_ms) <= tolerance_ms
        ):
            return True
    return False


def create_marker(
    plex_url: str,
    plex_token: str,
    metadata_id: str,
    marker_type_str: str,
    start_time_ms: int,
    end_time_ms: int,
    title: str,
    dry_run: bool = False,
    existing_markers: list[dict] | None = None,
) -> str | None:
    """Create a timed marker on a Plex media item.

    Returns a human-readable description string if the marker was (or would be)
    created, or None if it was skipped (already exists, bad type, etc.).
    Pass pre-fetched `existing_markers` to avoid re-fetching per scene.
    """
    normalised = _MARKER_TYPE_ALIASES.get(marker_type_str.lower(), marker_type_str.lower())
    if normalised not in VALID_MARKER_TYPES:
        logger.error(
            f"Unknown marker type '{marker_type_str}'. "
            f"Valid types: {sorted(VALID_MARKER_TYPES)}"
        )
        return None

    start_tc = _ms_to_timecode(start_time_ms)
    end_tc = _ms_to_timecode(end_time_ms)

    # Idempotency: skip if a marker already covers the same range
    markers_to_check = existing_markers if existing_markers is not None else \
        get_existing_markers(plex_url, plex_token, metadata_id)
    if _marker_already_exists(markers_to_check, start_time_ms, end_time_ms):
        logger.debug(f"Marker already exists at {start_tc}–{end_tc} on item {metadata_id}, skipping.")
        return None

    description = f"{marker_type_str} {start_tc}–{end_tc} — {title}"

    if dry_run:
        return description

    url = f"{plex_url}/library/metadata/{metadata_id}/marker"
    params = {
        "X-Plex-Token": plex_token,
        "type": normalised,
        "startTimeOffset": start_time_ms,
        "endTimeOffset": end_time_ms,
        "attributes[title]": title,
    }

    try:
        resp = requests.post(url, params=params, timeout=10)
        resp.raise_for_status()
        return description
    except requests.HTTPError as e:
        status = e.response.status_code
        body = e.response.text[:300]
        logger.error(f"Failed to create marker on {metadata_id}: HTTP {status} — {body}")
        return None
    except Exception as e:
        logger.error(f"Failed to create marker on {metadata_id}: {e}")
        return None


def create_markers_for_scenes(
    plex_url: str,
    plex_token: str,
    metadata_id: str,
    marker_type_str: str,
    scenes: list[dict],
    dry_run: bool = False,
    min_duration_ms: int = 0,
) -> list[str]:
    """Create Plex markers for a list of scene dicts (as returned by gemini_extractor).

    Args:
        min_duration_ms: Skip scenes shorter than this many milliseconds.
                         0 (default) = accept any duration.

    Returns:
        List of human-readable description strings for every marker created
        (or that would be created in dry_run mode).
    """
    # Fetch existing markers once for idempotency checks
    existing = get_existing_markers(plex_url, plex_token, metadata_id)

    created: list[str] = []
    for scene in scenes:
        start_ms = scene.get("start_time_ms")
        end_ms = scene.get("end_time_ms")
        description = scene.get("description", "")
        topic = scene.get("trigger_topic", "")

        if start_ms is None or end_ms is None:
            logger.warning(f"Scene missing start/end time, skipping: {scene}")
            continue

        duration_ms = end_ms - start_ms
        if min_duration_ms > 0 and duration_ms < min_duration_ms:
            logger.debug(
                f"Scene too short ({duration_ms}ms < {min_duration_ms}ms), skipping: {topic}"
            )
            continue

        start_tc = _ms_to_timecode(start_ms)
        end_tc = _ms_to_timecode(end_ms)
        if description:
            title = f"Trigger: {topic} — {description[:60]} ({start_tc}–{end_tc})"
        else:
            title = f"Trigger: {topic} ({start_tc}–{end_tc})"

        desc = create_marker(
            plex_url=plex_url,
            plex_token=plex_token,
            metadata_id=metadata_id,
            marker_type_str=marker_type_str,
            start_time_ms=start_ms,
            end_time_ms=end_ms,
            title=title,
            dry_run=dry_run,
            existing_markers=existing,
        )
        if desc is not None:
            created.append(desc)

    return created
