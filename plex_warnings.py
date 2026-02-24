#!/usr/bin/env python3
"""
DoesTheDogWatchPlex — Content warnings from DoesTheDogDie.com in your Plex library.

Usage:
    python plex_warnings.py              # Process all configured libraries
    python plex_warnings.py --dry-run    # Preview changes without writing
    python plex_warnings.py --clear      # Remove all content warnings from Plex
    python plex_warnings.py --clear-cache  # Clear the local DTDD API cache
    python plex_warnings.py --movie "Midsommar"  # Process a single movie by title
    python plex_warnings.py --show "House of the Dragon"  # Process a single TV show
    python plex_warnings.py --list-topics  # Show all available topic names for filtering
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from plexapi.server import PlexServer

from dtdd import DTDDClient

try:
    import config
except ImportError:
    print("ERROR: config.py not found.")
    print("Copy config.py.example to config.py and fill in your details.")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)

# Cached result of marker config validation (None = not yet checked)
_markers_config_ok: bool | None = None


# ---------------------------------------------------------------------------
# Helpers: warning text formatting
# ---------------------------------------------------------------------------

def get_separator() -> str:
    return getattr(config, "SEPARATOR", "\n\n———— Content Warnings (via DoesTheDogDie.com) ————")


def strip_warnings(summary: str) -> str:
    """Remove existing DTDD content warnings from a summary."""
    sep = get_separator()
    if sep in summary:
        return summary.split(sep)[0].rstrip()
    # Also handle the old-style separator from the original project
    if "\ndoesthedogdie:" in summary.lower():
        for i, line in enumerate(summary.split("\n")):
            if line.strip().lower().startswith("doesthedogdie:"):
                return "\n".join(summary.split("\n")[:i]).rstrip()
    return summary


def format_warnings(media_data: dict, episode_index1: int | None = None, episode_index2: int | None = None) -> str | None:
    """Extract and format trigger warnings from DTDD media response.

    For TV shows, pass episode_index1/episode_index2 to filter to a specific episode.
    Returns a formatted string of warnings, or None if no relevant warnings found.
    """
    stats = media_data.get("topicItemStats", [])
    if not stats:
        return None

    min_yes = getattr(config, "MIN_YES_VOTES", 3)
    min_ratio = getattr(config, "MIN_YES_RATIO", 0.6)

    show_nos = getattr(config, "SHOW_SAFE_TOPICS", False)
    include_topics = getattr(config, "INCLUDE_TOPICS", None)
    exclude_topics = getattr(config, "EXCLUDE_TOPICS", None)

    warnings_yes = []
    warnings_no = []

    for stat in stats:
        yes_count = stat.get("yesSum", 0)
        no_count = stat.get("noSum", 0)
        total = yes_count + no_count
        topic = stat.get("topic", {})
        topic_name = topic.get("name", "")
        topic_not_name = topic.get("notName", "")

        if total == 0 or not topic_name:
            continue

        # For TV show episode filtering: skip if this stat has no votes for our episode.
        # index1/index2 on the stat reflect the top-voted comment's episode.
        # A null index means the vote is show-wide (not episode-specific).
        if episode_index1 is not None and episode_index2 is not None:
            stat_i1 = stat.get("index1")
            stat_i2 = stat.get("index2")
            # Accept if: stat is show-wide (null) OR matches our episode
            if stat_i1 is not None and stat_i2 is not None:
                if stat_i1 != episode_index1 or stat_i2 != episode_index2:
                    continue

        # Apply topic filtering
        if include_topics is not None:
            if topic_name.lower() not in [t.lower() for t in include_topics]:
                continue
        elif exclude_topics is not None:
            if topic_name.lower() in [t.lower() for t in exclude_topics]:
                continue

        ratio = yes_count / total

        if ratio >= min_ratio and yes_count >= min_yes:
            warnings_yes.append((topic_name, yes_count, no_count))
        elif show_nos and (1 - ratio) >= min_ratio and no_count >= min_yes:
            warnings_no.append((topic_not_name, yes_count, no_count))

    if not warnings_yes and not warnings_no:
        return None

    # Translate topic names if LANGUAGE is configured
    target_lang = getattr(config, "LANGUAGE", None)
    if target_lang:
        from translate import translate_topics
        all_names = [w[0] for w in warnings_yes] + [w[0] for w in warnings_no]
        translations = translate_topics(all_names, target_lang)
    else:
        translations = None

    lines = []
    if warnings_yes:
        names = [translations[w[0]] if translations else w[0] for w in warnings_yes]
        lines.append("⚠️  " + " · ".join(names))
    if warnings_no:
        names = [translations[w[0]] if translations else w[0] for w in warnings_no]
        lines.append("✅  " + " · ".join(names))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers: marker pipeline
# ---------------------------------------------------------------------------

def _markers_enabled() -> bool:
    """Return True only if markers are enabled and all required config is present and valid.

    Validates once on first call and caches the result. Logs a single clear
    warning if markers are auto-disabled due to bad/missing config — subsequent
    calls are silent. Always returns False if ENABLE_MARKERS is False or unset.
    """
    global _markers_config_ok

    if not getattr(config, "ENABLE_MARKERS", False):
        return False

    # Return cached result after the first validation
    if _markers_config_ok is not None:
        return _markers_config_ok

    issues: list[str] = []

    # Check google-genai is importable
    try:
        import google.genai  # noqa: F401
    except ImportError:
        issues.append("google-genai is not installed (run: pip install google-genai)")

    # Check Gemini API key
    gemini_key = (getattr(config, "GEMINI_API_KEY", "") or "").strip()
    if not gemini_key or gemini_key == "your-gemini-api-key-here":
        issues.append("GEMINI_API_KEY is not configured")

    # Check marker type is a known value
    from plex_markers import VALID_MARKER_TYPES, _MARKER_TYPE_ALIASES
    marker_type = (getattr(config, "MARKER_TYPE", "") or "").strip()
    valid_types = sorted(VALID_MARKER_TYPES)
    normalised = _MARKER_TYPE_ALIASES.get(marker_type.lower(), marker_type.lower())
    if not marker_type or normalised not in VALID_MARKER_TYPES:
        issues.append(
            f"MARKER_TYPE '{marker_type}' is invalid — use one of: {', '.join(valid_types)}"
        )

    if issues:
        logger.warning("Marker pipeline auto-disabled — fix these settings to enable:")
        for issue in issues:
            logger.warning(f"  • {issue}")
        _markers_config_ok = False
        return False

    _markers_config_ok = True
    return True


def _get_marker_type() -> str:
    return getattr(config, "MARKER_TYPE", "commercial")


def _get_min_duration_ms() -> int:
    """Read MIN_MARKER_DURATION_S from config and return value in milliseconds."""
    val = getattr(config, "MIN_MARKER_DURATION_S", 0)
    try:
        return int(val) * 1000 if val else 0
    except (TypeError, ValueError):
        return 0


def _get_gemini_model() -> str:
    return getattr(config, "GEMINI_MODEL", "gemini-3.1-pro-preview")


def _get_discovery_cache_ttl() -> int:
    """TTL for the all-episodes (-1,-1) discovery call.

    Shorter than the normal CACHE_TTL so new episodes on ongoing shows
    are picked up on the next scheduled run.
    """
    val = getattr(config, "DISCOVERY_CACHE_TTL", 86400)
    try:
        return int(val) if val else 86400
    except (TypeError, ValueError):
        return 86400


def _get_gemini_key() -> str | None:
    return getattr(config, "GEMINI_API_KEY", None)


def _get_triggered_stats(media_data: dict) -> list[dict]:
    """Return topicItemStats entries where yesSum > noSum (trigger is confirmed)."""
    min_yes = getattr(config, "MIN_YES_VOTES", 3)
    min_ratio = getattr(config, "MIN_YES_RATIO", 0.6)
    include_topics = getattr(config, "INCLUDE_TOPICS", None)
    exclude_topics = getattr(config, "EXCLUDE_TOPICS", None)

    triggered = []
    for stat in media_data.get("topicItemStats", []):
        yes_count = stat.get("yesSum", 0)
        no_count = stat.get("noSum", 0)
        total = yes_count + no_count
        topic_name = stat.get("topic", {}).get("name", "")

        if total == 0 or not topic_name:
            continue

        if include_topics is not None:
            if topic_name.lower() not in [t.lower() for t in include_topics]:
                continue
        elif exclude_topics is not None:
            if topic_name.lower() in [t.lower() for t in exclude_topics]:
                continue

        ratio = yes_count / total
        if ratio >= min_ratio and yes_count >= min_yes:
            triggered.append(stat)

    return triggered


def run_marker_pipeline(
    plex_url: str,
    plex_token: str,
    metadata_id: str,
    media_data: dict,
    media_title: str,
    dry_run: bool = False,
    episode_index1: int | None = None,
    episode_index2: int | None = None,
) -> list[str]:
    """Run the Gemini → Plex marker pipeline for a media item.

    For TV episodes, pass the season/episode numbers so we only consider
    comments relevant to that specific episode.

    Returns a list of human-readable descriptions for every marker
    created (or that would be created in dry_run mode).
    """
    gemini_key = _get_gemini_key()
    from gemini_extractor import extract_timestamps
    from plex_markers import create_markers_for_scenes

    marker_type = _get_marker_type()
    min_duration = _get_min_duration_ms()
    gemini_model = _get_gemini_model()
    cache_ttl = getattr(config, "CACHE_TTL", 604800)
    triggered_stats = _get_triggered_stats(media_data)

    if not triggered_stats:
        return []

    all_descriptions: list[str] = []

    for stat in triggered_stats:
        topic_name = stat.get("topic", {}).get("name", "")
        topic_item_id = stat.get("topicItemId")
        comments = stat.get("comments", [])

        # For TV episodes: only use comments that match our episode (or show-wide)
        if episode_index1 is not None and episode_index2 is not None:
            comments = [
                c for c in comments
                if (
                    (c.get("index1") is None and c.get("index2") is None)
                    or (c.get("index1") == episode_index1 and c.get("index2") == episode_index2)
                )
            ]

        if not comments:
            continue

        result = extract_timestamps(
            comments=comments,
            topic_name=topic_name,
            media_title=media_title,
            api_key=gemini_key,
            cache_ttl=cache_ttl,
            topic_item_id=topic_item_id,
            model=gemini_model,
        )

        if result.get("found") and result.get("scenes"):
            descs = create_markers_for_scenes(
                plex_url=plex_url,
                plex_token=plex_token,
                metadata_id=metadata_id,
                marker_type_str=marker_type,
                scenes=result["scenes"],
                dry_run=dry_run,
                min_duration_ms=min_duration,
            )
            all_descriptions.extend(descs)

    return all_descriptions


# ---------------------------------------------------------------------------
# Movie processing
# ---------------------------------------------------------------------------

def match_movie(dtdd: DTDDClient, movie) -> dict | None:
    """Try to match a Plex movie to a DTDD entry.

    Strategy:
    1. Search by IMDB ID if available (most reliable)
    2. Fall back to title + year search
    3. Try title-only search as last resort

    Returns the DTDD media data dict, or None if no match.
    """
    title = movie.title
    year = movie.year

    # Try IMDB ID first (available via guids on modern Plex)
    imdb_id = None
    try:
        for guid in movie.guids:
            if guid.id.startswith("imdb://"):
                imdb_id = guid.id.replace("imdb://", "")
                break
    except Exception:
        pass

    if imdb_id:
        results = dtdd.search_by_imdb(imdb_id)
        if results:
            return dtdd.get_media(results[0]["id"])

    # Fall back to title search
    results = dtdd.search(title)
    if not results:
        return None

    # Try to match by year if we have it
    if year:
        for item in results:
            item_year = item.get("releaseYear", "")
            if str(year) == str(item_year):
                return dtdd.get_media(item["id"])

    # If no year match, take the first Movie result
    for item in results:
        item_type = item.get("itemType", {}).get("name", "")
        if item_type == "Movie":
            return dtdd.get_media(item["id"])

    # Last resort: first result
    return dtdd.get_media(results[0]["id"])


def process_movie(dtdd: DTDDClient, movie, plex_url: str, plex_token: str, dry_run: bool = False) -> bool:
    """Process a single movie. Returns True if the summary was updated."""
    title = f"{movie.title} ({movie.year})" if movie.year else movie.title

    original_summary = movie.summary or ""
    clean_summary = strip_warnings(original_summary)

    try:
        media_data = match_movie(dtdd, movie)
    except Exception as e:
        print(f"  ✗ {title} — API error: {e}")
        return False

    if not media_data:
        print(f"  – {title} — not found on DTDD")
        return False

    warning_text = format_warnings(media_data)
    if not warning_text:
        print(f"  – {title} — no significant warnings")
        return False

    new_summary = clean_summary + get_separator() + "\n" + warning_text
    updated = False

    if new_summary == (movie.summary or ""):
        print(f"  – {title} — already up to date")
        updated = False  # no change needed, but continue to check markers
    elif dry_run:
        print(f"  ✓ {title} — would add warnings:")
        for line in warning_text.split("\n"):
            print(f"      {line}")
        updated = True
    else:
        try:
            movie.editSummary(new_summary)
            print(f"  ✓ {title} — warnings added")
            updated = True
        except Exception as e:
            print(f"  ✗ {title} — failed to update summary: {e}")

    # Marker pipeline
    if _markers_enabled():
        try:
            metadata_id = str(movie.ratingKey)
            descs = run_marker_pipeline(
                plex_url=plex_url,
                plex_token=plex_token,
                metadata_id=metadata_id,
                media_data=media_data,
                media_title=title,
                dry_run=dry_run,
            )
            if descs:
                verb = "Would create" if dry_run else "Created"
                print(f"      + {verb} {len(descs)} marker(s):")
                for d in descs:
                    print(f"          • {d}")
        except Exception as e:
            logger.warning(f"  Marker pipeline error for {title}: {e}")

    return updated


# ---------------------------------------------------------------------------
# TV show processing
# ---------------------------------------------------------------------------

def match_show(dtdd: DTDDClient, show) -> dict | None:
    """Try to match a Plex TV show to a DTDD entry.

    Returns the DTDD search result item dict (not full media data), or None.
    """
    title = show.title

    # Try IMDB ID
    imdb_id = None
    try:
        for guid in show.guids:
            if guid.id.startswith("imdb://"):
                imdb_id = guid.id.replace("imdb://", "")
                break
    except Exception:
        pass

    if imdb_id:
        results = dtdd.search_by_imdb(imdb_id)
        if results:
            return results[0]

    # Fall back to title search, prefer TV Show type
    results = dtdd.search(title)
    if not results:
        return None

    year = getattr(show, "year", None)
    if year:
        for item in results:
            if (
                str(item.get("releaseYear", "")) == str(year)
                and item.get("itemType", {}).get("name") == "TV Show"
            ):
                return item

    for item in results:
        if item.get("itemType", {}).get("name") == "TV Show":
            return item

    return results[0]


def process_show(dtdd: DTDDClient, show, plex_url: str, plex_token: str, dry_run: bool = False) -> int:
    """Process a single TV show: update episode summaries and create markers.

    Returns the number of episodes updated.
    """
    show_title = show.title
    print(f"\n  Show: {show_title}")

    try:
        match = match_show(dtdd, show)
    except Exception as e:
        print(f"  ✗ {show_title} — DTDD search error: {e}")
        return 0

    if not match:
        print(f"  – {show_title} — not found on DTDD")
        return 0

    dtdd_id = match["id"]

    # Fetch ALL episode trigger data in one call.
    # Use a short TTL so new episodes on ongoing shows are discovered on the next run.
    try:
        all_data = dtdd.get_media(dtdd_id, index1=-1, index2=-1, cache_ttl_override=_get_discovery_cache_ttl())
    except Exception as e:
        print(f"  ✗ {show_title} — DTDD media fetch error: {e}")
        return 0

    stats = all_data.get("topicItemStats", [])
    if not stats:
        print(f"  – {show_title} — no topic data from DTDD")
        return 0

    # Build a map: (season, episode) → list of triggered stats with comments
    # We need to know which episodes have triggers so we can match Plex episodes.
    episode_triggers: dict[tuple[int, int], list[dict]] = {}
    show_wide_triggers: list[dict] = []  # index1/index2 is None → applies to whole show

    for stat in stats:
        yes_count = stat.get("yesSum", 0)
        no_count = stat.get("noSum", 0)
        total = yes_count + no_count
        if total == 0:
            continue

        topic_name = stat.get("topic", {}).get("name", "")
        include_topics = getattr(config, "INCLUDE_TOPICS", None)
        exclude_topics = getattr(config, "EXCLUDE_TOPICS", None)
        if include_topics is not None:
            if topic_name.lower() not in [t.lower() for t in include_topics]:
                continue
        elif exclude_topics is not None:
            if topic_name.lower() in [t.lower() for t in exclude_topics]:
                continue

        min_yes = getattr(config, "MIN_YES_VOTES", 3)
        min_ratio = getattr(config, "MIN_YES_RATIO", 0.6)
        if not (yes_count / total >= min_ratio and yes_count >= min_yes):
            continue

        # Collect per-episode comment groups from this stat
        for comment in stat.get("comments", []):
            ci1 = comment.get("index1")
            ci2 = comment.get("index2")
            if ci1 is not None and ci2 is not None:
                key = (int(ci1), int(ci2))
                episode_triggers.setdefault(key, [])
                # Attach a copy of the stat with only this comment for context
                episode_triggers[key].append(stat)
            else:
                show_wide_triggers.append(stat)

        # Also capture the stat-level index for episodes with no per-comment index
        si1 = stat.get("index1")
        si2 = stat.get("index2")
        if si1 is not None and si2 is not None:
            key = (int(si1), int(si2))
            episode_triggers.setdefault(key, [])
            if stat not in episode_triggers[key]:
                episode_triggers[key].append(stat)

    # Deduplicate episode keys
    all_episode_keys = set(episode_triggers.keys())

    if not all_episode_keys and not show_wide_triggers:
        print(f"  – {show_title} — no triggered episode data")
        return 0

    # Fetch all Plex episodes for this show
    try:
        plex_episodes = show.episodes()
    except Exception as e:
        print(f"  ✗ {show_title} — could not fetch Plex episodes: {e}")
        return 0

    # Build a map of (season, episode) → Plex episode object
    plex_ep_map: dict[tuple[int, int], object] = {}
    for ep in plex_episodes:
        try:
            key = (ep.seasonNumber, ep.index)
            plex_ep_map[key] = ep
        except Exception:
            pass

    updated_count = 0

    # Process each episode that has trigger data
    for ep_key in sorted(all_episode_keys):
        season, episode = ep_key
        plex_ep = plex_ep_map.get(ep_key)

        if not plex_ep:
            logger.debug(f"  No Plex episode for S{season:02d}E{episode:02d}, skipping")
            continue

        ep_title = f"{show_title} S{season:02d}E{episode:02d}"

        # all_data (-1,-1) already contains full stats for every episode;
        # filter to this episode in format_warnings — no extra API call needed.
        warning_text = format_warnings(all_data, episode_index1=season, episode_index2=episode)

        ep_updated = False
        if warning_text:
            original_summary = plex_ep.summary or ""
            clean_summary = strip_warnings(original_summary)
            new_summary = clean_summary + get_separator() + "\n" + warning_text

            if new_summary == original_summary:
                print(f"  – {ep_title} — already up to date")
            elif dry_run:
                print(f"  ✓ {ep_title} — would add warnings:")
                for line in warning_text.split("\n"):
                    print(f"      {line}")
                ep_updated = True
            else:
                try:
                    plex_ep.editSummary(new_summary)
                    print(f"  ✓ {ep_title} — warnings added")
                    ep_updated = True
                except Exception as e:
                    print(f"  ✗ {ep_title} — failed to update summary: {e}")

        # Marker pipeline for this episode
        if _markers_enabled():
            try:
                metadata_id = str(plex_ep.ratingKey)
                descs = run_marker_pipeline(
                    plex_url=plex_url,
                    plex_token=plex_token,
                    metadata_id=metadata_id,
                    media_data=all_data,
                    media_title=ep_title,
                    dry_run=dry_run,
                    episode_index1=season,
                    episode_index2=episode,
                )
                if descs:
                    verb = "Would create" if dry_run else "Created"
                    print(f"      + {verb} {len(descs)} marker(s):")
                    for d in descs:
                        print(f"          • {d}")
            except Exception as e:
                logger.warning(f"  Marker pipeline error for {ep_title}: {e}")

        if ep_updated:
            updated_count += 1

    return updated_count


# ---------------------------------------------------------------------------
# Library helpers
# ---------------------------------------------------------------------------

def clear_warnings(plex: PlexServer, library_names: list[str] | None):
    """Remove all DTDD content warnings from movie and TV show summaries."""
    libraries = get_libraries(plex, library_names, types=("movie", "show"))
    total_cleared = 0

    for lib in libraries:
        print(f"\nClearing warnings from: {lib.title}")
        for item in lib.all():
            original = item.summary or ""
            cleaned = strip_warnings(original)
            if cleaned != original:
                item.editSummary(cleaned)
                print(f"  ✓ {item.title} — warnings removed")
                total_cleared += 1

            # For TV shows, also clear episode summaries
            if lib.type == "show":
                try:
                    for ep in item.episodes():
                        orig = ep.summary or ""
                        clean = strip_warnings(orig)
                        if clean != orig:
                            ep.editSummary(clean)
                            total_cleared += 1
                except Exception:
                    pass

    print(f"\nDone. Cleared warnings from {total_cleared} item(s).")


def get_libraries(plex: PlexServer, library_names: list[str] | None, types: tuple[str, ...] = ("movie",)):
    """Get libraries of the requested types to process."""
    if library_names:
        libraries = []
        for name in library_names:
            try:
                lib = plex.library.section(name)
                if lib.type in types:
                    libraries.append(lib)
                else:
                    print(f"Warning: '{name}' is type '{lib.type}', skipping (want: {types}).")
            except Exception:
                print(f"Warning: Library '{name}' not found, skipping.")
        return libraries
    else:
        return [s for s in plex.library.sections() if s.type in types]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Add DoesTheDogDie.com content warnings to your Plex library."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without modifying Plex")
    parser.add_argument("--clear", action="store_true",
                        help="Remove all content warnings from Plex summaries")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear the local DTDD API response cache")
    parser.add_argument("--movie", type=str,
                        help="Process a single movie by title (exact match)")
    parser.add_argument("--show", type=str,
                        help="Process a single TV show by title (exact match)")
    parser.add_argument("--list-topics", action="store_true",
                        help="Show all available DTDD topic names")
    args = parser.parse_args()

    dry_run = args.dry_run or getattr(config, "DRY_RUN", False)

    # Handle cache clear
    if args.clear_cache:
        client = DTDDClient(config.DTDD_API_KEY)
        client.clear_cache()
        if not args.clear and not args.movie and not args.show and not args.list_topics:
            return

    # Handle list-topics
    if args.list_topics:
        dtdd = DTDDClient(
            api_key=config.DTDD_API_KEY,
            cache_ttl=getattr(config, "CACHE_TTL", 604800),
            api_delay=getattr(config, "API_DELAY", 1.0),
        )
        print("Fetching topic list from DTDD...\n")
        results = dtdd.search("Avengers Endgame")
        if results:
            media = dtdd.get_media(results[0]["id"])
            stats = media.get("topicItemStats", [])
            topics = sorted(set(
                stat.get("topic", {}).get("name", "")
                for stat in stats
                if stat.get("topic", {}).get("name")
            ))
            print("Available topic names (copy these into INCLUDE_TOPICS or EXCLUDE_TOPICS):\n")
            for topic in topics:
                print(f'    "{topic}",')
            print(f"\n{len(topics)} topics found.")
        else:
            print("Could not fetch topics. Check your DTDD_API_KEY.")
        return

    # Connect to Plex
    print(f"Connecting to Plex at {config.PLEX_URL}...")
    try:
        plex = PlexServer(config.PLEX_URL, config.PLEX_TOKEN)
        print(f"Connected to: {plex.friendlyName}")
    except Exception as e:
        print(f"ERROR: Could not connect to Plex: {e}")
        sys.exit(1)

    plex_url = config.PLEX_URL
    plex_token = config.PLEX_TOKEN
    library_names = getattr(config, "PLEX_LIBRARIES", None)

    # Handle clear mode
    if args.clear:
        clear_warnings(plex, library_names)
        return

    # Initialize DTDD client
    dtdd = DTDDClient(
        api_key=config.DTDD_API_KEY,
        cache_ttl=getattr(config, "CACHE_TTL", 604800),
        api_delay=getattr(config, "API_DELAY", 1.0),
    )

    if dry_run:
        print("DRY RUN — no changes will be made to Plex\n")

    markers_note = " (markers enabled)" if _markers_enabled() else ""
    print(f"Mode: description warnings{markers_note}")

    # Single movie mode
    if args.movie:
        libraries = get_libraries(plex, library_names, types=("movie",))
        found = False
        for lib in libraries:
            results = lib.search(title=args.movie)
            for movie in results:
                found = True
                process_movie(dtdd, movie, plex_url=plex_url, plex_token=plex_token, dry_run=dry_run)
        if not found:
            print(f"Movie '{args.movie}' not found in Plex.")
        return

    # Single show mode
    if args.show:
        libraries = get_libraries(plex, library_names, types=("show",))
        found = False
        for lib in libraries:
            results = lib.search(title=args.show)
            for show in results:
                found = True
                process_show(dtdd, show, plex_url=plex_url, plex_token=plex_token, dry_run=dry_run)
        if not found:
            print(f"Show '{args.show}' not found in Plex.")
        return

    # Process all configured libraries
    movie_libs = get_libraries(plex, library_names, types=("movie",))
    show_libs = get_libraries(plex, library_names, types=("show",))

    if not movie_libs and not show_libs:
        print("No movie or TV show libraries found to process.")
        sys.exit(1)

    total_processed = 0
    total_updated = 0
    start_time = time.time()

    # Movies
    for lib in movie_libs:
        movies = lib.all()
        print(f"\nProcessing movies: {lib.title} ({len(movies)} movies)")
        print("-" * 50)
        for movie in movies:
            total_processed += 1
            if process_movie(dtdd, movie, plex_url=plex_url, plex_token=plex_token, dry_run=dry_run):
                total_updated += 1

    # TV shows
    for lib in show_libs:
        shows = lib.all()
        print(f"\nProcessing TV shows: {lib.title} ({len(shows)} shows)")
        print("-" * 50)
        for show in shows:
            total_processed += 1
            n = process_show(dtdd, show, plex_url=plex_url, plex_token=plex_token, dry_run=dry_run)
            if n:
                total_updated += n

    elapsed = time.time() - start_time
    print(f"\n{'=' * 50}")
    print(f"Done in {elapsed:.1f}s")
    print(f"Processed: {total_processed} item(s)")
    print(f"Updated:   {total_updated} episode(s)/movie(s)")
    if dry_run:
        print("(DRY RUN — no actual changes made)")


if __name__ == "__main__":
    main()
