"""Search the MCP tool catalog for meta-tool discovery."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from .constants import META_TOOL_NAMES

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_TOKEN_SPLIT = re.compile(r"[\s_\-./]+")

# Filler words that carry no tool-selection signal. Dropped from queries so a
# natural phrase ("pause the music") matches as well as terse keywords
# ("pause"). Kept deliberately conservative — articles, prepositions, pronouns
# and conjunctions only, never verbs that might name an action.
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "to",
        "for",
        "of",
        "on",
        "in",
        "into",
        "my",
        "me",
        "i",
        "it",
        "its",
        "is",
        "are",
        "be",
        "and",
        "or",
        "with",
        "please",
        "you",
        "your",
        "up",
        "some",
        "any",
        "all",
        "from",
        "at",
        "by",
        "as",
        "so",
        "then",
        "now",
    }
)

# Domain synonyms — agents phrase requests one way, the tools name things
# another ("song" → "track", "speaker" → "player").
_SYNONYMS = {
    "song": "track",
    "songs": "track",
    "tune": "track",
    "tunes": "track",
    "speaker": "player",
    "speakers": "players",
    "device": "player",
    "devices": "players",
}

# Generic terms dropped only when more specific tokens remain, so "music" does
# not force-match the literal "Music Assistant" in unrelated descriptions while
# a bare "music" query still searches.
_GENERIC_TERMS = frozenset({"music", "audio"})

_BROAD_QUERY_HINT = (
    "Query matched many tools. Narrow with namespace + action, e.g. "
    "'library albums', 'library tracks', 'playback play', 'players list'."
)

_EMPTY_QUERY_HINT = (
    "No matches. Use natural phrases with namespace + action, e.g. "
    "'library search albums', 'playback play media', 'players list', 'volume set'."
)

# Intent tuning — avoids play_pause tying playback_play_media on "playback play".
_EXPLICIT_PAUSE_TOKENS = frozenset({"pause"})
_EXPLICIT_RESUME_TOKENS = frozenset({"resume"})
_PLAY_CONTENT_TOKENS = frozenset(
    {"album", "albums", "track", "tracks", "media", "uri", "playlist", "radio", "artist", "artists"}
)
_PLAY_START_TOKENS = frozenset({"play", "start", "queue"})
_ENQUEUE_TOKENS = frozenset({"add", "append", "enqueue"})
_REMOVE_QUEUE_TOKENS = frozenset({"remove", "delete", "drop"})
_REMOVE_ITEM_TARGET_TOKENS = frozenset(
    {"track", "tracks", "song", "songs", "item", "items", "entry", "this", "that"}
)
_REORDER_QUEUE_TOKENS = frozenset({"move", "reorder", "rearrange", "bump", "shift"})
_MOVE_UP_TOKENS = frozenset({"up", "higher", "earlier"})
_MOVE_DOWN_TOKENS = frozenset({"down", "lower", "later"})
_MOVE_END_TOKENS = frozenset({"end", "back", "bottom", "last"})
_REPLACE_REST_TOKENS = frozenset({"next", "rest", "remaining", "upcoming", "after"})

_QUEUE_OPTIONS_CHEATSHEET = (
    "queue_add_to_queue options: add=append, next=after current, "
    "play=insert+start, replace_next=drop rest, replace=clear+load."
)

_ENQUEUE_SUMMARIES: dict[str, str] = {
    "add": "Append without interrupting playback — not playback_play_media.",
    "next": "Insert after the current item; keep the rest of the queue.",
    "play": "Insert after current and start playing immediately.",
    "replace_next": "Replace everything after the current item.",
    "replace": "Clear the queue and load new media (like playback_play_media).",
}


def _make_enqueue_workflow(option: str) -> dict[str, Any]:
    """Two-step playbook for queue_add_to_queue with a specific option."""
    return {
        "task": "enqueue_media_on_queue",
        "summary": f"{_ENQUEUE_SUMMARIES[option]} {_QUEUE_OPTIONS_CHEATSHEET}",
        "steps": [
            {
                "tool": "library_search_artists",
                "purpose": "Resolve a URI (use search_albums/search_tracks if needed).",
                "arguments": {"query": "<artist or album name>", "limit": 10},
            },
            {
                "tool": "queue_add_to_queue",
                "purpose": ("Artist URI adds full discography; album URI adds all tracks."),
                "arguments": {
                    "queue_id": "<player_id>",
                    "uri": "<uri from search>",
                    "option": option,
                },
            },
        ],
    }


_PLAY_MEDIA_WORKFLOW: dict[str, Any] = {
    "task": "play_media_on_player",
    "summary": "Play an album, track, or playlist on a speaker (multi-step).",
    "steps": [
        {
            "tool": "library_search_albums",
            "purpose": "Find the album URI (skip if you already have a URI).",
            "arguments": {"query": "<album or artist name>", "limit": 10},
        },
        {
            "tool": "library_get_album_tracks",
            "purpose": "Optional: list tracks on an album when you need track URIs or a tracklist.",
            "arguments": {"album_uri": "<uri from search>"},
        },
        {
            "tool": "library_search_artists",
            "purpose": "Alternative: find an artist URI when browsing by artist name.",
            "arguments": {"query": "<artist name>", "limit": 10},
        },
        {
            "tool": "library_get_artist_albums",
            "purpose": "Optional: list an artist's albums when you have their URI.",
            "arguments": {"artist_uri": "<uri from search>"},
        },
        {
            "tool": "players_list_players",
            "purpose": "List players; use player_id as queue_id (fuzzy-match names like 'office quads' → 'BRAVIA Theatre Quad').",
            "arguments": {},
        },
        {
            "tool": "playback_play_media",
            "purpose": "Start playback — queue_id is the player_id from the previous step.",
            "arguments": {"queue_id": "<player_id>", "uri": "<uri from search>"},
        },
    ],
}

# Backward-compatible alias for tests that referenced the add-only workflow.
_ADD_TO_QUEUE_WORKFLOW = _make_enqueue_workflow("add")

_REMOVE_FROM_QUEUE_WORKFLOW: dict[str, Any] = {
    "task": "remove_items_from_queue",
    "summary": (
        "Remove specific queue rows by item_id — not clear_queue, "
        "playlists_remove_tracks, or media_remove_from_library."
    ),
    "steps": [
        {
            "tool": "queue_get_active_queue",
            "purpose": "List queue items; note each item's item_id.",
            "arguments": {"player_id": "<player_id>", "include_items": 50},
        },
        {
            "tool": "queue_remove_item",
            "purpose": "Pass item_id values from the previous step.",
            "arguments": {
                "queue_id": "<queue_id from QueueBrief>",
                "item_ids": ["<item_id>"],
            },
        },
    ],
}


def _make_reorder_workflow(*, pos_shift: int | None = None, to_end: bool = False) -> dict[str, Any]:
    """Two-step playbook for reordering an existing queue row."""
    if to_end:
        move_tool = "queue_move_item_to_end"
        move_args: dict[str, Any] = {
            "queue_id": "<queue_id from QueueBrief>",
            "item_id": "<item_id>",
        }
        summary = "Move an existing queue row to the back — not add_to_queue or shuffle."
    else:
        move_tool = "queue_move_item"
        move_args = {
            "queue_id": "<queue_id from QueueBrief>",
            "item_id": "<item_id>",
            "pos_shift": pos_shift if pos_shift is not None else -1,
        }
        summary = (
            "Reorder by item_id — pos_shift -1=up, +1=down, 0=play next; "
            "use move_item_to_end for the back of the queue."
        )
    return {
        "task": "reorder_queue_item",
        "summary": summary,
        "steps": [
            {
                "tool": "queue_get_active_queue",
                "purpose": "List queue items; note each item's item_id.",
                "arguments": {"player_id": "<player_id>", "include_items": 50},
            },
            {
                "tool": move_tool,
                "purpose": "Move the chosen row.",
                "arguments": move_args,
            },
        ],
    }


def _has_queue_context(query_tokens: list[str]) -> bool:
    """Return whether the query is about loading media onto a queue, not bare playback."""
    return "queue" in query_tokens or any(t in _PLAY_CONTENT_TOKENS for t in query_tokens)


def _detect_enqueue_option(query_tokens: list[str]) -> str | None:
    """
    Map natural enqueue phrases to a ``queue_add_to_queue`` option.

    Returns ``None`` when the query is not queue-mutation intent (e.g. bare
    ``play album`` with no queue context → use the play workflow instead).
    """
    if not _has_queue_context(query_tokens):
        return None

    if "skip" in query_tokens and "next" in query_tokens and "queue" not in query_tokens:
        return None

    if "replace" in query_tokens or "clear" in query_tokens:
        if any(t in _REPLACE_REST_TOKENS for t in query_tokens):
            return "replace_next"
        return "replace"

    if any(t in _ENQUEUE_TOKENS for t in query_tokens):
        return "add"

    if any(t in _REORDER_QUEUE_TOKENS for t in query_tokens) and any(
        t in _PLAY_CONTENT_TOKENS for t in query_tokens
    ):
        return "add"

    if "next" in query_tokens and ("queue" in query_tokens or "after" in query_tokens):
        return "next"

    if "queue" in query_tokens and any(t in {"play", "start"} for t in query_tokens):
        return "play"

    return None


def _detect_remove_queue_intent(query_tokens: list[str]) -> bool:
    """
    Return whether the query asks to drop specific queue rows.

    Bare ``delete queue`` / ``remove queue`` (no item target) is excluded so
    ``clear_queue`` wins instead.
    """
    if "queue" not in query_tokens:
        return False
    if any(t in {"clear", "replace", *_ENQUEUE_TOKENS} for t in query_tokens):
        return False
    has_remove = any(t in _REMOVE_QUEUE_TOKENS for t in query_tokens)
    has_take_off = "take" in query_tokens and "off" in query_tokens
    if not (has_remove or has_take_off):
        return False
    if any(t in _REMOVE_ITEM_TARGET_TOKENS for t in query_tokens):
        return True
    return "from" in query_tokens


def _detect_reorder_queue_intent(query_tokens: list[str]) -> bool:
    """
    Return whether the query asks to reorder existing queue rows.

    Excludes shuffle-mode toggles and enqueue/remove phrasing.
    """
    if "queue" not in query_tokens:
        return False
    if _detect_remove_queue_intent(query_tokens):
        return False
    if any(t in _ENQUEUE_TOKENS for t in query_tokens):
        return False
    if "shuffle" in query_tokens and not any(t in _REORDER_QUEUE_TOKENS for t in query_tokens):
        return False
    has_reorder = any(t in _REORDER_QUEUE_TOKENS for t in query_tokens)
    has_direction = any(
        t in _MOVE_UP_TOKENS | _MOVE_DOWN_TOKENS | _MOVE_END_TOKENS for t in query_tokens
    )
    has_item = any(t in _REMOVE_ITEM_TARGET_TOKENS for t in query_tokens)
    if has_direction:
        return True
    if has_reorder and has_item:
        return True
    if "order" in query_tokens and has_item:
        return True
    return has_reorder and not any(t in _PLAY_CONTENT_TOKENS for t in query_tokens)


def _detect_move_shift(query_tokens: list[str]) -> int | None:
    """Map reorder phrasing to a ``queue_move_item`` pos_shift, if applicable."""
    if not _detect_reorder_queue_intent(query_tokens):
        return None
    if any(t in _MOVE_END_TOKENS for t in query_tokens):
        return None
    if any(t in _MOVE_UP_TOKENS for t in query_tokens):
        return -1
    if any(t in _MOVE_DOWN_TOKENS for t in query_tokens):
        return 1
    if "next" in query_tokens and any(
        t in _REORDER_QUEUE_TOKENS | _REMOVE_ITEM_TARGET_TOKENS for t in query_tokens
    ):
        return 0
    return None


def _detect_move_to_end(query_tokens: list[str]) -> bool:
    """Return whether reorder phrasing targets the back of the queue."""
    return _detect_reorder_queue_intent(query_tokens) and any(
        t in _MOVE_END_TOKENS for t in query_tokens
    )


def tokenize_query(text: str) -> list[str]:
    """Split a query into lowercase tokens (min length 2)."""
    return [t for t in _TOKEN_SPLIT.split(text.casefold()) if len(t) >= 2]


def normalize_query_tokens(tokens: list[str]) -> list[str]:
    """
    Clean raw query tokens for matching: drop filler words and apply synonyms.

    Stopwords are removed, known synonyms are rewritten to the vocabulary the
    tools actually use, and generic terms (``music``/``audio``) are dropped only
    when more specific tokens remain. Falls back to the un-stripped tokens when
    normalisation would otherwise empty the query (e.g. a bare ``music``).
    """
    cleaned = [_SYNONYMS.get(t, t) for t in tokens if t not in _STOPWORDS]
    specific = [t for t in cleaned if t not in _GENERIC_TERMS]
    return specific or cleaned


def score_tool_match(name: str, description: str, query_tokens: list[str]) -> int:
    """
    Score how well a tool matches *query_tokens*.

    Tool names use ``namespace_action`` segments; queries often use spaces
    (``library search albums``). Underscores are treated as word boundaries.
    Tokens that match the name score highest, then the description; unmatched
    tokens do not disqualify the tool, but a tool that only matches the
    description must cover at least half the query tokens to count.
    """
    if not query_tokens:
        return 0

    name_tokens = tokenize_query(name.replace("_", " "))
    desc_tokens = tokenize_query(description)
    name_spaced = name.replace("_", " ").casefold()
    desc_cf = description.casefold()

    score = 0
    matched = 0
    name_hits = 0
    exact_name_hits = 0
    for token in query_tokens:
        token_score = 0
        in_name = False
        if token in name_tokens:
            token_score = 30
            in_name = True
            exact_name_hits += 1
        elif any(
            seg.startswith(token) or token.startswith(seg) for seg in name_tokens if len(seg) >= 3
        ):
            token_score = 20
            in_name = True
        elif token in name_spaced:
            token_score = 15
            in_name = True
        elif token in desc_cf:
            token_score = 8
        elif any(
            word.startswith(token) or token.startswith(word)
            for word in desc_tokens
            if len(word) >= 3
        ):
            token_score = 5

        if token_score:
            matched += 1
            name_hits += int(in_name)
            score += token_score

    if score == 0:
        return 0

    # Soft match: unmatched filler/synonym tokens no longer disqualify a tool,
    # but to avoid a single stray description hit dragging in noise we require
    # either a name match or at least half the query tokens to land somewhere.
    if name_hits == 0 and matched * 2 < len(query_tokens):
        return 0

    # Prefer tools whose name segments cover more query tokens exactly.
    score += exact_name_hits * 5

    # Shorter names with same coverage are usually more specific (e.g. albums vs tracks).
    if exact_name_hits == len(query_tokens):
        score += max(0, 40 - len(name_tokens) * 3)

    return score


def apply_intent_adjustments(name: str, query_tokens: list[str], base_score: int) -> int:
    """Nudge rankings for common agent intents (play vs pause, list vs get)."""
    if base_score == 0:
        return 0

    score = base_score
    has_pause_intent = any(t in _EXPLICIT_PAUSE_TOKENS for t in query_tokens)
    has_resume_intent = any(t in _EXPLICIT_RESUME_TOKENS for t in query_tokens)
    has_play_content = any(t in _PLAY_CONTENT_TOKENS for t in query_tokens)
    has_play_start = any(t in _PLAY_START_TOKENS for t in query_tokens)

    if name == "playback_play_media":
        if has_play_content:
            score += 25
        if has_play_start and not has_pause_intent:
            score += 15
        if any(t in _ENQUEUE_TOKENS for t in query_tokens) and "queue" in query_tokens:
            score -= 30
        enqueue_option = _detect_enqueue_option(query_tokens)
        if enqueue_option in {"add", "next", "play", "replace_next"}:
            score -= 25
        elif enqueue_option == "replace":
            score -= 10
    elif name == "playback_pause":
        if has_pause_intent:
            score += 25
    elif name == "playback_resume":
        if has_resume_intent:
            score += 25
    elif name == "playback_play_pause":
        if has_pause_intent or has_resume_intent:
            score -= 35
        elif has_play_start and has_play_content:
            score -= 40
        elif has_play_start and not has_pause_intent:
            score -= 15
    else:
        score = _adjust_secondary_intent(name, query_tokens, score)

    return score


def detect_workflow(query_tokens: list[str]) -> dict[str, Any] | None:
    """Return a multi-step playbook for enqueue, remove, reorder, or play requests."""
    has_play_content = any(t in _PLAY_CONTENT_TOKENS for t in query_tokens)
    has_pause_intent = any(t in _EXPLICIT_PAUSE_TOKENS for t in query_tokens)

    if _detect_remove_queue_intent(query_tokens) and not has_pause_intent:
        return _REMOVE_FROM_QUEUE_WORKFLOW

    if _detect_move_to_end(query_tokens) and not has_pause_intent:
        return _make_reorder_workflow(to_end=True)

    move_shift = _detect_move_shift(query_tokens)
    if move_shift is not None and not has_pause_intent:
        return _make_reorder_workflow(pos_shift=move_shift)

    if _detect_reorder_queue_intent(query_tokens) and not has_pause_intent:
        return _make_reorder_workflow()

    enqueue_option = _detect_enqueue_option(query_tokens)
    if enqueue_option is not None and not has_pause_intent:
        return _make_enqueue_workflow(enqueue_option)

    if not any(t in {"play", "start", "playback"} for t in query_tokens):
        return None
    if has_pause_intent and not has_play_content:
        return None
    return _PLAY_MEDIA_WORKFLOW


def _pick_recommended(matches: list[dict[str, Any]]) -> str | None:
    if not matches:
        return None
    top = matches[0]
    if len(matches) == 1:
        return str(top["name"])
    top_score = int(top.get("score", 0))
    second_score = int(matches[1].get("score", 0))
    if top_score >= second_score * 1.5 or top_score >= second_score + 20:
        return str(top["name"])
    return None


async def search_tool_catalog(
    query: str,
    *,
    list_tools: Callable[..., Awaitable[Any]],
    is_tool_visible: Callable[[str], Awaitable[bool]],
    limit: int = 25,
) -> dict[str, Any]:
    """
    Return tools matching *query*, ranked by relevance.

    Matching tokenizes the query and tool name (``library_search_albums`` matches
    ``library search albums``). Results are lightweight (name, description, score);
    use ``get_tool_schema`` to fetch a single tool's full schema. Includes a
    ``recommended`` tool when one match is clearly best.
    """
    query_tokens = normalize_query_tokens(tokenize_query(query.strip()))
    if not query_tokens:
        return {"query": query, "count": 0, "tools": [], "hint": _EMPTY_QUERY_HINT}

    all_tools = await list_tools(run_middleware=False)
    ranked: list[tuple[int, str, str]] = []

    for listed in all_tools:
        name = str(getattr(listed, "name", "") or "")
        if not name or name in META_TOOL_NAMES:
            continue
        if not await is_tool_visible(name):
            continue

        description = str(getattr(listed, "description", "") or "")
        score = apply_intent_adjustments(
            name, query_tokens, score_tool_match(name, description, query_tokens)
        )
        if score > 0:
            ranked.append((score, name, description))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    cap = max(1, min(limit, 100))

    matches: list[dict[str, Any]] = []
    for score, name, description in ranked[:cap]:
        entry: dict[str, Any] = {
            "name": name,
            "description": description,
            "score": score,
        }
        matches.append(entry)

    result: dict[str, Any] = {"query": query, "count": len(matches), "tools": matches}
    recommended = _pick_recommended(matches)
    if recommended:
        result["recommended"] = recommended
    workflow = detect_workflow(query_tokens)
    if workflow is not None:
        result["workflow"] = workflow
    if not matches:
        result["hint"] = _EMPTY_QUERY_HINT
    elif len(matches) >= 8 and len(query_tokens) == 1:
        result["hint"] = _BROAD_QUERY_HINT
    return result


def _adjust_queue_intent(name: str, query_tokens: list[str], score: int) -> int | None:
    """Nudge rankings for queue mutation and skip-next queries."""
    enqueue_option = _detect_enqueue_option(query_tokens)
    if name == "queue_add_to_queue" and enqueue_option is not None:
        return score + 25
    if name == "queue_remove_item" and _detect_remove_queue_intent(query_tokens):
        return score + 25
    if name == "queue_move_item_to_end" and _detect_move_to_end(query_tokens):
        return score + 25
    if name == "queue_move_item" and _detect_reorder_queue_intent(query_tokens):
        return score + 25
    if name == "queue_set_shuffle" and "shuffle" in query_tokens and "queue" in query_tokens:
        return score + 15
    if name == "queue_add_to_queue" and _detect_reorder_queue_intent(query_tokens):
        return score - 20
    if (
        name == "queue_clear_queue"
        and "queue" in query_tokens
        and any(t in _REMOVE_QUEUE_TOKENS for t in query_tokens)
    ):
        return score + 15
    if name == "playlists_remove_tracks" and "queue" in query_tokens:
        return score - 20
    if name == "media_remove_from_library" and "queue" in query_tokens:
        return score - 20
    if name == "playback_next_track" and "skip" in query_tokens and "next" in query_tokens:
        return score + 15
    return None


def _adjust_secondary_intent(name: str, query_tokens: list[str], score: int) -> int:
    """Nudge rankings for library drill-down, queue enqueue, and player tools."""
    if name == "players_list_players" and any(
        t in {"list", "players", "player"} for t in query_tokens
    ):
        return score + 10
    if name == "library_get_album_tracks" and any(
        t in {"tracklist", "tracklists", "listing", "listings"} for t in query_tokens
    ):
        return score + 25
    if (
        name == "library_get_album_tracks"
        and "album" in query_tokens
        and any(t in {"track", "tracks", "song", "songs"} for t in query_tokens)
    ):
        return score + 20
    if name == "library_get_album_by_uri" and any(
        t in {"track", "tracks", "tracklist", "song", "songs", "listing"} for t in query_tokens
    ):
        return score - 15
    if name == "library_get_artist_albums" and any(
        t in {"discography", "discographies"} for t in query_tokens
    ):
        return score + 25
    if (
        name == "library_get_artist_albums"
        and "artist" in query_tokens
        and any(t in {"album", "albums"} for t in query_tokens)
    ):
        return score + 20
    if name == "library_get_artist_by_uri" and any(
        t in {"album", "albums", "discography"} for t in query_tokens
    ):
        return score - 15
    if (
        name == "library_search_albums"
        and any(t in {"discography", "discographies"} for t in query_tokens)
        and "artist" in query_tokens
    ):
        return score - 10
    if name == "playback_play_index" and "index" not in query_tokens:
        return score - 25
    if name == "players_get_player" and "list" in query_tokens:
        return score - 20
    if name == "players_ungroup_player" and "ungroup" in query_tokens:
        return score + 15
    if name == "players_group_player" and "ungroup" in query_tokens:
        return score - 25
    queue_score = _adjust_queue_intent(name, query_tokens, score)
    if queue_score is not None:
        return queue_score
    return score
