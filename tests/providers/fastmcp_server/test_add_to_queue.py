"""
Tests for the add_to_queue MCP tool.

Validates that:
1. Valid option values (add/next/play/replace/replace_next) are accepted and forwarded to MA.
2. Invalid option values raise a clean ToolError.
3. A successful add returns AddToQueueResult with item_id, uri, name, and option.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError
from music_assistant_models.enums import MediaType, QueueOption


def _queue_item(*, item_id: str, uri: str, name: str) -> SimpleNamespace:
    return SimpleNamespace(queue_item_id=item_id, uri=uri, name=name, media_item=None)


def _mock_items_before_after(
    mock_mass: MagicMock, *, before: list[SimpleNamespace], after: list[SimpleNamespace]
) -> None:
    mock_mass.player_queues.items = MagicMock(side_effect=[before, after])


async def test_add_to_queue_accepts_valid_options(
    mounted_queue: FastMCP, mock_mass: MagicMock
) -> None:
    """Each valid option value is accepted and forwarded to MA."""
    uri = "spotify://track/1"
    for opt in ("add", "next", "play", "replace", "replace_next"):
        mock_mass.player_queues.play_media.reset_mock()
        _mock_items_before_after(
            mock_mass,
            before=[],
            after=[_queue_item(item_id=f"item-{opt}", uri=uri, name="Track One")],
        )
        async with Client(mounted_queue) as client:
            result = await client.call_tool(
                "queue_add_to_queue", {"queue_id": "q1", "uri": uri, "option": opt}
            )
        mock_mass.player_queues.play_media.assert_awaited_once_with(
            "q1", uri, option=QueueOption(opt)
        )
        assert result.data.item_id == f"item-{opt}"
        assert result.data.uri == uri
        assert result.data.option == opt


async def test_add_to_queue_rejects_invalid_option(mounted_queue: FastMCP) -> None:
    """Invalid option raises ToolError with the list of valid options."""
    async with Client(mounted_queue) as client:
        with pytest.raises(ToolError, match="bogus") as exc_info:
            await client.call_tool(
                "queue_add_to_queue",
                {"queue_id": "q1", "uri": "spotify://track/1", "option": "bogus"},
            )
    msg = str(exc_info.value)
    assert "``add``" in msg
    assert "``replace_next``" in msg


async def test_add_to_queue_defaults_to_add(mounted_queue: FastMCP, mock_mass: MagicMock) -> None:
    """Calling add_to_queue without option defaults to 'add'."""
    uri = "spotify://track/1"
    mock_mass.player_queues.play_media.reset_mock()
    _mock_items_before_after(
        mock_mass,
        before=[],
        after=[_queue_item(item_id="item-1", uri=uri, name="Track One")],
    )
    async with Client(mounted_queue) as client:
        result = await client.call_tool("queue_add_to_queue", {"queue_id": "q1", "uri": uri})
    mock_mass.player_queues.play_media.assert_awaited_once_with("q1", uri, option=QueueOption.ADD)
    assert result.data.option == "add"
    assert result.data.name == "Track One"


async def test_add_to_queue_returns_ack_for_new_row(
    mounted_queue: FastMCP, mock_mass: MagicMock
) -> None:
    """Returns the newly added row when the same uri already exists elsewhere."""
    uri = "library://track/169"
    _mock_items_before_after(
        mock_mass,
        before=[_queue_item(item_id="old-dup", uri=uri, name="If I Had $1000000")],
        after=[
            _queue_item(item_id="old-dup", uri=uri, name="If I Had $1000000"),
            _queue_item(item_id="new-row", uri=uri, name="If I Had $1000000"),
        ],
    )
    async with Client(mounted_queue) as client:
        result = await client.call_tool(
            "queue_add_to_queue", {"queue_id": "q1", "uri": uri, "option": "add"}
        )
    assert result.data.item_id == "new-row"
    assert result.data.uri == uri
    assert result.data.name == "If I Had $1000000"
    assert result.data.option == "add"


async def test_add_to_queue_expanded_album_uri(
    mounted_queue: FastMCP, mock_mass: MagicMock
) -> None:
    """Album URIs expand to track rows; ack uses the first newly added row."""
    album_uri = "library://album/203"
    _mock_items_before_after(
        mock_mass,
        before=[_queue_item(item_id="playing", uri="library://track/93", name="Adrift")],
        after=[
            _queue_item(item_id="playing", uri="library://track/93", name="Adrift"),
            _queue_item(item_id="stunt-1", uri="library://track/206", name="One Week"),
            _queue_item(item_id="stunt-2", uri="library://track/207", name="It's All Been Done"),
        ],
    )
    async with Client(mounted_queue) as client:
        result = await client.call_tool(
            "queue_add_to_queue", {"queue_id": "q1", "uri": album_uri, "option": "add"}
        )
    assert result.data.item_id == "stunt-1"
    assert result.data.uri == album_uri
    assert result.data.name == "One Week"
    assert result.data.option == "add"


async def test_add_to_queue_raises_when_row_not_found(
    mounted_queue: FastMCP, mock_mass: MagicMock
) -> None:
    """Surfaces ToolError when play_media succeeds but the new row cannot be located."""
    mock_mass.player_queues.items = MagicMock(side_effect=[[], []])
    async with Client(mounted_queue) as client:
        with pytest.raises(ToolError, match="could not locate"):
            await client.call_tool(
                "queue_add_to_queue",
                {"queue_id": "q1", "uri": "spotify://track/1", "option": "add"},
            )


def _mock_queue(
    *, current_index: int | None = 0, index_in_buffer: int | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        queue_id="q1",
        current_index=current_index,
        index_in_buffer=index_in_buffer,
    )


def _mock_track(*, uri: str = "spotify://track/1", name: str = "Track One") -> SimpleNamespace:
    return SimpleNamespace(
        uri=uri,
        name=name,
        available=True,
        media_type=MediaType.TRACK,
        provider="spotify",
        item_id="1",
        image=None,
        duration=180,
    )


def _setup_index_add_mocks(
    mock_mass: MagicMock,
    *,
    uri: str,
    before: list[SimpleNamespace],
    after: list[SimpleNamespace],
    current_index: int = 1,
    index_in_buffer: int | None = None,
) -> None:
    mock_mass.player_queues.get.return_value = _mock_queue(
        current_index=current_index,
        index_in_buffer=index_in_buffer,
    )
    mock_mass.player_queues.items = MagicMock(side_effect=[before, before, after])
    mock_mass.music.get_item_by_uri = AsyncMock(return_value=_mock_track(uri=uri))
    mock_mass.player_queues._resolve_media_items = AsyncMock(return_value=[_mock_track(uri=uri)])
    mock_mass.player_queues.load = AsyncMock()


async def test_add_to_queue_index_calls_load(mounted_queue: FastMCP, mock_mass: MagicMock) -> None:
    """Index path inserts via load() without calling play_media."""
    uri = "spotify://track/new"
    before = [_queue_item(item_id=f"i{i}", uri=f"u{i}", name=f"n{i}") for i in range(5)]
    after = [
        *before[:3],
        _queue_item(item_id="new-row", uri=uri, name="Track One"),
        *before[3:],
    ]
    _setup_index_add_mocks(mock_mass, uri=uri, before=before, after=after, current_index=1)
    async with Client(mounted_queue) as client:
        result = await client.call_tool(
            "queue_add_to_queue", {"queue_id": "q1", "uri": uri, "index": 3}
        )
    mock_mass.player_queues.play_media.assert_not_called()
    mock_mass.player_queues.load.assert_awaited_once()
    _, kwargs = mock_mass.player_queues.load.call_args
    assert kwargs["insert_at_index"] == 3
    assert kwargs["shuffle"] is False
    assert result.data.index == 3
    assert result.data.item_id == "new-row"


async def test_add_to_queue_index_rejects_past_current(
    mounted_queue: FastMCP, mock_mass: MagicMock
) -> None:
    """Index before the next insertable position raises ToolError."""
    mock_mass.player_queues.get.return_value = _mock_queue(current_index=2)
    mock_mass.player_queues.items = MagicMock(
        return_value=[_queue_item(item_id="i0", uri="u", name="n")] * 5
    )
    async with Client(mounted_queue) as client:
        with pytest.raises(ToolError, match="insertable position"):
            await client.call_tool(
                "queue_add_to_queue",
                {"queue_id": "q1", "uri": "spotify://track/1", "index": 1},
            )


async def test_add_to_queue_index_rejects_buffered(
    mounted_queue: FastMCP, mock_mass: MagicMock
) -> None:
    """Index at or before the buffered row raises ToolError."""
    mock_mass.player_queues.get.return_value = _mock_queue(current_index=1, index_in_buffer=2)
    mock_mass.player_queues.items = MagicMock(
        return_value=[_queue_item(item_id="i0", uri="u", name="n")] * 5
    )
    async with Client(mounted_queue) as client:
        with pytest.raises(ToolError, match="index_in_buffer=2"):
            await client.call_tool(
                "queue_add_to_queue",
                {"queue_id": "q1", "uri": "spotify://track/1", "index": 2},
            )


async def test_add_to_queue_index_rejects_replace_combo(
    mounted_queue: FastMCP, mock_mass: MagicMock
) -> None:
    """replace/replace_next cannot be combined with index."""
    mock_mass.player_queues.get.return_value = _mock_queue(current_index=0)
    mock_mass.player_queues.items = MagicMock(
        return_value=[_queue_item(item_id="i0", uri="u", name="n")] * 3
    )
    async with Client(mounted_queue) as client:
        with pytest.raises(ToolError, match="cannot be combined"):
            await client.call_tool(
                "queue_add_to_queue",
                {
                    "queue_id": "q1",
                    "uri": "spotify://track/1",
                    "index": 2,
                    "option": "replace",
                },
            )


async def test_add_to_queue_index_returns_ack_with_index(
    mounted_queue: FastMCP, mock_mass: MagicMock
) -> None:
    """Successful index insert returns AddToQueueResult with index set."""
    uri = "spotify://track/new"
    before = [_queue_item(item_id=f"i{i}", uri=f"u{i}", name=f"n{i}") for i in range(4)]
    after = [
        *before[:2],
        _queue_item(item_id="inserted", uri=uri, name="Track One"),
        *before[2:],
    ]
    _setup_index_add_mocks(mock_mass, uri=uri, before=before, after=after, current_index=0)
    async with Client(mounted_queue) as client:
        result = await client.call_tool(
            "queue_add_to_queue", {"queue_id": "q1", "uri": uri, "index": 2}
        )
    assert result.data.index == 2
    assert result.data.item_id == "inserted"
    assert result.data.uri == uri


async def test_add_to_queue_index_expands_album(
    mounted_queue: FastMCP, mock_mass: MagicMock
) -> None:
    """Album URIs expand to multiple rows in a single load() call."""
    album_uri = "library://album/203"
    track_a = _mock_track(uri="library://track/206", name="One Week")
    track_b = _mock_track(uri="library://track/207", name="It's All Been Done")
    before = [_queue_item(item_id="playing", uri="library://track/93", name="Adrift")]
    after = [
        *before,
        _queue_item(item_id="stunt-1", uri="library://track/206", name="One Week"),
        _queue_item(item_id="stunt-2", uri="library://track/207", name="It's All Been Done"),
    ]
    mock_mass.player_queues.get.return_value = _mock_queue(current_index=0)
    mock_mass.player_queues.items = MagicMock(side_effect=[before, before, after])
    mock_mass.music.get_item_by_uri = AsyncMock(
        return_value=_mock_track(uri=album_uri, name="Stunt")
    )
    mock_mass.player_queues._resolve_media_items = AsyncMock(return_value=[track_a, track_b])
    mock_mass.player_queues.load = AsyncMock()
    async with Client(mounted_queue) as client:
        result = await client.call_tool(
            "queue_add_to_queue",
            {"queue_id": "q1", "uri": album_uri, "index": 1},
        )
    queue_items = mock_mass.player_queues.load.call_args[0][1]
    assert len(queue_items) == 2
    assert result.data.item_id == "stunt-1"
    assert result.data.index == 1
