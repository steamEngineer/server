"""Queue: read state and edit / delete queue items."""
# ruff: noqa: TID252  -- relative imports are the canonical MA-provider pattern.

from __future__ import annotations

from typing import TYPE_CHECKING

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from music_assistant_models.enums import QueueOption, RepeatMode
from music_assistant_models.errors import InvalidDataError

from ..models import AddToQueueResult, QueueBrief
from ..tags import Tag
from ._common import (
    TIMEOUT_FAST,
    TIMEOUT_MUTATION,
    TIMEOUT_QUERY,
    confirm_or_raise,
    queue_item_display_name,
    resolve_added_queue_item,
    to_brief_queue,
)

if TYPE_CHECKING:
    from music_assistant.mass import MusicAssistant

# Matches MA's default queue page size (and the ``queue://`` resource cap).
MAX_QUEUE_ITEMS = 500


def build_queue_server(  # noqa: PLR0915 -- one sub-server registers all queue tools
    mass: MusicAssistant, *, require_confirmation: bool = True
) -> FastMCP:
    """Construct the ``queue/*`` sub-server."""
    sub: FastMCP = FastMCP(name="queue")

    def _queue_brief(queue_id: str, include_items: int) -> QueueBrief:
        queue = mass.player_queues.get(queue_id)
        if queue is None:
            raise ToolError(f"Queue {queue_id!r} not found after move.")
        limit = min(max(include_items, 0), MAX_QUEUE_ITEMS)
        items = mass.player_queues.items(queue.queue_id, limit=limit) if limit > 0 else []
        return to_brief_queue(queue, items=list(items))

    @sub.tool(
        tags={Tag.QUERY_QUEUE},
        annotations=ToolAnnotations(
            title="Get active queue",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
        timeout=TIMEOUT_FAST,
    )  # type: ignore[untyped-decorator, unused-ignore]
    async def get_active_queue(
        player_id: str = "",
        include_items: int = 25,
        queue_id: str = "",
    ) -> QueueBrief | None:
        """
        Return the active queue for a player, or ``None`` if the player is idle.

        Returns ``QueueBrief`` with ``queue_id``, ``current_index``,
        ``item_count``, shuffle / repeat flags, ``available`` and up to
        ``include_items`` lookahead ``items``. Note that
        ``QueueBrief.queue_id`` is the identifier the mutation tools
        (``set_shuffle``, ``set_repeat``, ``add_to_queue``, ``move_item``,
        ``move_item_to_end``, ``remove_item``, ``clear_queue``, ``transfer_queue``)
        expect — it is
        distinct from ``player_id``. For a queue fed by an external plugin
        source (Connect / AirPlay / Ynison), the current item's ``name`` is
        the real track title rather than the source wrapper name.

        :param player_id: Player identifier from ``PlayerBrief.player_id``. The
            ``queue_id`` used by the playback/queue tools is accepted as an
            alias here, since for a normal player queue the two are the same
            value.
        :param include_items: How many lookahead items to materialise. Clamped
            to the ``[0, 500]`` range — 500 matches MA's own queue page size
            and the ``queue://`` resource cap, preventing a hostile or
            sloppy client from forcing the server to load thousands of rows
            on every call.
        :param queue_id: Alias for ``player_id`` — provide either one.
        """
        # ponytail: alias covers agents mislabeling player_id as queue_id; when MA's
        # queue_id genuinely differs, this still needs a player_id-shaped value.
        target = player_id or queue_id
        if not target:
            raise ToolError(
                "Provide player_id or queue_id (PlayerBrief.player_id for the player to inspect)."
            )
        queue = mass.player_queues.get_active_queue(target)
        if queue is None:
            return None
        limit = min(max(include_items, 0), MAX_QUEUE_ITEMS)
        items = mass.player_queues.items(queue.queue_id, limit=limit) if limit > 0 else []
        return to_brief_queue(queue, items=list(items))

    @sub.tool(
        tags={Tag.EDIT_QUEUE},
        annotations=ToolAnnotations(
            title="Toggle queue shuffle",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
        timeout=TIMEOUT_MUTATION,
    )  # type: ignore[untyped-decorator, unused-ignore]
    async def set_shuffle(queue_id: str, enabled: bool) -> None:
        """
        Enable or disable shuffle on the given queue.

        Setting the current value again is a no-op. Returns nothing.

        :param queue_id: Queue identifier from ``QueueBrief.queue_id`` (distinct
            from ``PlayerBrief.player_id``).
        :param enabled: ``True`` to shuffle, ``False`` to play in queue order.
        """
        await mass.player_queues.set_shuffle(queue_id, enabled)

    @sub.tool(
        tags={Tag.EDIT_QUEUE},
        annotations=ToolAnnotations(
            title="Set queue repeat mode",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
        timeout=TIMEOUT_MUTATION,
    )  # type: ignore[untyped-decorator, unused-ignore]
    async def set_repeat(queue_id: str, repeat_mode: str = "off") -> None:
        """
        Set the repeat mode for the given queue.

        Setting the current value again is a no-op. Returns nothing.

        :param queue_id: Queue identifier from ``QueueBrief.queue_id`` (distinct
            from ``PlayerBrief.player_id``).
        :param repeat_mode: Repeat mode:

            - ``off`` (default): No repeating.
            - ``one``: Repeat the current track.
            - ``all``: Repeat the entire queue.
        """
        # RepeatMode._missing_ silently falls back to UNKNOWN for invalid values
        # instead of raising ValueError, so we must validate explicitly.
        mode = RepeatMode(repeat_mode.lower())
        if mode is RepeatMode.UNKNOWN:
            valid = ", ".join(f"``{e.value}``" for e in RepeatMode if e is not RepeatMode.UNKNOWN)
            raise ToolError(f"Invalid repeat_mode {repeat_mode!r}. Valid options: {valid}")

        mass.player_queues.set_repeat(queue_id, mode)

    @sub.tool(
        tags={Tag.DELETE_QUEUE},
        annotations=ToolAnnotations(
            title="Clear queue",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=True,
            openWorldHint=False,
        ),
        timeout=TIMEOUT_MUTATION,
    )  # type: ignore[untyped-decorator, unused-ignore]
    async def clear_queue(queue_id: str, ctx: Context | None = None) -> None:
        """
        Clear all items from the given queue. Cannot be undone.

        When ``Confirm destructive operations`` is enabled in the plugin
        settings the client is asked to confirm before the queue is cleared.
        Returns nothing.

        :param queue_id: Queue identifier from ``QueueBrief.queue_id``.
        """
        await confirm_or_raise(
            ctx,
            f"Clear all items from queue {queue_id!r}? This cannot be undone.",
            enabled=require_confirmation,
        )
        mass.player_queues.clear(queue_id)

    @sub.tool(
        tags={Tag.CONTROL_PLAYBACK},
        annotations=ToolAnnotations(
            title="Transfer queue between players",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=False,
            openWorldHint=False,
        ),
        timeout=TIMEOUT_MUTATION,
    )  # type: ignore[untyped-decorator, unused-ignore]
    async def transfer_queue(source_queue_id: str, target_queue_id: str) -> None:
        """
        Move the contents and playback state of one queue onto another player.

        The source player stops playing and its queue is emptied. Returns
        nothing.

        :param source_queue_id: Queue identifier of the player currently
            holding the queue (from ``QueueBrief.queue_id``).
        :param target_queue_id: Queue identifier of the player that should
            receive the queue.
        """
        await mass.player_queues.transfer_queue(source_queue_id, target_queue_id)

    @sub.tool(
        tags={Tag.EDIT_QUEUE},
        annotations=ToolAnnotations(
            title="Add media to queue",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=False,
        ),
        timeout=TIMEOUT_QUERY,
    )  # type: ignore[untyped-decorator, unused-ignore]
    async def add_to_queue(
        queue_id: str,
        uri: str,
        option: str = "add",
    ) -> AddToQueueResult:
        """
        Enqueue media on a queue with an explicit placement mode.

        Supports different enqueue modes to control where items are placed
        and whether playback is affected.

        Returns ``AddToQueueResult`` with the new row's ``item_id``, ``uri``,
        ``name``, and ``option`` so callers can confirm the add succeeded
        before enqueueing the next item.

        :param queue_id: Queue identifier from ``QueueBrief.queue_id`` (distinct
            from ``PlayerBrief.player_id``).
        :param uri: Music Assistant URI of the media to add, of the form
            ``<provider>://<media_type>/<id>`` (e.g. as found on
            ``TrackBrief.uri`` / ``AlbumBrief.uri`` / ``PlaylistBrief.uri``).
        :param option: Enqueue mode controlling placement and playback:

            - ``add`` (default): Append to the end of the queue without
              interrupting the current item. Preferred for "add to queue"
              requests — unlike ``playback_play_media``, this keeps what is
              already playing.
            - ``next``: Insert after the currently playing item (plays next).
            - ``play``: Insert after current item and start playing immediately.
            - ``replace_next``: Replace all items after the current one.
            - ``replace``: Clear the queue and replace with the new media.
        """
        # QueueOption._missing_ silently falls back to UNKNOWN for invalid values
        # instead of raising ValueError, so we must validate explicitly.
        queue_option = QueueOption(option)
        if queue_option is QueueOption.UNKNOWN:
            valid = ", ".join(f"``{e.value}``" for e in QueueOption if e is not QueueOption.UNKNOWN)
            raise ToolError(f"Invalid option {option!r}. Valid options: {valid}")

        before_items = mass.player_queues.items(queue_id, limit=MAX_QUEUE_ITEMS)
        before_item_ids = frozenset(str(getattr(it, "queue_item_id", "")) for it in before_items)
        await mass.player_queues.play_media(queue_id, uri, option=queue_option)
        after_items = mass.player_queues.items(queue_id, limit=MAX_QUEUE_ITEMS)
        added = resolve_added_queue_item(after_items, uri, before_item_ids=before_item_ids)
        if added is None:
            raise ToolError(
                f"Added {uri!r} to queue {queue_id!r} but could not locate the new queue row."
            )
        return AddToQueueResult(
            item_id=str(getattr(added, "queue_item_id", "")),
            uri=uri,
            name=queue_item_display_name(added),
            option=option,
        )

    @sub.tool(
        tags={Tag.DELETE_QUEUE},
        annotations=ToolAnnotations(
            title="Remove items from queue",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=False,
            openWorldHint=False,
        ),
        timeout=TIMEOUT_MUTATION,
    )  # type: ignore[untyped-decorator, unused-ignore]
    async def remove_item(
        queue_id: str,
        item_ids: list[str],
        ctx: Context | None = None,
    ) -> None:
        """
        Remove one or more items from a queue by ``item_id``.

        Call ``get_active_queue`` first to list items and their stable
        ``item_id`` values. Pass all ids in a single call rather than
        removing one at a time. The currently playing or buffered item
        cannot be removed — MA ignores that request.

        When ``Confirm destructive operations`` is enabled the client is
        asked to confirm before items are removed. Returns nothing.

        :param queue_id: Queue identifier from ``QueueBrief.queue_id``.
        :param item_ids: ``item_id`` values from ``QueueItemBrief`` returned
            by ``get_active_queue``. At least one id is required.
        """
        if not item_ids:
            raise ToolError(
                "Provide at least one item_id from QueueBrief.items[].item_id "
                "(use get_active_queue first)."
            )
        await confirm_or_raise(
            ctx,
            f"Remove {len(item_ids)} item(s) from queue {queue_id!r}?",
            enabled=require_confirmation,
        )
        for item_id in item_ids:
            try:
                mass.player_queues.delete_item(queue_id, item_id)
            except InvalidDataError as exc:
                raise ToolError(str(exc)) from exc

    @sub.tool(
        tags={Tag.EDIT_QUEUE},
        annotations=ToolAnnotations(
            title="Move queue item",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=False,
        ),
        timeout=TIMEOUT_MUTATION,
    )  # type: ignore[untyped-decorator, unused-ignore]
    async def move_item(
        queue_id: str, item_id: str, pos_shift: int = 1, include_items: int = 25
    ) -> QueueBrief:
        """
        Move an existing queue row up, down, or to play next.

        Call ``get_active_queue`` first for ``item_id`` values. The currently
        playing or buffered item cannot be moved. Returns the reordered
        ``QueueBrief`` so the new order can be confirmed without a separate
        ``get_active_queue`` call.

        :param queue_id: Queue identifier from ``QueueBrief.queue_id``.
        :param item_id: ``item_id`` from ``QueueItemBrief`` returned by
            ``get_active_queue``.
        :param pos_shift: Relative move — ``-1`` up one slot, ``+1`` down one
            slot (default), ``0`` to insert after the currently playing item
            (play next).
        :param include_items: How many lookahead items to materialise in the
            returned brief. Clamped to the ``[0, 500]`` range.
        """
        try:
            mass.player_queues.move_item(queue_id, item_id, pos_shift)
        except (IndexError, InvalidDataError) as exc:
            raise ToolError(str(exc)) from exc
        return _queue_brief(queue_id, include_items)

    @sub.tool(
        tags={Tag.EDIT_QUEUE},
        annotations=ToolAnnotations(
            title="Move queue item to end",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=False,
        ),
        timeout=TIMEOUT_MUTATION,
    )  # type: ignore[untyped-decorator, unused-ignore]
    async def move_item_to_end(queue_id: str, item_id: str, include_items: int = 25) -> QueueBrief:
        """
        Move an existing queue row to the back of the queue.

        Call ``get_active_queue`` first for ``item_id`` values. The currently
        playing or buffered item cannot be moved. Returns the reordered
        ``QueueBrief`` so the new order can be confirmed without a separate
        ``get_active_queue`` call.

        :param queue_id: Queue identifier from ``QueueBrief.queue_id``.
        :param item_id: ``item_id`` from ``QueueItemBrief`` returned by
            ``get_active_queue``.
        :param include_items: How many lookahead items to materialise in the
            returned brief. Clamped to the ``[0, 500]`` range.
        """
        try:
            mass.player_queues.move_item_end(queue_id, item_id)
        except (IndexError, InvalidDataError) as exc:
            raise ToolError(str(exc)) from exc
        return _queue_brief(queue_id, include_items)

    return sub
