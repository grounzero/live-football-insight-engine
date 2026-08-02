"""State shared across requests, and the lifetime of the replay task.

Its own module because the objects here outlive any single request: the
subscriber set, the task driving the replay, and the two flags that let a
request rearrange a running replay without lying to the clients watching it.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fastapi import Request

from football_insights.config import Settings
from football_insights.insight.types import Insight
from football_insights.serving.messages import StreamMessage
from football_insights.serving.metrics import Metrics

if TYPE_CHECKING:
    from football_insights.domain import Event, MatchTracking, Orientation
    from football_insights.replay.player import ReplayPlayer
    from football_insights.serving.engine import InsightEngine

_Subscriber = asyncio.Queue[StreamMessage]


@dataclass(frozen=True, slots=True)
class FixtureRotation:
    """One fixture in the public rotation, pre-built at startup.

    Tracking, events and orientation are generated once and held, rather than
    regenerated at each changeover. Generating a five-minute fixture takes about
    a quarter of a second, which is a long time to stall the event loop in the
    middle of a replay every viewer is watching.
    """

    match_id: str
    name: str
    narrative: str
    tracking: MatchTracking
    events: tuple[Event, ...]
    orientation: Orientation


@dataclass
class AppState:
    """Objects shared across requests."""

    settings: Settings
    metrics: Metrics
    engine: InsightEngine | None = None
    player: ReplayPlayer | None = None
    #: Where the running replay's frames came from, reported by ``/ready``.
    data_source: str = "unknown"
    #: Whether the built frontend was found and mounted at ``/``. Part of
    #: readiness in public-demo mode, where an image without the page is not a
    #: usable deployment however healthy the API is.
    ui_available: bool = False
    recent_insights: list[Insight] = field(default_factory=list[Insight])
    #: The public rotation, and where in it the replay currently is. Empty
    #: outside public-demo mode, where a single match loops as before and
    #: changing it is an explicit request rather than a schedule.
    fixtures: tuple[FixtureRotation, ...] = ()
    fixture_index: int = 0
    _subscribers: set[_Subscriber] = field(default_factory=set[_Subscriber])
    _task: asyncio.Task[None] | None = None
    _restart: bool = False
    #: What a client joining a replay already in progress is sent before
    #: anything else: the match announcement it missed, then the most recent
    #: picture of that match. Either may be absent — the public demo announces
    #: nothing at startup, because it never changes match.
    #:
    #: One field holding both rather than two fields, and always rebound whole.
    #: The pair has to be consistent, and there is no lock here: a subscriber
    #: arriving between two assignments would be seeded with an announcement
    #: from one fixture and a picture from another.
    _snapshot: tuple[StreamMessage | None, StreamMessage | None] = (None, None)
    #: Held for the whole of a match switch. Two switches at once, or a switch
    #: racing a restart, would interleave a cancel against a half-installed
    #: player; nothing else in this object serialises them.
    _switch: asyncio.Lock = field(default_factory=asyncio.Lock)
    _suppress_end: bool = False

    @property
    def switch(self) -> asyncio.Lock:
        """The match-switch lock, so handlers can test it before taking it."""
        return self._switch

    @property
    def end_marker_wanted(self) -> bool:
        """Whether a replay loop stopping now should tell clients it ended.

        False only while the loop is being cancelled on purpose. Clients close
        their stream for good on an end marker, so publishing one during a
        deliberate swap would present a match change as a finished replay and
        leave every tab disconnected.
        """
        return not self._suppress_end

    def request_restart(self) -> None:
        """Ask the replay loop to rebuild its state before the next frame.

        The request is not applied here. When it arrives the loop may already be
        holding a frame taken from the old position, and letting that frame
        reach the engine would leave its monotonic frame check ahead of
        everything about to be replayed — every frame of the new run would be
        rejected as out of order while the pitch kept animating. Only the loop
        knows it is holding that frame, so only the loop can drop it.
        """
        self._restart = True

    def take_restart(self) -> bool:
        """Consume a pending restart request, if there is one."""
        pending, self._restart = self._restart, False
        return pending

    @property
    def has_subscribers(self) -> bool:
        """Whether anyone is listening to the stream."""
        return bool(self._subscribers)

    @property
    def replay_running(self) -> bool:
        """Whether the replay loop is currently driving frames.

        Exposed so readiness can *report* it without depending on it, and so the
        lifecycle around an unwatched demo can be asserted without reaching into
        the private task handle.
        """
        return self._task is not None and not self._task.done()

    def subscribe(self) -> _Subscriber:
        """Register a new SSE subscriber, seeded with the current snapshot.

        A shared replay is always mid-match by the time anyone arrives. Without
        this the visitor watches an empty pitch until the next scheduled picture
        and, worse, has no idea which fixture or lap the frames belong to when
        they do start — the match announcement was published before they
        connected and is not repeated.

        Seeded here, synchronously, rather than by asking the replay loop to
        re-announce: the loop is between frames at an arbitrary point and would
        have to be interrupted, and the messages are already serialised.
        """
        queue: _Subscriber = asyncio.Queue(maxsize=256)
        for message in self._snapshot:
            if message is not None:
                queue.put_nowait(message)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: _Subscriber) -> None:
        """Remove an SSE subscriber."""
        self._subscribers.discard(queue)

    def ensure_replay_task(self) -> None:
        """Start the replay loop once, on the first subscriber.

        Owned by the state rather than by the route handler, so the task handle
        stays private to the object that manages its lifetime.
        """
        # Imported here rather than at module scope: the loop reads this object
        # on every frame, so importing it eagerly would be a cycle.
        from football_insights.serving.stream import run_replay

        if self._task is not None and not self._task.done():
            return
        # Cleared here rather than by whatever stopped the previous task: a
        # replay that ended because nobody was watching set this on its way out,
        # and the new one must be free to announce a genuine end.
        self._suppress_end = False
        self._task = asyncio.create_task(run_replay(self))

    def should_stop_unwatched(self) -> bool:
        """Whether a public replay should end because nobody is watching it.

        A hosted demo loops forever, so without this the first visitor of the day
        leaves a container scoring two hundred frames a second until it is
        redeployed.

        Consulted by the replay loop, on a frame it is already awake for, rather
        than triggered from the stream handler when a client goes away. Two
        earlier shapes were wrong:

        * pausing the *player* deadlocks the demo. A paused player never yields
          another frame, so the loop body that would notice a viewer returning
          never runs again — the demo goes dark permanently for everyone after
          the first person closes their tab, while still reporting itself ready.
        * awaiting a stop from the stream generator's ``finally`` is not
          reliable. That block runs while the generator is being closed, where
          awaiting is not guaranteed to complete.

        Ending the *task* leaves the player untouched, so ``ensure_replay_task``
        on the next subscriber picks the match up from the same position with no
        special case anywhere. The end marker is suppressed on the way out: this
        is a pause in service, not the end of a replay, and a client connecting
        in the gap must not be told the match finished.

        Outside public mode this is always false — a local run is often watched
        by a CLI or a `curl` that comes and goes, and tearing the replay down
        between them would be surprising rather than economical.

        Returns:
            Whether the loop should return now.
        """
        if not self.settings.service.public_demo or self.has_subscribers:
            return False
        self._suppress_end = True
        return True

    async def stop_replay_task(self) -> None:
        """Stop the replay loop and wait for it, without announcing an end.

        Swapping ``player`` is not enough to stop the loop: it binds the player
        into a local once and then sits inside its stream, so an assignment here
        would leave it replaying the old match indefinitely. The task has to be
        cancelled and awaited.

        A task that has already finished on its own is left alone — it published
        its end marker legitimately, and clients have acted on it.
        """
        task, self._task = self._task, None
        if task is None or task.done():
            return
        self._suppress_end = True
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        self._suppress_end = False

    def announce(self, message: StreamMessage) -> None:
        """Publish a match announcement and make it the head of the snapshot.

        The frame is dropped rather than kept alongside the new announcement.
        It describes the *previous* fixture, and a visitor seeded with a match
        message from one replay and a picture from another would draw the old
        positions under the new name — then interpolate from them into the first
        real frame, walking every player across the pitch.
        """
        self._snapshot = (message, None)
        self.publish(message, critical=True)

    def publish_barrier(self, message: StreamMessage) -> None:
        """Publish a critical barrier and discard the snapshot's stale picture.

        After a rewind or a lap wrap the retained frame belongs to a replay that
        no longer exists. Seeding a new subscriber with it would show them the
        end of the last lap as though it were the current position, and their
        interpolator would then blend that into the first frame of the new one.
        """
        self._snapshot = (self._snapshot[0], None)
        self.publish(message, critical=True)

    def publish_frame(self, message: StreamMessage) -> None:
        """Publish a visual frame and keep it as the snapshot's current picture.

        Replaces the picture without touching the announcement, so the pair
        always describes one fixture.
        """
        self._snapshot = (self._snapshot[0], message)
        self.publish(message)

    def publish(self, message: StreamMessage, *, critical: bool = False) -> None:
        """Fan out a message, dropping it for any subscriber that has fallen behind.

        A slow browser tab must never stall the replay loop, so a full queue
        loses the message rather than applying back pressure. Control messages
        pass ``critical=True``: losing an end marker would leave that client
        waiting forever, so one slot is made for it by discarding the oldest
        pending frame.
        """
        for queue in list(self._subscribers):
            if queue.full():
                if not critical:
                    continue
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            queue.put_nowait(message)


def get_state(request: Request) -> AppState:
    """FastAPI dependency returning the shared application state."""
    return request.app.state.fi  # type: ignore[no-any-return]
