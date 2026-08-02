"""The message types carried on the insight stream.

Its own module, and deliberately importing nothing from the rest of the serving
package: the producer (:mod:`football_insights.serving.stream`) and the consumer
(:mod:`football_insights.serving.app`) reach it through
:mod:`football_insights.serving.state`, which the producer may not import at
module scope without a cycle.

A message is a *kind* plus the JSON that goes on the wire. The kind is carried
alongside the text rather than recovered from it, because the consumer has to
recognise the end marker, and the alternative it replaces was a substring search
for ``'"type": "end"'`` over the serialised JSON.

That search was correct and would have stayed correct for as long as nobody
touched how the payload is serialised — which is exactly what makes it a bad
way to express the intent. It depends on ``json.dumps`` defaults nothing else
depends on: passing ``separators=(",", ":")`` anywhere, to trim the frame
traffic, produces ``{"type":"end"}``, the search stops matching, and every
connected client waits on a stream that has already finished. A silent hang, in
the browser, from a change that looks like a size optimisation.

Carrying the type out of band removes the coupling entirely: the wire format can
be reserialised however anyone likes and the consumer never reads it.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import NamedTuple

from football_insights.types import JsonDict


class StreamMessageType(StrEnum):
    """What a stream message describes.

    The values are the wire format: they are serialised into the ``type`` field
    that the browser switches on, so renaming one is a breaking API change.
    """

    FRAME = "frame"
    INSIGHT = "insight"
    SUPPRESSION = "suppression"
    RESTART = "restart"
    MATCH = "match"
    #: Terminal. A client closes its stream for good on receiving this, so it is
    #: published only when the replay has genuinely finished — never when the
    #: loop is being torn down to swap match.
    END = "end"


class StreamMessage(NamedTuple):
    """One fan-out message: its kind, and the JSON to send."""

    type: StreamMessageType
    data: str


def stream_message(kind: StreamMessageType, payload: JsonDict) -> StreamMessage:
    """Serialise one message once, for fan-out to every subscriber.

    Args:
        kind: What the message describes.
        payload: Body, under the ``payload`` key.

    Returns:
        The message, ready to publish.
    """
    return StreamMessage(kind, json.dumps({"type": kind.value, "payload": payload}))
