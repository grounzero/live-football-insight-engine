"""Tactical archetypes for the generated fixtures.

Its own module because the simulation reads these and the public loader picks
between them, and putting them in either would make the other import a module
it does not otherwise need.

One simulation drives all three. Three implementations would drift apart the
first time anything downstream changed, and there would be no honest way to say
the fixtures differ in *tactics* rather than in code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class FixtureProfile:
    """How one archetype plays.

    Every field feeds a decision the simulation already has to make, rather
    than describing an outcome it is steered toward: how willing a player is to
    pass rather than carry, how hard the defending side squeezes, how far up
    the pitch it holds its line. The differences a viewer sees are consequences
    of those, not of a script.

    Attributes:
        key: Stable identifier, used in match ids and metadata.
        name: Short human name for the UI.
        narrative: One line describing what a viewer should expect to see.
        pass_appetite: Chance a player with a reasonable option passes rather
            than carrying.
        shot_appetite: Chance a player inside shooting range takes it on.
        directness: Weight on forward progress when choosing a pass, against
            how open the receiver is. High values play through the lines.
        width_scale: Multiplies the formation's lateral spread.
        line_offset_m: Shifts the defending block up (+) or back (-).
        press_squeeze: How hard the defending side collapses toward the ball.
        press_reaction_scale: Multiplies the defending side's reaction time.
            Below 1 is an aggressive press; above 1 sits off.
        player_speed_scale: Multiplies every player's top speed.
        accel_scale: Multiplies every player's acceleration.
    """

    key: str
    name: str
    narrative: str
    pass_appetite: float
    shot_appetite: float
    directness: float
    width_scale: float
    line_offset_m: float
    press_squeeze: float
    press_reaction_scale: float
    player_speed_scale: float
    accel_scale: float


#: The three public archetypes, in rotation order.
#:
#: Meant to be told apart by watching rather than by reading a label: a patient
#: side that keeps the ball and works it sideways, a pressing side that wins it
#: high and gives it away quickly, and a deep block that breaks fast.
PROFILES: Final[tuple[FixtureProfile, ...]] = (
    FixtureProfile(
        key="build_up",
        name="Patient build-up",
        narrative="Keeps the ball, works it wide, and takes few risks in midfield.",
        pass_appetite=0.86,
        shot_appetite=0.14,
        directness=0.25,
        width_scale=1.18,
        line_offset_m=-1.0,
        press_squeeze=0.30,
        press_reaction_scale=1.35,
        player_speed_scale=0.94,
        accel_scale=0.90,
    ),
    FixtureProfile(
        key="high_press",
        name="High press and turnovers",
        narrative="Compressed shape, aggressive pressing and possession that changes hands often.",
        pass_appetite=0.62,
        shot_appetite=0.18,
        directness=0.55,
        width_scale=0.82,
        line_offset_m=7.0,
        press_squeeze=0.42,
        press_reaction_scale=0.60,
        player_speed_scale=0.98,
        accel_scale=1.05,
    ),
    FixtureProfile(
        key="counter",
        name="Fast counterattacks",
        narrative="Sits deep, then breaks quickly and vertically once it wins the ball.",
        pass_appetite=0.70,
        shot_appetite=0.22,
        directness=0.95,
        width_scale=0.96,
        line_offset_m=-7.0,
        press_squeeze=0.42,
        press_reaction_scale=0.95,
        player_speed_scale=1.06,
        accel_scale=1.20,
    ),
)

DEFAULT_PROFILE: Final = PROFILES[0]

PROFILES_BY_KEY: Final[dict[str, FixtureProfile]] = {p.key: p for p in PROFILES}
