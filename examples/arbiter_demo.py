from dataclasses import dataclass

from swarm_field.contracts import Move, Treat
from swarm_field.swarm import (
    ArbiterConfig,
    Objective,
    ObjectiveArbiter,
    ObjectiveProposal,
)

PEST = (8, 1)
FRONTIER = (40, 30)
DOCK = (0, 0)
TICK = 60_000
"""One minute between decision points, in milliseconds."""


RIVAL_ADVANTAGE = 10.0
"""How much better than TREAT the competing EXPLORE target scores, in cells."""


@dataclass(frozen=True)
class Tick:
    """One decision point in a scripted run.

    Attributes:
        remaining: Route cost from the robot to the pest, in cells.
        rival_on_offer: Whether the competing EXPLORE target is visible this tick. When it is,
            it always scores ``RIVAL_ADVANTAGE`` better than TREAT, so the only thing that
            changes across runs is how far the robot has already driven.
        battery_critical: Whether RECHARGE bids as an override this tick.
    """

    remaining: float
    rival_on_offer: bool = False
    battery_critical: bool = False


def _proposals(tick: Tick) -> list[ObjectiveProposal]:
    """Build the bids a killer's behaviours would hand in at one decision point."""
    treat = ObjectiveProposal.create(
        objective=Objective.TREAT,
        action=Treat(cell=PEST),
        potential=tick.remaining - 8.28,
        target=PEST,
        distance=tick.remaining,
    )
    proposals = [treat]

    if tick.rival_on_offer:
        proposals.append(
            ObjectiveProposal.create(
                objective=Objective.EXPLORE,
                action=Move(to=FRONTIER),
                potential=treat.potential - RIVAL_ADVANTAGE,
                target=FRONTIER,
                distance=30.0,
            )
        )

    if tick.battery_critical:
        proposals.append(
            ObjectiveProposal.create(
                objective=Objective.RECHARGE,
                action=Move(to=DOCK),
                potential=55.0,
                target=DOCK,
                distance=55.0,
                urgency=200.0,
                is_override=True,
            )
        )

    return proposals


def _run(title: str, script: list[Tick]) -> None:
    """Play one scripted run and print what the arbiter decided at each tick."""
    arbiter = ObjectiveArbiter(ArbiterConfig(min_dwell=TICK))
    print(f"\n{title}")
    print("-" * len(title))
    print(f"{'min':>4} {'left':>6} {'prog':>6} {'margin':>7} {'treat':>8} {'rival':>8}  decision")

    for minute, tick in enumerate(script):
        proposals = _proposals(tick)
        treat = proposals[0]
        # Read progress before deciding: afterwards it would describe the new commitment,
        # not the one the margin was applied to.
        progress = arbiter.progress(treat)
        margin = 5.0 + 15.0 * progress

        choice = arbiter.choose(proposals, now=minute * TICK)
        if choice is None:
            print(f"{minute:>4}  idle")
            continue

        rival = ""
        if tick.rival_on_offer:
            rival = f"{treat.score + RIVAL_ADVANTAGE:+.1f}"

        note = ""
        if choice.switched:
            note = "  <- switched"
        print(
            f"{minute:>4} {tick.remaining:>6.1f} {progress:>6.2f} {margin:>7.1f} "
            f"{treat.score:>+8.1f} {rival:>8}  {choice.proposal.objective.name}{note}"
        )


def main() -> None:
    """Three runs over the same pest, differing only in when the rival appears."""
    print("A killer commits to a pest 40 cells away and drives toward it.")
    print(f"An EXPLORE target scoring {RIVAL_ADVANTAGE:.0f} better appears at different moments.")
    print("A switch needs margin = floor(5) + commitment_weight(15) x progress.")

    _run(
        "1. The rival appears early: almost nothing is invested, so the robot redirects",
        [
            Tick(remaining=40.0),
            Tick(remaining=36.0),
            Tick(remaining=32.0, rival_on_offer=True),
            Tick(remaining=32.0, rival_on_offer=True),
        ],
    )

    _run(
        "2. The same rival appears late: the drive is nearly done, so the robot holds",
        [
            Tick(remaining=40.0),
            Tick(remaining=30.0),
            Tick(remaining=16.0),
            Tick(remaining=4.0, rival_on_offer=True),
            Tick(remaining=2.0, rival_on_offer=True),
        ],
    )

    _run(
        "3. A flat battery overrides everything, however close the pest is",
        [
            Tick(remaining=40.0),
            Tick(remaining=12.0),
            Tick(remaining=2.0),
            Tick(remaining=2.0, battery_critical=True),
        ],
    )

    print(f"\nRuns 1 and 2 offer the robot the same {RIVAL_ADVANTAGE:.0f}-cell improvement.")
    print("Only how far it had already driven decides the outcome.")


if __name__ == "__main__":
    main()
