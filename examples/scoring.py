"""The score scale shared by all behaviours, and the helpers to bid on it.

Every ``ActionCandidate.score`` lies in ``[0, 1]``, so that the arbiter can compare bids from
any behaviours and the same score means roughly the same urgency whoever bids it:

- 0 is nothing to gain, such as an unreachable target;
- 0.5 is a target worth exactly the trip to it;
- about 0.73 is the best a field offers at ``SCORE_SCALE_CELLS`` = 10: on a target of value 1;
- from ``ArbiterConfig.override_threshold`` (0.9) up is a need that overrides everything else,
  such as a battery about to run out.

A field-based behaviour turns the potential at the robot's cell into its score with
``score_from_potential``; a behaviour without a field computes its score in ``[0, 1]``
directly, from memory and the robot's state.

This module depends on nothing in ``swarm_field.swarm.behavior``, so behaviours defined there
can import it.
"""

from math import exp, isnan
from typing import Final

SCORE_SCALE_CELLS: Final[float] = 10.0
"""The shared scale, in cells, of the logistic that turns a potential into a score.

One scale for every field-based behaviour keeps their scores comparable. At 10 cells, equal to
the TREAT field's beta, a target of value 1 under the robot scores about 0.73, a target worth
exactly its trip 0.5, and one 30 cells of net travel away about 0.05.
"""


def score_from_potential(potential: float, scale: float = SCORE_SCALE_CELLS) -> float:
    """A field's potential at the robot's cell as a score on the shared ``[0, 1]`` scale.

    The potential is the route to the target plus the target's start value ``-beta * value``,
    in cells, so lower is better. The logistic ``1 / (1 + exp(potential / scale))`` keeps the
    order of the bids, maps a potential of 0 (the target is worth exactly the trip) to 0.5, and
    an unreachable target (an infinite potential) to 0.
    """
    if isnan(potential):
        raise ValueError("potential must be a number")
    exponent = potential / scale
    if exponent > 700.0:
        return 0.0
    if exponent < -700.0:
        return 1.0
    return 1.0 / (1.0 + exp(exponent))
