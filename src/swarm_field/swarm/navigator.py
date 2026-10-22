"""The robot's navigator: the one seam that turns an objective's guidance into a movement action.

The navigator only moves the robot for the winning candidate: :meth:`Navigator.step_towards`
follows the guidance the brain hands it, from the ``FieldProvider``. A grid potential field is
followed step by step, and the reactive ``Steerer`` takes over when the field cannot guide the
robot, which is the fallback that keeps a robot moving over ground it has no plan for. A
``Heading`` goes to the steerer directly.
"""

from swarm_field.contracts import CellId, Idle, Move
from swarm_field.swarm.field import Quadtree, Seed
from swarm_field.swarm.guidance import Guidance, Heading
from swarm_field.swarm.seeder import SeededField
from swarm_field.swarm.steering import QuadtreeSensing, Sensing, Steerer, SteeringConfig

_PLAN_PATIENCE_TICKS = 3
"""Field-following ticks without the potential beating its minimum before the steerer takes over.

The field's window makes the potential approximate, so one tick of standstill is not a loop; a
few of them is.
"""


class Navigator:
    """Moves one robot along the field of its winning candidate, over its quadtree.

    It remembers the field it followed last, and how close following it has brought the
    robot, so it can tell when the field stopped guiding.
    """

    def __init__(
        self,
        quadtree: Quadtree,
        steering: SteeringConfig | None = None,
        heading_sensing: Sensing | None = None,
    ) -> None:
        """Steer over ``quadtree``, and steer headings by ``heading_sensing`` if given.

        A heading is a direction rather than a route, so a robot of the hop method steers it by
        what it senses around itself, without a map; without ``heading_sensing`` headings are
        steered over the quadtree like a stalled field.
        """
        self._config: SteeringConfig = steering if steering is not None else SteeringConfig()
        self._steerer: Steerer = Steerer(QuadtreeSensing(quadtree), self._config)
        self._heading_steerer: Steerer = self._steerer
        if heading_sensing is not None:
            self._heading_steerer = Steerer(heading_sensing, self._config)
        self._followed: SeededField | None = None
        self._heading_target: CellId | None = None
        self._best_value: float | None = None
        self._no_progress_ticks: int = 0

    def step_towards(self, guidance: Guidance, robot: CellId) -> tuple[Move | Idle, CellId | None]:
        """The one movement action of this decision point, on the way towards a field's targets.

        A ``Heading`` is handed to the steerer as it is. A heading towards a new target, or one
        that finds the steerer stalled, resets the steerer: a heading changes as the robot's
        beacons change, so a stall does not hold the robot for good. The rest of this describes
        a ``SeededField``.

        A field guides the robot while it can: the robot steps into the neighbouring cell the
        field points at, which is one step of a route that is near-optimal around the obstacles
        the quadtree knows. The reactive ``Steerer`` takes over when the field cannot guide —
        no target is reachable, or following the field stopped bringing the robot closer, which
        is how the window's edge can make a robot turn back and forth. The steerer then walks
        towards the target the seeder named, and gives way again as soon as the field points
        forward once more.

        Args:
            guidance: What to follow, from ``FieldProvider.field_for``. A different field
                than at the last call (another seeder's, or a rebuild) starts a new plan.
            robot: The robot's current cell, the origin the step is taken from.
        """
        if isinstance(guidance, Heading):
            self._followed = None
            self._forget_plan()
            if guidance.target != self._heading_target or self._heading_steerer.is_stalled:
                self._heading_target = guidance.target
                self._heading_steerer.reset()
            return self._heading_steerer.steer(guidance.offset, robot), guidance.target
        self._heading_target = None
        seeded = guidance
        if seeded is not self._followed:
            self._followed = seeded
            self._forget_plan()
        field = seeded.field
        target = field.target_for(robot)
        if target == robot:
            self._forget_plan()
            return Idle(), target
        if target is None:
            # No wave reaches this robot, so the field has no route to offer. The seeder's
            # targets are known all the same, so the steerer walks towards the most attractive
            # one and stalls if it is walled off for good, which is how the brain hears about it.
            return self._steer_towards(_attractive_seed(seeded.seeds).cell, robot), target

        step = field.next_cell(robot)
        if step is not None and self._plan_progresses(field.value_at(robot)):
            self._steerer.reset()
            return Move(to=step), target
        offset = (float(target[0] - robot[0]), float(target[1] - robot[1]))
        return self._steerer.steer(offset, robot), target

    def _steer_towards(self, target: CellId, robot: CellId) -> Move | Idle:
        """The ``Steerer``'s step towards an absolute ``target``, as a relative offset."""
        offset = (float(target[0] - robot[0]), float(target[1] - robot[1]))
        return self._steerer.steer(offset, robot)

    def _plan_progresses(self, value: float) -> bool:
        """Whether following the field still brings the robot closer to its target.

        ``value`` is the potential at the robot's cell, in cells of travel left to the target,
        so a potential that keeps beating its running minimum is progress. A few ticks of
        anything else means the plan is no longer guiding the robot.
        """
        if self._best_value is None or value < self._best_value - self._config.progress_epsilon:
            self._best_value = value
            self._no_progress_ticks = 0
            return True
        self._no_progress_ticks += 1
        return self._no_progress_ticks < _PLAN_PATIENCE_TICKS

    def _forget_plan(self) -> None:
        """Drop what was tracked about the plan that is no longer in force."""
        self._best_value = None
        self._no_progress_ticks = 0


def _attractive_seed(seeds: tuple[Seed, ...]) -> Seed:
    """The seed the robot is most drawn to: the lowest start value, ties by position."""
    return min(seeds, key=lambda seed: (seed.value, seed.cell))
