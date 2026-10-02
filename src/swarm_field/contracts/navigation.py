"""Navigation seam types — what A.3 hands to the rest of the brain as a goal.

The navigation module takes one goal shape from the decision module: the field of a
``FieldSeeder``, whose targets are ``Seed``s in ``swarm_field.swarm.field``. What it hands back
is one movement action per decision point. The one type here is the other way a goal reaches
the module: a relative displacement, the shape the reactive steering policy consumes when a
potential field cannot guide the robot (see ``docs/navigation-module.md``).
"""

Vector2 = tuple[float, float]
"""A relative displacement in grid cells, ``(dx, dy)`` with y increasing North.

Matches ``Direction.offset``, and the steerer measures it from the robot's own cell to the
cell it is heading for, so it is the offset of a target relative to the robot rather than a
position in the field. The module's prose at §3 speaks of metres; the implemented steering
works in cells (one cell is one ``cell_size`` wide).
"""
