import time

from swarm_field.contracts import TimeUnit
from swarm_field.experiments import PestScenario
from swarm_field.runner import run_experiment

config = PestScenario(
    size=256,
    scouts=77,
    killers=3,
    until=10 * TimeUnit.MINUTE,
    growth_rate=0.0,
    pests_per_hour=0,
    spread_probability=0.0,
    sample_period=TimeUnit.DAY,
).into_experiment()
start = time.perf_counter()
run_experiment(config)
print(f"{time.perf_counter() - start:.2f} s")
