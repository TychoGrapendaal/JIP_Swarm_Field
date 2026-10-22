import time
from swarm_field.contracts import TimeUnit
from swarm_field.experiments import PestScenario
from swarm_field.runner import run_experiment

config = PestScenario(size=128, scouts=17, killers=3, until=10 * TimeUnit.MINUTE).into_experiment()
start = time.perf_counter()
run_experiment(config)
print(f"{time.perf_counter() - start:.2f} s")