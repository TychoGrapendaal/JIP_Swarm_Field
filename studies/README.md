# Studies

One script per research question. Where `examples/` shows what the simulation and its tools can do, a study produces the results that go into the report: it runs a fixed set of configurations with fixed seeds through `swarm_field.experiments.Experiment`, which writes every run's result and the tables of the means over the seeds to a new folder in `output/`. The folder's `experiment.json` records what was run and from which commit, so a result can be traced back to its configuration without the script.

Run a study as a module from the repository root:

```bash
uv run python -m studies.radio_range --ranges 50 150 300 --seeds 3 --hours 1
```

| Study | Question | Results |
|---|---|---|
| [`radio_range.py`](radio_range.py) | How do detection latency, untreated pests and communication change with the radio range, in the pest experiment? | `output/<…>_radio_range/` |
| [`explore_variants.py`](explore_variants.py) | Which variant and parameters of the EXPLORE seeder keep a field covered and fresh best? | `output/<…>_explore_variants/`, recorded in [`docs/explore.md`](../docs/explore.md) |
| [`benchmark_pests.py`](benchmark_pests.py) | How do the seeded and the hop pest responses compare on an open field, over relays (`RELAYED_KILLERS`)? | `output/<…>_pest_benchmark/`, recorded in [`docs/research/hop-field.md`](../docs/research/hop-field.md) |
| [`benchmark_obstacles.py`](benchmark_obstacles.py) | How do the four pest responses compare on the two fields with obstacles (`RIVER_SMALL`, `RIVERS_LARGE`)? | `output/<…>_obstacle_comparison/`, recorded in [`docs/research/hop-field.md`](../docs/research/hop-field.md) |
| [`benchmark_field.py`](benchmark_field.py) | How much faster is the quadtree potential field than a plain Dijkstra over every cell, and how far off are its values? | printed, recorded in [`docs/robot-memory-and-navigation.md`](../docs/robot-memory-and-navigation.md) |
| [`benchmark_sharing.py`](benchmark_sharing.py) | How much of a pest run's wall time goes to knowledge sharing, full against periodic? | printed |

## Writing a study

1. State the question in the module docstring, together with the command that answers it.
2. Build one configuration per condition, its *arm*: usually a `SwarmScenario`, built from a ready one (see the tutorial in [`docs/experiments.md`](../docs/experiments.md)), with only what the question is about changed. Give each an arm name (`arm_name`).
3. Run every arm with every seed through `swarm_field.experiments.Experiment`: `runs_for` makes the runs, and `Experiment.run` runs them in parallel with a run function and writes every result to a new folder in `output/`. For a pest scenario, `measure_pest_response` is the ready run function, with pest response, coverage, idleness and redundancy, and `pest_table_headers` and `pest_table_cells` give its standard columns. A study that needs another measure writes a run function of its own, e.g. `measure_exploration` in `explore_variants.py`. `by_arm`, `mean_and_error`, `paired_difference` and `markdown_table` turn the results into tables. `radio_range.py` and `benchmark_obstacles.py` are examples.
4. Use the same seeds for every condition, so two rows differ only in what the question changes, and run several seeds: a run is a function of its seed, so one seed is one sample.
5. Add the study to the table above.
