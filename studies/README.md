# Studies

One script per research question. Where `examples/` shows what the simulation and its tools can do, a study produces the results that go into the report: it runs a fixed set of configurations with fixed seeds, writes every run's metrics series and one summary row per run under `runs/<study>/`, and prints a table of the means over the seeds. Every summary row also records how its run was set up, so a result can be traced back to its configuration without the script.

Run a study as a module from the repository root:

```bash
uv run python -m studies.radio_range --ranges 50 150 300 --seeds 3 --hours 1
```

| Study | Question | Results |
|---|---|---|
| [`radio_range.py`](radio_range.py) | How do detection latency, untreated pests and communication change with the radio range, in the pest experiment? | `runs/radio_range/` |
| [`explore_variants.py`](explore_variants.py) | Which variant and parameters of the EXPLORE seeder keep a field covered and fresh best? | printed as Markdown tables, recorded in [`docs/explore.md`](../docs/explore.md) |

## Writing a study

1. State the question in the module docstring, together with the command that answers it.
2. Build one configuration per condition, its *arm*: usually a `SwarmScenario`, built from a ready one (see the tutorial in [`docs/experiments.md`](../docs/experiments.md)), with only what the question is about changed. Give each an arm name (`arm_name`).
3. Run every arm with every seed through `swarm_field.experiments.Experiment`: `runs_for` makes the runs, and `Experiment.run` runs them in parallel with a run function and writes every result to a new folder in `output/`. For a pest scenario, `measure_pest_response` is the ready run function, with pest response, coverage, idleness and redundancy, and `pest_table_headers` and `pest_table_cells` give its standard columns. A study that needs another measure writes a run function of its own, e.g. `measure_exploration` in `explore_variants.py`. `by_arm`, `mean_and_error`, `paired_difference` and `markdown_table` turn the results into tables. `studies/radio_range.py` and `tests/benchmark_obstacles.py` are examples.
4. Use the same seeds for every condition, so two rows differ only in what the question changes, and run several seeds: a run is a function of its seed, so one seed is one sample.
5. Add the study to the table above.
