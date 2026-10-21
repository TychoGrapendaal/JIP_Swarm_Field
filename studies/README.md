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
2. Build one `ExperimentConfig` per condition and seed. Start from a scenario of the catalogue (`swarm_field.experiments.build_named_experiment`) or a hand-built one (`PestScenario`, `ExploreScenario`), change only what the question is about, and label each run `CONDITION/SEED`.
3. Run them with `swarm_field.runner.sweep`, and write the results with `export_summary_csv`, `export_summary_json` and `RunSummary.export`.
4. Use the same seeds for every condition, so two rows differ only in what the question changes, and run several seeds: a run is a function of its seed, so one seed is one sample.
5. Add the study to the table above.
