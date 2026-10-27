# NUM_EXPLORERS — hypotheses (written before the full sweep)

Setup: explorers (scouts) + a fixed group of killers (5 on river-small, 10 on rivers-large), seeded or hop, no relays or mock sources, static pests (no growth, no spread), instant treatment, unlimited battery and doses. The first run below used a 50 m radio, 2 / 0.5 m/s and `SEEDED_KILLER` / `HOP_KILLER`; the script now uses the team defaults (`RIVER_SMALL` / `RIVERS_LARGE`: 100 m radio, 8 / 1 m/s) and the tuned `SEEDED_BEST` / `HOP_BEST`, which addresses the first two review points below. Two loads: 2 and 4 pests per killer-hour. Explorer counts swept on a log scale.

- **H1 (seeded, diminishing returns).** Infested cells and response time fall as explorers are added, steeply at first and then flattening: once idleness (time to first sighting) is small compared with a killer's travel time, more explorers stop helping and the killers become the bottleneck.
- **H2 (hop needs density).** Hop killers depend on an unbroken chain of explorers relaying the beacon from the pest to the killer. With a 50 m radio a sparse swarm rarely forms that chain, so hop does badly with few explorers and improves with density. It should improve more steeply than seeded, but stay worse than seeded at every count, since a seeded killer can act on gossip that reached it at any time, while a hop killer needs a live path.
- **H3 (load moves the knee).** At high load the killers saturate sooner, so the curves flatten at fewer explorers and at a higher level of infested cells than at low load.
- **H4 (coverage and idleness).** Coverage reaches about 1 for all but the smallest swarms, since even a few explorers sweep the field within the run. Idleness falls roughly as 1/explorers.
- **H5 (field size).** On the large field (4x the cells) a similar idleness needs about 4x the explorers. Its killers also walk farther, so the response time is longer at the same explorer density.

Pilot (1 seed, high load, river-small) already hints at H2: seeded went from 7.0 to 3.0 infested cells between 5 and 40 explorers, while hop stayed near 15.

## First run and review (2026-10-27)

The first run (river-small, 5 seeds, 50 m radio) supported H1 and H4, partly supported H2 (hop is worse than seeded at every explorer count), and contradicted H3 (at high load the curves keep falling up to 40 explorers instead of flattening sooner). An independent review found issues to fix before the results go into the report:

- `hop-tuned` was tuned with standing relays and mock sources. With moving explorers, its 1-minute beacon memory looks too short: in a 3-seed check, a 5-minute memory cut hop's infested cells by about 5.5. Sweep the beacon memory and use the best setting before comparing hop with seeded.
- At 50 m the explorers never form a connected network, even with 40 explorers on the small field. Use 100 m (the team default), which goes from sparse to connected over the sweep.
- The pest stream depends on the number of robots, so arms with different explorer counts see different pests. Normalise by the number of spawned pests, or give the pests their own random stream.
- The response time counts only cleared pests, which flatters arms that clear few. Count the open pests too, censored at the end of the run.
- Idleness and coverage count the killers' sensing too, and seeded killers move about 10x more than hop killers.
- Runs are short for sparse swarms: 2 explorers need about 2 h for one sweep. Use 4 h with 8 seeds on the small field, and add a 0-explorer arm.
- Define "enough explorers", e.g. the smallest count within 2 standard errors of the best, and report it as explorers per killer.
