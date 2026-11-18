**Design evaluation: B003**

**1. Result**
| sDA300/50% | ASE1000,250h | LEED level | Source |
|---|---|---|---|
| 55.1% | 7.2% | 2 points (sDA >= 55%, ASE <= 10%) | Radiance simulation |

**2. Design inputs**
Windows on: north, south; floor-to-ceiling window, 0.70 m wide × 3.00 m high, sill 0.00 m; overhang 1.28 m.
Input coverage check: passed (inside the sampled design space).

**3. Basis of the result**
Radiance simulation: a LEED threshold lies within the 90% range of a prediction, so the LEED outcome is uncertain.

| Metric | Prediction | 90% range | Thresholds in range | Spread level | Radiance |
|---|---|---|---|---|---|
| sDA300/50% | 54.7% | 52.7–56.7% | 55% | high (2.35 pp) | 55.1% |
| ASE1000,250h | 7.0% | 5.7–8.3% | none | high (0.65 pp) | 7.2% |

Simulation files: thesis_data/results/extensions/workflow_examples/B003_radiance_run

**4. Daylight maps**
Simulated 32 × 32 maps of DA300 and of direct-sun hours from the Radiance run. In Rhino: DA300 map on the work plane (white line: 50%); direct-sun-hour map on a copy of the room to the east (cyan line: 250 h).

**5. Notes**
Warnings: none. The LEED level is a comparative benchmark, not a certification result.

**6. Next step**
2 points reached; sDA would have to reach 75% for 3 points (now 55.1%). You can ask for suggested changes, or change the design and evaluate it again.