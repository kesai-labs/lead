# Collect Data (10 minutes)

This guide covers local data collection on CARLA using LEAD.

```{note}
This tutorial is designed for local development and debugging. For large-scale data collection on HPC clusters, refer to the [SLURM Guide](slurm_data_collection.md).
```

## Start Data Collection Locally

1. Start CARLA server:

```bash
bash scripts/start_carla.sh
```

2. Run the expert agent:

```bash
python lead/leaderboard_wrapper.py --routes data/data_routes/lead/noScenarios/short_route.xml
```

## Inspect Collected Data

Inspect collected data using the notebook [notebooks/inspect_expert_output.ipynb](https://github.com/autonomousvision/lead/blob/main/notebooks/inspect_expert_output.ipynb).

## [Optional] Expert Overview

```{note}
The following sections are optional and only relevant if you want to understand or modify the expert's behavior.
```

The data collection expert in [lead/expert/expert.py](https://github.com/autonomousvision/lead/blob/main/lead/expert/expert.py) operates in four steps:

1. Proposes a shortest path by searching the lane graph with A\*
2. Augments this path to avoid collisions with static obstacles
3. Proposes an initial target speed using the Intelligent Driver Model (IDM)
4. Adjusts target speed to avoid collisions with dynamic hazards

## [Optional] Expert Implementation Walkthrough

```{note}
The expert is not optimal for human-level driving or policy transfer to learned models. We encourage experimentation with the expert's logic and parameters if you identify suboptimal behaviors.
```

The main driving logic is implemented in [lead/expert/expert.py](https://github.com/autonomousvision/lead/blob/main/lead/expert/expert.py),
with helper functions and properties in [lead/expert/expert_base.py](https://github.com/autonomousvision/lead/blob/main/lead/expert/expert_base.py) and data collection logic in [lead/expert/expert_data.py](https://github.com/autonomousvision/lead/blob/main/lead/expert/expert_data.py).

The entry point is the `run_step` function in [lead/expert/expert.py](https://github.com/autonomousvision/lead/blob/main/lead/expert/expert.py), which:

1. Collects sensor data, bounding boxes, and metadata
2. Executes the driving logic in `_get_control`

The expert accesses CARLA simulator internal states to handle various scenarios. The central state registry is in [3rd_party/scenario_runner_autopilot/srunner/scenariomanager/carla_data_provider.py](https://github.com/autonomousvision/lead/blob/main/3rd_party/scenario_runner_autopilot/srunner/scenariomanager/carla_data_provider.py).

To understand scenario data registration, examine the scenario classes in [3rd_party/scenario_runner_autopilot/srunner/scenarios](https://github.com/autonomousvision/lead/tree/main/3rd_party/scenario_runner_autopilot/srunner/scenarios).

We maintain custom forks of CARLA evaluation tools with modifications for expert access: [scenario_runner_autopilot](https://github.com/ln2697/scenario_runner_autopilot) and [leaderboard_autopilot](https://github.com/ln2697/leaderboard_autopilot).

## [Optional] Expert Evaluation

The expert can be evaluated directly on Longest6 v2 and Town13 benchmarks.

Bench2Drive evaluation is non-trivial due to extensive code modifications for internal state access. In our paper, we estimated a lower bound on expert performance by evaluating Bench2Drive routes with the expert's evaluation tools. Routes with only `MinSpeed` infractions are assigned a score of 100.
