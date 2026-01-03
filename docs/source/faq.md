# FAQ

Additional documentation:

- [scenario_runner](https://scenario-runner.readthedocs.io/en/latest/FAQ/)
- [carla](https://carla.readthedocs.io/en/latest/build_faq/)
- [leaderboard](https://leaderboard.carla.org/help/)

## I get different results when reproducing the provided checkpoints

CARLA evaluation results vary between runs. Typical variations we observe: ~1-2 DS on Bench2Drive, ~5-7 DS on Longest6 v2, ~1.0 DS on Town13. These are empirical estimates and actual variance depends on system configuration and randomness.

## Why are there multiple versions of `leaderboard` and `scenario_runner`?

Each benchmark requires its own evaluation protocol, which means separate forks of those repositories. The expert data collector also uses its own fork.

## How do I create more routes?

See [carla_route_generator](https://github.com/autonomousvision/carla_route_generator) and [LEAD's supplemental](https://ln2697.github.io/assets/pdf/Nguyen2026LEADSUPP.pdf).

## Where can I see the modifications to `leaderboard` and `scenario_runner`?

Our forks with modifications:

- [scenario_runner_autopilot](https://github.com/ln2697/scenario_runner_autopilot)
- [leaderboard_autopilot](https://github.com/ln2697/leaderboard_autopilot)
- [Bench2Drive](https://github.com/ln2697/Bench2Drive)
- [scenario_runner](https://github.com/ln2697/scenario_runner)
- [leaderboard](https://github.com/ln2697/leaderboard)

## Which TransFuser versions are available?

See [this list](https://github.com/autonomousvision/carla_garage/blob/main/docs/history.md).

## How often does CARLA fail to start?

Roughly 10% of launch attempts may fail due to startup hangs, port conflicts, or GPU initialization issues. This is typical CARLA behavior.

**Recovery steps:**

- Clean zombie processes: `bash scripts/clean_carla.sh`
- Restart: `bash scripts/start_carla.sh`
- Verify ports 2000-2002 are available
- Docker: `docker compose restart carla`

## How do I add custom scenarios?

See [3rd_party/scenario_runner_autopilot/srunner/scenarios](https://github.com/autonomousvision/lead/tree/main/3rd_party/scenario_runner_autopilot/srunner/scenarios).

## How does the expert access scenario-specific data?

See [3rd_party/scenario_runner_autopilot/srunner/scenariomanager/carla_data_provider.py](https://github.com/autonomousvision/lead/blob/main/3rd_party/scenario_runner_autopilot/srunner/scenariomanager/carla_data_provider.py).
