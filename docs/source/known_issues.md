# Known Issues

Known issues in the evaluation pipeline and models.

## Town13 Evaluation Has Target Point Bugs

The evaluation pipeline has bugs affecting target points on Town13, visible in videos on the [project website](https://ln2697.github.io/lead). This degrades policy performance, so Town13 numbers don't reflect true capability.

## Multi-GPU Training Can Slightly Degrade Performance

Training on 4 GPUs sometimes yields marginally lower closed-loop performance than single-GPU training. The effect is small and doesn't change qualitative conclusions, but appears consistently in certain runs.

## Static Graph Can Degrade Performance

We avoid [static_graph](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) in the pipeline due to observed performance issues.

## CARLA Waypoint PID Controller Needs Better Tuning

A well-tuned controller (e.g., MPC) can significantly improve performance. Preliminary experiments showed ~5-7 DS improvement on Bench2Drive for TFv5, though these numbers are approximate since controller tuning wasn't the focus.

## CARLA 0.9.16 Has Goal-Point Issues

CARLA 0.9.16 currently has problems with the goal-point pipeline that degrade policy behavior. We don't recommend evaluating models on this version.

## Expert Performance Drops on Town13

The expert performs reliably on short and medium routes but shows notable performance degradation on Town13. The causes are under investigation.

## Expert Is Designed for Simplicity, Not Optimality

The provided expert prioritizes simplicity and extensibility over optimal performance. We encourage future work to improve the expert or explore alternative designs within the LEAD framework.
