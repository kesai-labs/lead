# Tutorial Jupyter Notebooks

The notebooks have been tested in VSCode. This [settings file](https://github.com/autonomousvision/lead/blob/main/.vscode/settings.json) ensures notebooks start with the project root as the working directory.

```{warning}
If using an editor other than VSCode, ensure the Jupyter kernel's root directory matches the project root.
```

## Pipeline Verification Notebooks

### Inspect Expert Output

**Notebook:** [notebooks/inspect_expert_output.ipynb](https://github.com/autonomousvision/lead/blob/main/notebooks/inspect_expert_output.ipynb)

Run the expert agent, produce data samples, and verify correct operation.

### Load Pre-trained Model and Run Offline Inference

**Notebook:** [notebooks/carla_offline_inference.ipynb](https://github.com/autonomousvision/lead/blob/main/notebooks/carla_offline_inference.ipynb)

Load model checkpoints, visualize random data samples, and run offline inference on collected data.

## Understanding Data Format

**Notebook:** [notebooks/data_format.ipynb](https://github.com/autonomousvision/lead/blob/main/notebooks/data_format.ipynb)

Explore the dataloader output structure and learn how to adapt it for custom models.

## Debug Closed-Loop Evaluation

**Notebook:** [notebooks/inspect_sensor_agent_io.ipynb](https://github.com/autonomousvision/lead/blob/main/notebooks/inspect_sensor_agent_io.ipynb)

Interactively debug model inputs and outputs during closed-loop evaluation to diagnose issues.
