<h2 align="center">
<b> LEAD: Minimizing Learner–Expert Asymmetry in End-to-End Driving </b>
</h2>

<p align="center">
  <h4 align="center">
  <a href="https://ln2697.github.io/lead" style="text-decoration: none;">Project Page</a> |
  <a href="https://ln2697.github.io/lead/docs" style="text-decoration: none;">Documentation</a> |
  <a href="https://huggingface.co/ln2697/tfv6" style="text-decoration: none;">CARLA Model Zoo</a> |
  <a href="https://huggingface.co/ln2697/tfv6_navsim" style="text-decoration: none;">NAVSIM Checkpoints</a> |
  <a href="https://huggingface.co/datasets/ln2697/lead_carla" style="text-decoration: none;">CARLA Dataset</a> |
  <a href="https://ln2697.github.io/assets/pdf/Nguyen2026LEADSUPP.pdf" style="text-decoration: none;">Supplementary Material</a> |
  <a href="https://arxiv.org/abs/2512.20563" style="text-decoration: none;">Paper</a>
  </h4>
</p>

<div align="center">

https://github.com/user-attachments/assets/0c2dfb28-93b6-4324-be49-16c6744677da

**Driving stress test:** Closed-loop execution of TransFuser v6, the latest iteration of the TransFuser family, demonstrating stable control in a complex urban scenario under degraded perception and adversarial traffic.

</div>

## Overview

We release the complete pipeline required to achieve state-of-the-art closed-loop performance on the Bench2Drive benchmark. Built around the CARLA simulator, the stack features a data-centric design with:

- Extensive visualization suite and runtime type validation.
- Optimized storage format, packs 72 hours of driving in ~260GB.
- Native support for NAVSIM and Waymo Vision-based E2E. Extending those benchmarks through closed-loop simulation and synthetic data for additional supervision during training.


## Table of Contents

- [Roadmap](#roadmap)
- [Updates](#updates)
- [Quick Start (Get Driving in 20 Minutes)](#quick-start-get-driving-in-20-minutes)
- [Beyond CARLA: Cross-Benchmark Deployment](#beyond-carla-cross-benchmark-deployment)
- [Further Documentation](#further-documentation)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)

## Roadmap

- [x] ✅ Checkpoints and inference code (stable)
- [x] 🟨 Documentation, training pipeline and expert code (released, under test)
- [x] 🟨 Full CARLA dataset release on HuggingFace (released, under test)
- [ ] 🚧 Datasets for cross-benchmark (coming soon)
- [ ] 🚧 Cross-benchmark training tools and documentation (coming soon)

Status: Active development.

## Updates

- **`[2026/01/13]`** CARLA dataset and full CARLA training doc release
  > We publicly release a CARLA dataset generated with the same pipeline as used in the paper. However, due to subsequent refactoring and cleanup of the expert driver, the released dataset is not bit-identical to the dataset used for the reported experiments. A verification of the dataset is running right now.

- **`[2026/01/05]`** Bug in RoutePlanner fixed
  > An index error caused driving policy to to crash at end of routes in Town13. New Driving Score are updated.

- **`[2025/12/24]`** Arxiv paper and code release

## Quick Start (Get Driving in 20 Minutes)

### 1. Environment initialization

Clone the repository and map the project root to your environment

```bash
git clone https://github.com/autonomousvision/lead.git
cd lead

# Set the project root directory and configure paths for CARLA, datasets, and dependencies.
{
  echo -e "export LEAD_PROJECT_ROOT=$(pwd)"  # Set project root variable
  echo "source $(pwd)/scripts/main.sh"       # Persist more environment variables
} >> ~/.bashrc  # Append to bash config to persist across sessions

source ~/.bashrc  # Reload config to apply changes immediately
```

> [!NOTE]
> Please verify that ~/.bashrc reflects these paths correctly.

### 2. Setup experiment infrastructure

We utilize [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install), conda-lock and uv:

```bash
# Install conda-lock and create conda environment
pip install conda-lock && conda-lock install -n lead conda-lock.yml
# Activate conda environment
conda activate lead
# Install dependencies and setup git hooks
pip install uv && uv pip install -r requirements.txt && uv pip install -e .
# Install other tools needed for development
conda install conda-forge::ffmpeg conda-forge::parallel conda-forge::tree conda-forge::gcc
# Optional: Activate git hooks
pre-commit install
```

While waiting for dependencies installation, we recommend CARLA setup on parallel:

```bash
bash scripts/setup_carla.sh # Download and setup CARLA at 3rd_party/CARLA_0915
```

### 3. Model zoo

Pre-trained driving policies are hosted on [HuggingFace](https://huggingface.co/ln2697/tfv6) for reproducibility. These checkpoints follow the TFv6 architecture, but differ in their sensor configurations, vision backbones or dataset composition.

Tab. 1 shows available checkpoints with their performance on three major CARLA benchmarks. As first step, we recommend `tfv6_resnet34` as it provides a good balance between performance and resource usage.

<br>
<div align="center">

| Checkpoint                                                                                    | Description               | Bench2Drive | Longest6 v2 |  Town13  |
| --------------------------------------------------------------------------------------------- | ------------------------- | :---------: | :---------: | :------: |
| [tfv6_regnety032](https://huggingface.co/ln2697/tfv6/tree/main/tfv6_regnety032)               | TFv6                      |  **95.2**   |   **62**    | **5.24** |
| [tfv6_resnet34](https://huggingface.co/ln2697/tfv6/tree/main/tfv6_resnet34)                   | ResNet34 Backbone         |    94.7     |     57      |   5.01   |
| [4cameras_resnet34](https://huggingface.co/ln2697/tfv6/tree/main/4cameras_resnet34)           | Additional rear camera    |    95.1     |     53      |    -     |
| [noradar_resnet34](https://huggingface.co/ln2697/tfv6/tree/main/noradar_resnet34)             | No radar sensor           |    94.7     |     52      |    -     |
| [visiononly_resnet34](https://huggingface.co/ln2697/tfv6/tree/main/visiononly_resnet34)       | Vision-only driving model |    91.6     |     43      |    -     |
| [town13heldout_resnet34](https://huggingface.co/ln2697/tfv6/tree/main/town13heldout_resnet34) | Generalization evaluation |    93.1     |     52      |   3.52   |

**Table 1:** Performance of pre-trained checkpoints. We report Driving Score, for which higher is better.

</div>
<br>

To download one checkpoint:

```bash
bash scripts/download_one_checkpoint.sh
```

Or download all checkpoints at once with <a href="https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage">git lfs</a>

```bash
git clone https://huggingface.co/ln2697/tfv6 outputs/checkpoints
cd outputs/checkpoints
git lfs pull
```

### 4. Verify driving stack

To initiate closed-loop evaluation and verify the integration of the driving stack, execute the following:

```bash
# Start driving environment
bash scripts/start_carla.sh
# Start policy on one route
bash scripts/eval_bench2drive.sh
```

Driving logs will be saved to <code>outputs/local_evaluation</code> with the following structure:

```html
outputs/local_evaluation/23687
├── 23687_debug.mp4
├── 23687_demo.mp4
├── checkpoint_endpoint.json
├── debug_images
├── demo_images
├── input_log
└── metric_info.json
```

> [!TIP]
> 1. Disable video recording in [config_closed_loop](lead/inference/config_closed_loop.py) by turning off `produce_demo_video` and `produce_debug_video`.
> 2. If memory is limited, modify the file prefixes to load only the first checkpoint seed. By default, the pipeline loads all three seeds as an ensemble.

### 5. Verify autopilot

Verify the expert policy and data acquisition pipeline by executing a test run on a sample route:

```bash
# Start CARLA if not done already
bash scripts/start_carla.sh
# Run expert on one route
bash scripts/run_expert.sh
```

Data collected will be stored at <code>data/expert_debug</code> and should have following structure:

```html
data/expert_debug
├── data
│   └── BlockedIntersection
│       └── 999_Rep-1_Town06_13_route0_12_22_22_34_45
│           ├── bboxes
│           ├── depth
│           ├── depth_perturbated
│           ├── hdmap
│           ├── hdmap_perturbated
│           ├── lidar
│           ├── metas
│           ├── radar
│           ├── radar_perturbated
│           ├── results.json
│           ├── rgb
│           ├── rgb_perturbated
│           ├── semantics
│           └── semantics_perturbated
└── results
    └── Town06_13_result.json
```

## Beyond CARLA: Cross-Benchmark Deployment

The LEAD pipeline and TFv6 models are deployed as **reference implementations and benchmark entries** across multiple autonomous driving simulators and evaluation suites:

* **[Waymo Vision-based End-to-End Driving Challenge (DiffusionLTF)](https://waymo.com/open/challenges/2025/e2e-driving/)**
  Strong baseline entry for the inaugural end-to-end driving challenge hosted by Waymo, achieving **2nd place** in the final leaderboard.

* **[NAVSIM v1 (LTFv6)](https://huggingface.co/spaces/AGC2024-P/e2e-driving-navtest)**
  Latent TransFuser v6 is an updated reference baseline for the `navtest` split, improving PDMS by +3 points over the Latent TransFuser baseline, used to evaluate navigation and control under diverse driving conditions.

* **[NAVSIM v2 (LTFv6)](https://huggingface.co/spaces/AGC2025/e2e-driving-navhard)**
  The same Latent TransFuser v6 improves EPMDS by +6 points over the Latent TransFuser baseline, targeting distribution shift and scenario complexity.

* **[NVIDIA AlpaSim Simulator (TransFuserModel)](https://github.com/NVlabs/alpasim)**
  Adapting the NAVSIM's Latent TransFuser v6 checkpoints, AlpaSim also features an official TransFuser driver, serving as a baseline policy for closed-loop simulation.

These deployments demonstrate the flexibility of LEAD as a portable training and inference stack that can be adapted across environments, sensor configurations, and evaluation protocols.

LEAD complements existing benchmarks by extending them with reproducible long-horizon closed-loop evaluation and additional synthetic supervision benefits of CARLA.

## Further Documentation

For more detailed instructions, see the [full documentation](https://ln2697.github.io/lead/docs). In particular:
- [Training data collection](https://ln2697.github.io/lead/docs/data_collection.html)
- [Training](https://ln2697.github.io/lead/docs/carla_training.html)
- [Evaluation](https://ln2697.github.io/lead/docs/carla_training.html)

## Acknowledgements

Special thanks to [carla_garage](https://github.com/autonomousvision/carla_garage) for the foundational codebase. We also thank the creators of the numerous open-source projects we use:

- [PDM-Lite](https://github.com/OpenDriveLab/DriveLM/blob/DriveLM-CARLA/pdm_lite/docs/report.pdf), [leaderboard](https://github.com/carla-simulator/leaderboard), [scenario_runner](https://github.com/carla-simulator/scenario_runner), [NAVSIM](https://github.com/autonomousvision/navsim), [Waymo Open Dataset](https://github.com/waymo-research/waymo-open-dataset)

Other helpful repositories:

- [SimLingo](https://github.com/RenzKa/simlingo), [PlanT2](https://github.com/autonomousvision/plant2), [Bench2Drive Leaderboard](https://github.com/autonomousvision/Bench2Drive-Leaderboard), [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive/), [CaRL](https://github.com/autonomousvision/CaRL)

Long Nguyen led development of the project. Kashyap Chitta, Bernhard Jaeger, and Andreas Geiger contributed through technical discussion and advisory feedback. Daniel Dauner provided guidance with NAVSIM.

## Citation

If you find this work useful, please consider giving this repository a star ⭐ and citing our work in your research:

```bibtex
@article{Nguyen2025ARXIV,
  title={LEAD: Minimizing Learner-Expert Asymmetry in End-to-End Driving},
  author={Nguyen, Long and Fauth, Micha and Jaeger, Bernhard and Dauner, Daniel and Igl, Maximilian and Geiger, Andreas and Chitta, Kashyap},
  journal={arXiv preprint arXiv:2512.20563},
  year={2025}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
