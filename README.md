<h2 align="center">
<b> LEAD: Minimizing Learner–Expert Asymmetry in End-to-End Driving </b>
</h2>

<p align="center">
  <h4 align="center">
  <a href="https://ln2697.github.io/lead" style="text-decoration: none;">Project Page</a> |
  <a href="https://ln2697.github.io/lead/docs" style="text-decoration: none;">Documentation</a> |
  <a href="https://huggingface.co/ln2697/TFv6" style="text-decoration: none;">Checkpoints</a> |
  <a href="https://huggingface.co/ln2697/LEAD" style="text-decoration: none;">Dataset</a> |
  <a href="https://ln2697.github.io/assets/pdf/Nguyen2026LEADSUPP.pdf" style="text-decoration: none;">Supplementary Material</a> |
  <a href="https://arxiv.org/abs/2512.20563" style="text-decoration: none;">Paper</a>
  </h4>
</p>

<p align="center">Official implementation of LEAD & TransFuser v6, an expert-student policy pair for autonomous driving research in CARLA. Includes a complete pipeline for data collection, training, and evaluation.</p>

<div align="center">

https://github.com/user-attachments/assets/311f9f81-17f1-4741-a37d-93da81d25d08

**End-to-end stress test:** Closed-loop execution of TransFuser v6, a state-of-the-art driving policy in CARLA, demonstrating stable control in a complex urban scenario under degraded perception condition and dense adversarial traffic.

</div>

## Overview

LEAD provides a comprehensive framework for end-to-end driving research in the CARLA simulator, features:

- **Data-centric infrastructure**:
  - Comprehensive visualization suite.
  - Runtime tensor shape and type validation.
  - Efficient storage format, fitting 72 hours of driving in ~200GB.

- **Multi-dataset training pipeline**:
  - Support for real-world datasets: NAVSIM, Waymo Vision-based E2E, Waymo Perception.
  - Domain adaptation through co-training on heterogeneous data sources.

## Table of Contents

- [Roadmap](#roadmap)
- [Updates](#updates)
- [Setup Project (15 minutes)](#setup-project-15-minutes)
- [Quick Start (5 minutes)](#quick-start-5-minutes)
- [Bench2Drive Results](#bench2drive-results)
- [Documentation and Resources](#documentation-and-resources)
- [External Resources](#external-resources)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Roadmap

- [x] ✅ Checkpoints and inference code (stable)
- [x] 🚧 Documentation, training pipeline and expert code (partial release)
- [ ] Full dataset release on HuggingFace
- [ ] Cross-dataset training tools and documentation

Status: Active development. Core code and checkpoints are released; remaining components coming soon.

## Updates

- `2025/12/24` Arxiv paper and code release

## Setup Project (15 minutes)

**1. Setup repository**

Clone repository

```bash
git clone https://github.com/autonomousvision/lead.git
cd lead
```

Set the project root directory and configure paths for CARLA, datasets, and dependencies.

```bash
{
  echo -e "export LEAD_PROJECT_ROOT=$(pwd)"  # Set project root variable
  echo "source $(pwd)/scripts/main.sh"       # Persist more environment variables
} >> ~/.bashrc  # Append to bash config to persist across sessions

source ~/.bashrc  # Reload config to apply changes immediately
```

<details>
<summary>In case you use Zsh, you can instead run:</summary>

```bash
{
  echo
  echo "export LEAD_PROJECT_ROOT=$(pwd)"  # Set project root variable
  echo "source $(pwd)/scripts/main.sh"    # Persist more environment variables
} >> ~/.zshrc  # Append to zsh config to persist across sessions

source ~/.zshrc  # Reload config to apply changes immediately
```

</details>

> [!NOTE]
> For some terminal (VSCode) you might need to manually edit `~/.bashrc` and `~/.zshrc` to ensure the lines are clean.

**2. Create python environment**

We use [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) for this project. First, create Conda environment

```bash
# Install conda-lock
pip install conda-lock

# Create Conda environment
conda-lock install -n lead conda-lock.yml

# Activate conda environment
conda activate lead
```

Install dependencies with uv

```bash
# Install uv
pip install uv

# Install dependencies
uv pip install -r requirements.txt

# Install project
uv pip install -e .
```

Further dependencies
```bash
# Set-up git hooks
pre-commit install

# Install other tools needed for development
conda install conda-forge::ffmpeg conda-forge::parallel conda-forge::tree conda-forge::gcc
```

While waiting for dependencies installation, we recommend CARLA setup on parallel.

**3. Setup CARLA**

Install CARLA 0.9.15 at `3rd_party/CARLA_0915`

```bash
bash scripts/setup_carla.sh
```

<details>
<summary>If you have an existing CARLA installation, you can softlink it by running:</summary>

```bash
ln -s /your/carla/path $LEAD_PROJECT_ROOT/3rd_party/CARLA_0915
```

</details>

## Quick Start (5 minutes)

After setting up the project, you can evaluate the pre-trained checkpoints or expert (collecting data) to verify the pipeline.

**1. Download model checkpoints**

We provide pre-trained checkpoints on [HuggingFace](https://huggingface.co/ln2697/TFv6) for reproducibility. Fig. 1 shows the TFv6 network architecture. All checkpoints share the same structure but differ in their sensor configurations (e.g., with/without radar, different camera setups).

<br>

<p align="center">
  <img src="https://ln2697.github.io/lead/static/images/tfv6.png" alt="TFv6 Architecture" width="90%">
</p>

<p align="center"><b>Figure 1:</b> TFv6 architecture.</p>

<br>

Tab. 1 shows available checkpoints with their performance on three major CARLA benchmarks. To verify the pipeline, we recommend downloading `tfv6_resnet34` as it provides a good balance between performance and resource usage.

<br>
<div align="center">

| Checkpoint                                                                                    | Description               | Bench2Drive | Longest6 v2 |  Town13  |
| --------------------------------------------------------------------------------------------- | ------------------------- | :---------: | :---------: | :------: |
| [tfv6_regnety032](https://huggingface.co/ln2697/TFv6/tree/main/tfv6_regnety032)               | TFv6                      |  **95.2**   |   **62**    | **5.01** |
| [tfv6_resnet34](https://huggingface.co/ln2697/TFv6/tree/main/tfv6_resnet34)                   | ResNet34 Backbone         |    94.7     |     57      |   3.31   |
| [4cameras_resnet34](https://huggingface.co/ln2697/TFv6/tree/main/4cameras_resnet34)           | Additional rear camera    |    95.1     |     53      |    -     |
| [noradar_resnet34](https://huggingface.co/ln2697/TFv6/tree/main/noradar_resnet34)             | No radar sensor           |    94.7     |     52      |    -     |
| [visiononly_resnet34](https://huggingface.co/ln2697/TFv6/tree/main/visiononly_resnet34)       | Vision-only driving model |    91.6     |     43      |    -     |
| [town13heldout_resnet34](https://huggingface.co/ln2697/TFv6/tree/main/town13heldout_resnet34) | Generalization evaluation |    93.1     |     52      |   2.65   |

**Table 1:** Performance of pre-trained checkpoints. We report Driving Score where higher is better.

</div>
<br>

To download one checkpoint:

```bash
# Create checkpoint directory
mkdir -p outputs/checkpoints/tfv6_resnet34

# Download config
wget https://huggingface.co/ln2697/TFv6/resolve/main/tfv6_resnet34/config.json \
  -O outputs/checkpoints/tfv6_resnet34/config.json

# Download one checkpoint
wget https://huggingface.co/ln2697/TFv6/resolve/main/tfv6_resnet34/model_0030_0.pth \
  -O outputs/checkpoints/tfv6_resnet34/model_0030_0.pth
```


<details>
<summary>Alternatively, to download all checkpoints at once with <a href="https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage">git lfs</a>:</summary>

```bash
git clone https://huggingface.co/ln2697/TFv6 outputs/checkpoints
cd outputs/checkpoints
git lfs pull
```

</details>

**2. Run model evaluation**

To start the evaluation, follow those commands

```bash
# Start CARLA server
bash scripts/start_carla.sh

# Evaluate one Bench2Drive route
bash scripts/eval_bench2drive.sh

# Optional: clean CARLA server
bash scripts/clean_carla.sh
```

<details>
<summary>Results will be saved to <code>outputs/local_evaluation</code> with the following structure:</summary>

```html
outputs/local_evaluation
├── 23687
│   ├── checkpoint_endpoint.json
│   ├── debug_images
│   ├── demo_images
│   └── metric_info.json
├── 23687_debug.mp4
└── 23687_demo.mp4
```
</details>

> [!TIP]
> 1. Disable video generation in [config_closed_loop](lead/inference/config_closed_loop.py) by turning off `produce_demo_video` and `produce_debug_video`.
> 2. Modify the file prefixes to load only the first seed, if memory is limited. By default, the pipeline loads all three seeds as an ensemble.

**3. Run expert evaluation**

Evaluate expert and collect data

```bash
# Start CARLA if not done already
bash scripts/start_carla.sh

# Run expert on one route
bash scripts/run_expert.sh

# Optional: clean CARLA server
bash scripts/clean_carla.sh
```

<details>
<summary>Data collected will be stored at <code>data/expert_debug</code> and should have following structure:</summary>

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

</details>

## Bench2Drive Results

We evaluate TFv6 on the [Bench2Drive](https://github.com/autonomousvision/Bench2Drive-Leaderboard/tree/ab8021b027fa9c4765f9a732355d3b2ae93736a0) benchmark, which consists of 220 routes across multiple towns with challenging weather conditions and traffic scenarios. Tab. 2 compares TFv6 with other recent methods.

<br>

<div align="center">

| Method          |    DS     |    SR     |   Merge   | Overtake  | EmgBrake  | Give Way  | Traffsign | Venue  |
| --------------- | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | ------ |
| TF++ (TFv5)     |   84.21   |   67.27   |   58.75   |   57.77   |   83.33   |   40.00   |   82.11   | ICCV23 |
| SimLingo        |   85.07   |   67.27   |   54.01   |   57.04   |   88.33   | **53.33** |   82.45   | CVPR25 |
| R2SE            |   86.28   |   69.54   |   53.33   |   61.25   |   90.00   |   50.00   |   84.21   | -      |
| HiP-AD          |   86.77   |   69.09   |   50.00   |   84.44   |   83.33   |   40.00   |   72.10   | ICCV25 |
| BridgeDrive     |   86.87   |   72.27   |   63.50   |   57.77   |   83.33   |   40.00   |   82.11   | -      |
| DiffRefiner     |   87.10   |   71.40   |   63.80   |   60.00   |   85.00   |   50.00   |   86.30   | AAAI26 |
| **TFv6 (Ours)** | **95.28** | **86.80** | **72.50** | **97.77** | **91.66** |   40.00   | **89.47** | -      |

**Table 2**: Bench2Drive Performance. DS = Driving Score, SR = Success Rate. Higher is better.

</div>

<br>

## Documentation and Resources

For detailed training, data-collection, and large-scale experiment instructions, see the [full documentation](https://ln2697.github.io/lead/docs). In particular, we provide:

- [Tutorial notebooks](https://ln2697.github.io/lead/docs/jupyter_notebooks.html)
- [Frequently asked questions](https://ln2697.github.io/lead/docs/faq)
- [Known issues](https://ln2697.github.io/lead/docs/known_issues.html)

We maintain custom forks of CARLA evaluation tools with our modifications:

- [scenario_runner_autopilot](https://github.com/ln2697/scenario_runner_autopilot), [leaderboard_autopilot](https://github.com/ln2697/leaderboard_autopilot), [Bench2Drive](https://github.com/ln2697/Bench2Drive), [scenario_runner](https://github.com/ln2697/scenario_runner), [leaderboard](https://github.com/ln2697/leaderboard)

## External Resources

Useful documentations from other repositories:

- A cheatsheat of [CARLA coordinate systems](https://github.com/autonomousvision/carla_garage/blob/leaderboard_2/docs/coordinate_systems.md)
- About [Longest6 v2 benchmark](https://github.com/autonomousvision/CaRL/tree/main/CARLA#longest6-v2) and [Town13 benchmark](https://github.com/autonomousvision/carla_garage?tab=readme-ov-file#carla-leaderboard-20-validation-routes)
- Generate scenario XML files [randomly](https://github.com/autonomousvision/CaRL/tree/main/CARLA#scenario-generation) and [manually](https://github.com/autonomousvision/carla_route_generator)

Other helpful repositories:

- [SimLingo](https://github.com/RenzKa/simlingo), [PlanT2](https://github.com/autonomousvision/plant2), [Bench2Drive Leaderboard](https://github.com/autonomousvision/Bench2Drive-Leaderboard), [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive/), [CaRL](https://github.com/autonomousvision/CaRL)

E2E self-driving research:

- [Why study self-driving?](https://emergeresearch.substack.com/p/why-study-self-driving?triedRedirect=true) A timely perspective given recent commercial deployments of autonomous taxi services.
- [End-to-end Autonomous Driving: Challenges and Frontiers](https://arxiv.org/abs/2306.16927) A comprehensive introductory survey of the field.
- [Common Mistakes in Benchmarking Autonomous Driving](https://github.com/autonomousvision/carla_garage/blob/leaderboard_2/docs/common_mistakes_in_benchmarking_ad.md) Common pitfalls to avoid when evaluating autonomous driving systems.

## Acknowledgements

Special thanks to [carla_garage](https://github.com/autonomousvision/carla_garage) for the foundational codebase. We also thank the creators of the numerous open-source projects we use:

- [PDM-Lite](https://github.com/OpenDriveLab/DriveLM/blob/DriveLM-CARLA/pdm_lite/docs/report.pdf), [leaderboard](https://github.com/carla-simulator/leaderboard), [scenario_runner](https://github.com/carla-simulator/scenario_runner), [NAVSIM](https://github.com/autonomousvision/navsim), [Waymo Open Dataset](https://github.com/waymo-research/waymo-open-dataset)

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
