# Download and installation

To get started with NAVSIM: 

### 1. Clone the navsim-devkit
This repo serves as a submodule, if the main repo was not cloned recursively, execute
```bash
git submodule update --init --recursive
```
and switch to current branch
```bash
cd $LEAD_PROJECT_ROOT/3rd_party/navsim_workspace/navsimv2.2
git switch <branch>
```

### 2. Download the dataset
Download navhard
```bash
cd $LEAD_PROJECT_ROOT/3rd_party/navsim_workspace/dataset
bash $LEAD_PROJECT_ROOT/3rd_party/navsim_workspace/navsimv2.2/download/download_navhard_two_stage.sh
```

This will download the splits into the download directory. From there, move it to create the following structure.
```angular2html
~/navsim_workspace
├── navsim (containing the devkit)
├── exp
└── dataset
    ├── maps
    ├── navsim_logs
    |    ├── test
    |    ├── trainval
    └── sensor_blobs
    |    ├── test
    |    ├── trainval
    └── navhard_two_stage
    |    ├── openscene_meta_datas
    |    ├── sensor_blobs
    |    ├── synthetic_scene_pickles
    |    └── synthetic_scenes_attributes.csv

```
Set the required environment variables, by adding the following to your `~/.bashrc` file
Based on the structure above, the environment variables need to be defined as:

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/navsimv2.2"
export OPENSCENE_DATA_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/dataset"
```

### 3. Install the navsim-devkit

Finally, install navsim.
To this end, create a new environment and install the required dependencies:

```bash
# Install navsimv2.2
cd ${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/navsimv2.2
conda env create --name navsimv2.2 -f environment.yml
conda activate navsimv2.2
pip install -e . 

# Install lead in navsimv2.2 conda environment
cd $LEAD_PROJECT_ROOT
pip install -e .
```

### 4. Install needed dependencies to integrate CARLA transfuser

```bash
pip install beartype jaxtyping carla numba
```

### 5. Build `navhard` cache
Run those scripts, you might want to adapt them

```bash
bash $LEAD_PROJECT_ROOT/3rd_party/navsim_workspace/navsimv2.2/scripts/evaluation/run_metric_caching_navhard_two_stage.sh
```

This will create the metric cache under `$NAVSIM_EXP_ROOT/metric_cache_navhard_two_stage_v2.2`.