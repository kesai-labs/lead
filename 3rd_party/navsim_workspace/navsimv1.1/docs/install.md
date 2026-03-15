# Download and installation

To get started with NAVSIM: 

### 1. [Only relevant for maintainers] Clone the directory
This repo serves as a submodule, if the main repo was not cloned recursively, execute
```bash
git submodule update --init --recursive
```

and switch to current branch
```bash
cd $LEAD_PROJECT_ROOT/3rd_party/navsim_workspace/navsimv1.1
git switch main # or other branch
```

### 2. Download the demo data
You need to download the OpenScene logs and sensor blobs, as well as the nuPlan maps.
```bash
mkdir -p $LEAD_PROJECT_ROOT/3rd_party/navsim_workspace/dataset
cd $LEAD_PROJECT_ROOT/3rd_party/navsim_workspace/dataset
bash $LEAD_PROJECT_ROOT/3rd_party/navsim_workspace/navsimv1.1/download/download_maps.sh
```

Next download the data splits you want to use.
```bash
cd $LEAD_PROJECT_ROOT/3rd_party/navsim_workspace/dataset
bash $LEAD_PROJECT_ROOT/3rd_party/navsim_workspace/navsimv1.1/download/download_navtrain_parallel.sh
bash $LEAD_PROJECT_ROOT/3rd_party/navsim_workspace/navsimv1.1/download/download_test_parallel.sh
```

Restructure after download
```bash
mv $LEAD_PROJECT_ROOT/3rd_party/navsim_workspace/dataset/test_navsim_logs/test \
   $LEAD_PROJECT_ROOT/3rd_party/navsim_workspace/dataset/navsim_logs/test

mv $LEAD_PROJECT_ROOT/3rd_party/navsim_workspace/dataset/test_sensor_blobs/test \
   $LEAD_PROJECT_ROOT/3rd_party/navsim_workspace/dataset/sensor_blobs/test
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
         ├── test
         ├── trainval
```
Set the required environment variables, by adding the following to your `~/.bashrc` file
Based on the structure above, the environment variables need to be defined as:
```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/navsimv1.1"
export OPENSCENE_DATA_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/dataset"
```

### 3. Install the navsim-devkit
Finally, install navsim.
To this end, create a new environment and install the required dependencies:

```
# Install navsimv1.1
cd ${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/navsimv1.1
conda env create --name navsimv1.1 -f environment.yml
conda activate navsimv1.1
pip install -e . 

# Install lead in navsimv1.1 conda environment
cd $LEAD_PROJECT_ROOT
pip install -e .
```

### 4. Install needed dependencies to integrate CARLA Model

Install those dependencies in `navsimv1.1` environment.

```bash
pip install beartype jaxtyping carla numba
```

### 5. Build `navtrain` training cache

```bash
bash $LEAD_PROJECT_ROOT/3rd_party/navsim_workspace/navsimv1.1/scripts/training/run_navtrain_caching.sh
```

### 6. Build `navtest` metrics cache

```bash
bash $LEAD_PROJECT_ROOT/3rd_party/navsim_workspace/navsimv1.1/scripts/evaluation/run_navtest_caching.sh
```

### 7. Common issues
- In case of weird errors, most likely redownloading data would be helpful.
- Remove the `rm` in download scripts to inspect the downloaded files.