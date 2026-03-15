#!/bin/bash

# Function to download and process current splits
download_current() {
    split=$1
    wget https://s3.eu-central-1.amazonaws.com/avg-projects-2/navsim/navtrain_current_${split}.tgz
    tar -xzf navtrain_current_${split}.tgz --strip-components=1 -C sensor_blobs/trainval
    rm navtrain_current_${split}.tgz
}

# Function to download and process history splits
download_history() {
    split=$1
    wget https://s3.eu-central-1.amazonaws.com/avg-projects-2/navsim/navtrain_history_${split}.tgz
    tar -xzf navtrain_history_${split}.tgz --strip-components=1 -C sensor_blobs/trainval
    rm navtrain_history_${split}.tgz
}

# Export functions so they're available in subshells
export -f download_current
export -f download_history

mkdir -p $LEAD_PROJECT_ROOT/3rd_party/navsim_workspace/dataset
cd $LEAD_PROJECT_ROOT/3rd_party/navsim_workspace/dataset


# Initial setup
mkdir -p sensor_blobs/trainval

for split in {1..4}; do
    download_current $split &
    download_history $split &
done

wait

wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip
unzip nuplan-maps-v1.1.zip
mv nuplan-maps-v1.0 maps
rm nuplan-maps-v1.1.zip

wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_trainval.tgz
tar -xzf openscene_metadata_trainval.tgz --strip-components=1 -C .
mv meta_datas navsim_logs
rm openscene_metadata_trainval.tgz

wait
