wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_test.tgz
tar -xzf openscene_metadata_test.tgz
rm openscene_metadata_test.tgz

export -f wget
export -f tar
export -f rm

# camera
printf "%s\n" {0..31} | parallel -j 8 '
    wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_test_camera/openscene_sensor_test_camera_{}.tgz &&
    echo "Extracting file openscene_sensor_test_camera_{}.tgz" &&
    tar -xzf openscene_sensor_test_camera_{}.tgz &&
    rm openscene_sensor_test_camera_{}.tgz
'

# lidar
printf "%s\n" {0..31} | parallel -j 8 '
    wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_test_lidar/openscene_sensor_test_lidar_{}.tgz &&
    echo "Extracting file openscene_sensor_test_lidar_{}.tgz" &&
    tar -xzf openscene_sensor_test_lidar_{}.tgz &&
    rm openscene_sensor_test_lidar_{}.tgz
'

mv openscene-v1.1/meta_datas navsim_logs/test
mv openscene-v1.1/sensor_blobs sensor_blobs/test
rm -r openscene-v1.1
