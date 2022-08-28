# steam_icp

Paper: Picking Up Speed: Continuous-Time Lidar-Only Odometry using Doppler Velocity Measurements

Dataset:

- KITTI-raw/360: obtain from [CT-ICP codebase](https://github.com/jedeschaud/ct_icp).
- Aeva: waiting for permission.

## Installation

Clone this repository and its submodules.

We use docker to install dependencies The recommended way to build the docker image is

```bash
docker build -t steam_icp \
  --build-arg USERID=$(id -u) \
  --build-arg GROUPID=$(id -g) \
  --build-arg USERNAME=$(whoami) \
  --build-arg HOMEDIR=${HOME} .
```

When starting a container, remember to mount the code and datasets directories to proper locations in the container. An example command to start a docker container with the image is

```bash
docker run -it --name steam_icp \
  --privileged \
  --network=host \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ${HOME}:${HOME}:rw \
  steam_icp
```

(Inside Container) Go to the root directory of this repository and build STEAM-ICP

```bash
bash build.sh
```

## Experiments

Commands below should be run in the docker container.

```bash
# set to the root directory of this repository
WORKING_DIR=$(pwd)

# add library paths
EXTERNAL_ROOT=${WORKING_DIR}/cmake-build-Release/external/install/Release
LGMATH_ROOT=${WORKING_DIR}/cmake-build-Release/lgmath/install/Release
STEAM_ROOT=${WORKING_DIR}/cmake-build-Release/steam/install/Release
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${EXTERNAL_ROOT}/Ceres/lib:${EXTERNAL_ROOT}/glog/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${LGMATH_ROOT}/lib:${STEAM_ROOT}/lib

# run experiments - configure algorithm and dataset in the params file
source ${WORKING_DIR}/steam_icp/install/setup.bash
ros2 run steam_icp steam_icp --ros-args --params-file ${WORKING_DIR}/steam_icp/config/default_config.yaml

# (optional) rviz for visualization - run following commands in another terminal
source /opt/ros/galactic/setup.bash
ros2 run rviz2 rviz2 -d ${WORKING_DIR}/steam_icp/rviz/steam_icp.rviz # launch rviz
```

## Evaluation

TODO

## Citation

TODO

## [License](./LICENSE)
