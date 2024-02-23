# STEAM-ICP

Paper: [Continuous-Time Radar-Inertial and Lidar-Inertial Odometry using a Gaussian Process Motion Prior](https://arxiv.org/abs/2402.06174)

Previous Paper: [Picking Up Speed: Continuous-Time Lidar-Only Odometry using Doppler Velocity Measurements](https://ieeexplore.ieee.org/document/9968059)

For code that pertains to the picking-up-speed paper, `git checkout picking_up_speed`

Dataset: [Boreas](https://www.boreas.utias.utoronto.ca/) and [KITTI-raw/360 (from CT-ICP)](https://github.com/jedeschaud/ct_icp) and [Aeva](https://drive.google.com/file/d/1JpQNnXejow3qy1qp5tVzak9qnuFmjYHW/view?usp=share_link).

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

When starting a container, remember to mount the code, dataset, and output directories to proper locations in the container.
An example command to start a docker container with the image is

```bash
docker run -it --name steam_icp \
  --privileged \
  --network=host \
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

In terms of outputs, you should get something similar to the files in the [results](./results) directory.

## Evaluation

See python scripts in the [script](./script) directory for evaluation.

## The following papers are based on this repository and can be cited as follows:

```bibtex
@ARTICLE{wu_ral23,
  author={Wu, Yuchen and Yoon, David J. and Burnett, Keenan and Kammel, Soeren and Chen, Yi and Vhavle, Heethesh and Barfoot, Timothy D.},
  journal={IEEE Robotics and Automation Letters}, 
  title={Picking up Speed: Continuous-Time Lidar-Only Odometry Using Doppler Velocity Measurements}, 
  year={2023},
  volume={8},
  number={1},
  pages={264-271},
  doi={10.1109/LRA.2022.3226068}
}
```

```bibtex
@article{burnett_arxiv24,
  title={Continuous-Time Radar-Inertial and Lidar-Inertial Odometry using a Gaussian Process Motion Prior},
  author={Burnett, Keenan and Schoellig, Angela P and Barfoot, Timothy D},
  journal={arXiv preprint arXiv:2402.06174},
  year={2024}
}
```

## [License](./LICENSE)
