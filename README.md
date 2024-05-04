# STEAM-ICP

Paper: [IMU as an Input vs. a Measurement of the State in Inertial-Aided State Estimation](https://arxiv.org/abs/2403.05968)

In this paper, we implemented lidar-inertial odometry using a Singer prior where body-centric acceleration is included in the state. In the paper, we refer to this as our Singer prior lidar(-inertial) odometry which can be configured to be lidar-only, lidar + gyro, or lidar-inertial using both accelerometer and gyroscope. This is implemented in `steam_icp/src/odometry/steam_lio.cpp` where `use_imu=True` in the configuration yaml enables the use of an IMU.

Paper: [Continuous-Time Radar-Inertial and Lidar-Inertial Odometry using a Gaussian Process Motion Prior](https://arxiv.org/abs/2402.06174)

In the paper, we implemented radar-inertial and lidar-inertial odometry using a white-noise-on-acceleration prior where accelerometer measurements are preintegrated to form relative velocity factors. We compare several variants of our approach such as lidar-only: STEAM-LO, lidar-inertial: STEAM-LIO, radar-only: STEAM-RO, and radar-inertial: STEAM-RIO. STEAM-LO and STEAM-LIO are implemented in `steam_icp/src/odometry/steam_lo.cpp` where `use_imu=True` in the configuration yaml enables the use of an IMU. Similarly, STEAM-RO and STEAM-RIO are implemented in `steam_icp/odometry/steam_ro.cpp`.

To be clear, the algorithms associated with the "continuous-time radar-inertial ..." are implemented in `steam_lo.cpp` and `steam_ro.cpp` which use the white-noise-on-acceleration prior. The algorithms associated with the "imu as input ..." paper are implemented in `steam_lio.cpp` which uses a Singer prior.

Previous Paper: [Picking Up Speed: Continuous-Time Lidar-Only Odometry using Doppler Velocity Measurements](https://ieeexplore.ieee.org/document/9968059)

In our previous work, we demonstrate continuous-time lidar odometry which incorporated Doppler velocity measurements from an Aeva lidar. For code that pertains to the picking-up-speed paper, `git checkout picking_up_speed`

Datasets: [Boreas (ours)](https://www.boreas.utias.utoronto.ca/) and [KITTI-raw/360 (made available by the authors of CT-ICP)](https://github.com/jedeschaud/ct_icp) and [Aeva (a collaboration between us and Aeva)](https://drive.google.com/file/d/1JpQNnXejow3qy1qp5tVzak9qnuFmjYHW/view?usp=share_link) and [Newer College Dataset](https://ori-drs.github.io/newer-college-dataset/)

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
# this would run STEAM-ICP from our "picking up speed" paper
ros2 run steam_icp steam_icp --ros-args --params-file ${WORKING_DIR}/steam_icp/config/default_config.yaml
# this would run STEAM-LIO from our "continuous-time radar-inertial..." paper (on the Boreas dataset)
ros2 run steam_icp steam_icp --ros-args --params-file ${WORKING_DIR}/steam_icp/config/boreas_velodyne_steamlio_config.yaml
# this would run STEAM-LO from our "continuous-time radar-inertial..." paper (on the Boreas dataset)
ros2 run steam_icp steam_icp --ros-args --params-file ${WORKING_DIR}/steam_icp/config/boreas_velodyne_steamlo_config.yaml
# this would run STEAM-RIO from our "continuous-time radar-inertial..." paper (on the Boreas dataset)
ros2 run steam_icp steam_icp --ros-args --params-file ${WORKING_DIR}/steam_icp/config/boreas_navtech_paper_results_config.yaml
# after setting use_imu=False in the yaml file, this would run STEAM-RO from our "continuous-time radar-inertial..." paper (on the Boreas dataset)
ros2 run steam_icp steam_icp --ros-args --params-file ${WORKING_DIR}/steam_icp/config/boreas_navtech_paper_results_config.yaml
# this would run STEAM-RIO++ from our "continuous-time radar-inertial..." paper (on the Boreas dataset)
ros2 run steam_icp steam_icp --ros-args --params-file ${WORKING_DIR}/steam_icp/config/boreas_navtech_steamro_config.yaml
# this would run STEAM-LO from our "continuous-time radar-inertial..." paper (on the KITTI-raw dataset)
ros2 run steam_icp steam_icp --ros-args --params-file ${WORKING_DIR}/steam_icp/config/kitti_raw_steamlo_config.yaml
# this would run STEAM-LIO from our "continuous-time radar-inertial..." paper (on the KITTI-raw dataset)
ros2 run steam_icp steam_icp --ros-args --params-file ${WORKING_DIR}/steam_icp/config/kitti_raw_steamlio_config.yaml
# this would run STEAM-LO from our "continuous-time radar-inertial..." paper (on the Newer College dataset)
ros2 run steam_icp steam_icp --ros-args --params-file ${WORKING_DIR}/steam_icp/config/ncd_steamlo_config.yaml
# after setting use_imu=True, use_accel=True, qc_diag = [500., 500., 500., 50., 50., 50.] in the yaml file, this would run STEAM-LIO from our "continuous-time radar-inertial..." paper (on the Newer College dataset)
ros2 run steam_icp steam_icp --ros-args --params-file ${WORKING_DIR}/steam_icp/config/ncd_steamlo_config.yaml
# after setting use_imu=True, use_accel=False, qc_diag = [500., 500., 500., 50., 50., 50.] in the yaml file, this would run STEAM-LO+Gyro from our "continuous-time radar-inertial..." paper (on the Newer College dataset)
ros2 run steam_icp steam_icp --ros-args --params-file ${WORKING_DIR}/steam_icp/config/ncd_steamlo_config.yaml

### IMU as input vs. as measurement paper:
# To run the Singer prior as lidar-inertial odometry on a simulated dataset
ros2 run steam_icp steam_icp --ros-args --params-file ${WORKING_DIR}/steam_icp/config/boreas_simulation_singer_config.yaml
# after setting use_accel=False, this would run Singer prior as lidar + Gyro only on a simulated dataset
ros2 run steam_icp steam_icp --ros-args --params-file ${WORKING_DIR}/steam_icp/config/boreas_simulation_singer_config.yaml
# after setting use_imu=False, qc_diag=[5000.,5000.,5000.,500.,500.,500.], this would run Singer prior as lidar-only on a simulated dataset
ros2 run steam_icp steam_icp --ros-args --params-file ${WORKING_DIR}/steam_icp/config/boreas_simulation_singer_config.yaml
# To run the Singer prior as lidar-inertial odometry on the Newer College dataset
ros2 run steam_icp steam_icp --ros-args --params-file ${WORKING_DIR}/steam_icp/config/ncd_singer_config.yaml
# after setting use_accel=False, to run the Singer prior as lidar + gyro only on the Newer College dataset
ros2 run steam_icp steam_icp --ros-args --params-file ${WORKING_DIR}/steam_icp/config/ncd_singer_config.yaml
# after setting use_imu=False, to run the Singer prior as lidar-only odometry on the Newer College Dataset, use:
ros2 run steam_icp steam_icp --ros-args --params-file ${WORKING_DIR}/steam_icp/config/ncd_singer_config.yaml


# (optional) rviz for visualization - run following commands in another terminal
source /opt/ros/galactic/setup.bash
ros2 run rviz2 rviz2 -d ${WORKING_DIR}/steam_icp/rviz/steam_icp.rviz # launch rviz
```

Running the lidar-inertial simulation:
```
ros2 run steam_icp simulation --ros-args --params-file ${WORKING_DIR}/steam_icp/simulation/sim.yaml
```

Note that some of the code for generating maps and plots for the "continuous-time radar-inertial..." paper is located on the `map_and_plots` branch.

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
