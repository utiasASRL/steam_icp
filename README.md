# steam_icp

Build docker image

```bash
docker build -t steam_icp \
  --build-arg USERID=$(id -u) \
  --build-arg GROUPID=$(id -g) \
  --build-arg USERNAME=$(whoami) \
  --build-arg HOMEDIR=${HOME} .
```

Run docker image

```bash
docker run -it --name steam_icp \
  --privileged \
  --network=host \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ${HOME}:${HOME}:rw \
  -v ${HOME}/ASRL:${HOME}/ASRL:rw \
  -v ${HOME}/ASRL/data/boreas:${HOME}/ASRL/data/boreas \
  -v /media/yuchen/T7/ASRL/data/dicp_corrected:${HOME}/ASRL/data/dicp_corrected \
  -v /media/yuchen/T7/ASRL/data/dicp:${HOME}/ASRL/data/dicp \
  -v /media/yuchen/T7/ASRL/data/kitti_raw:${HOME}/ASRL/data/kitti_raw \
  -v /media/yuchen/T7/ASRL/data/kitti_360:${HOME}/ASRL/data/kitti_360 \
  steam_icp
```

Build STEAM ICP

```bash
# steam icp
bash build.sh

# python venv for evaluation
virtualenv venv
source venv/bin/activate
pip install -e pylgmath
pip install -e pysteam
```

Run STEAM ICP

```bash
# ROOT directory of this repository
WORKING_DIR=$(pwd)

## First launch RViz for visualization (in a separate terminal)
source /opt/ros/galactic/setup.bash
ros2 run rviz2 rviz2 -d ${WORKING_DIR}/steam_icp/rviz/steam_icp.rviz # launch rviz

## Add libraries
EXTERNAL_ROOT=${WORKING_DIR}/cmake-build-Release/external/install/Release
LGMATH_ROOT=${WORKING_DIR}/cmake-build-Release/lgmath/install/Release
STEAM_ROOT=${WORKING_DIR}/cmake-build-Release/steam/install/Release
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${EXTERNAL_ROOT}/Ceres/lib:${EXTERNAL_ROOT}/glog/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${LGMATH_ROOT}/lib:${STEAM_ROOT}/lib

## Run STEAM ICP on any dataset - change config file accordingly
source ${WORKING_DIR}/steam_icp/install/setup.bash
ros2 run steam_icp steam_icp --ros-args --params-file ${WORKING_DIR}/steam_icp/config/aeva_steam_config.yaml
```
