#!/bin/bash

BUILD_TYPE="Release"
GENERATOR="Unix Makefiles"

SRC_DIR=$(pwd)
EXT_SRC_DIR="${SRC_DIR}/external"
LGMATH_SRC_DIR="${SRC_DIR}/lgmath"
STEAM_SRC_DIR="${SRC_DIR}/steam"
STEAM_ICP_SRC_DIR="${SRC_DIR}/steam_icp"

BUILD_DIR="${SRC_DIR}/cmake-build-${BUILD_TYPE}"
EXT_BUILD_DIR=$BUILD_DIR/external
LGMATH_BUILD_DIR=$BUILD_DIR/lgmath
STEAM_BUILD_DIR=$BUILD_DIR/steam

mkdir -p $BUILD_DIR
mkdir -p $EXT_BUILD_DIR
mkdir -p $LGMATH_BUILD_DIR
mkdir -p $STEAM_BUILD_DIR

check_status_code() {
	if [ $1 -ne 0 ]; then
		echo "[STEAM_ICP] Failure. Exiting."
		exit 1
	fi
}

echo "[STEAM_ICP] -- [EXTERNAL DEPENDENCIES] -- Generating the cmake project"
cd ${EXT_BUILD_DIR}
cmake -G "$GENERATOR" -S $EXT_SRC_DIR -DCMAKE_BUILD_TYPE=${BUILD_TYPE}
check_status_code $?

echo "[STEAM_ICP] -- [EXTERNAL DEPENDENCIES] -- building CMake Project"
cmake --build . --config $BUILD_TYPE
check_status_code $?

echo "[STEAM_ICP] -- [LGMATH] -- Generating the cmake project"
cd ${LGMATH_BUILD_DIR}
cmake -G "$GENERATOR" -S $LGMATH_SRC_DIR \
	-DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
	-DUSE_AMENT=OFF \
	-DEigen3_DIR=${EXT_BUILD_DIR}/install/${BUILD_TYPE}/Eigen3/share/eigen3/cmake \
	-DCMAKE_INSTALL_PREFIX=${LGMATH_BUILD_DIR}/install/${BUILD_TYPE}
check_status_code $?

echo "[STEAM_ICP] -- [LGMATH] -- building CMake Project"
cmake --build . --config $BUILD_TYPE --target install --parallel 6
check_status_code $?

echo "[STEAM_ICP] -- [STEAM] -- Generating the cmake project"
cd ${STEAM_BUILD_DIR}
cmake -G "$GENERATOR" -S $STEAM_SRC_DIR \
	-DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
	-DUSE_AMENT=OFF \
  -DBUILD_TESTING=ON \
	-DEigen3_DIR=${EXT_BUILD_DIR}/install/${BUILD_TYPE}/Eigen3/share/eigen3/cmake \
	-Dlgmath_DIR=${LGMATH_BUILD_DIR}/install/${BUILD_TYPE}/lib/cmake/lgmath \
	-DCMAKE_INSTALL_PREFIX=${STEAM_BUILD_DIR}/install/${BUILD_TYPE}
check_status_code $?
# ctest
# check_status_code $?

echo "[STEAM_ICP] -- [STEAM] -- building CMake Project"
cmake --build . --config $BUILD_TYPE --target install --parallel 6
check_status_code $?

echo "[STEAM_ICP] -- [STEAM_ICP] -- building steam icp package"
cd ${STEAM_ICP_SRC_DIR}
source /opt/ros/${ROS_DISTRO}/setup.bash
colcon build --symlink-install \
	--packages-select steam_icp \
	--cmake-args \
	-DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
	-Dlgmath_DIR=${LGMATH_BUILD_DIR}/install/${BUILD_TYPE}/lib/cmake/lgmath \
	-Dsteam_DIR=${STEAM_BUILD_DIR}/install/${BUILD_TYPE}/lib/cmake/steam \
	-DEigen3_DIR=${EXT_BUILD_DIR}/install/${BUILD_TYPE}/Eigen3/share/eigen3/cmake \
	-Dglog_DIR=${EXT_BUILD_DIR}/install/${BUILD_TYPE}/glog/lib/cmake/glog \
	-Dtessil_DIR=${EXT_BUILD_DIR}/install/${BUILD_TYPE}/tessil/share/cmake/tsl-robin-map \
	-DCeres_DIR=${EXT_BUILD_DIR}/install/${BUILD_TYPE}/Ceres/lib/cmake/Ceres
