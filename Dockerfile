FROM ubuntu:20.04

CMD ["/bin/bash"]

# Args for setting up non-root users, example command to use your own user:
# docker build -t ct_icp \
#   --build-arg USERID=$(id -u) \
#   --build-arg GROUPID=$(id -g) \
#   --build-arg USERNAME=$(whoami) \
#   --build-arg HOMEDIR=${HOME} .
ARG GROUPID=0
ARG USERID=0
ARG USERNAME=root
ARG HOMEDIR=/root

RUN if [ ${GROUPID} -ne 0 ]; then addgroup --gid ${GROUPID} ${USERNAME}; fi \
  && if [ ${USERID} -ne 0 ]; then adduser --disabled-password --gecos '' --uid ${USERID} --gid ${GROUPID} ${USERNAME}; fi

ENV DEBIAN_FRONTEND=noninteractive

## Switch to specified user to create directories
USER ${USERID}:${GROUPID}

## Switch to root to install dependencies
USER 0:0

## Dependencies
RUN apt update && apt upgrade -q -y
RUN apt update && apt install -q -y cmake git build-essential lsb-release curl gnupg2
RUN apt update && apt install -q -y libboost-all-dev libomp-dev
RUN apt update && apt install -q -y libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev
RUN apt update && apt install -q -y freeglut3-dev
RUN apt update && apt install -q -y python3 python3-distutils python3-pip
RUN pip3 install pybind11

## Install ROS2
# UTF-8
RUN apt install -q -y locales \
  && locale-gen en_US en_US.UTF-8 \
  && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
# Add ROS2 key and install from Debian packages
RUN apt install -q -y curl gnupg2 lsb-release
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key  -o /usr/share/keyrings/ros-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
  && apt update && apt install -q -y ros-galactic-desktop

## Install VTR specific ROS2 dependencies
RUN apt update && apt install -q -y \
  ros-galactic-xacro \
  ros-galactic-vision-opencv \
  ros-galactic-perception-pcl ros-galactic-pcl-ros

## Install misc dependencies
RUN apt update && apt install -q -y \
  tmux \
  libboost-all-dev libomp-dev \
  libpcl-dev \
  python3-colcon-common-extensions \
  virtualenv \
  texlive-latex-extra \
  texlive-fonts-recommended \
  cm-super \
  dvipng \
  clang-format

## Install python dependencies
# jupyter
RUN pip3 install pexpect ipympl
# evaluation
RUN pip3 install asrl-pylgmath asrl-pysteam

RUN apt update && apt install -q -y \
  libsuitesparse-dev \
  wget

## Switch to specified user
USER ${USERID}:${GROUPID}
