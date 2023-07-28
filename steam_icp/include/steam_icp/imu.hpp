#pragma once

#include <Eigen/Dense>

struct IMUData {
  uint64_t timestamp = 0;
  Eigen::Vector3d ang_vel = Eigen::Vector3d::Zero();
  Eigen::Vector3d lin_acc = Eigen::Vector3d::Zero();
};