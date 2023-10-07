#pragma once

#include <Eigen/Dense>

namespace steam_icp {

struct IMUData {
  double timestamp = 0;
  Eigen::Vector3d ang_vel = Eigen::Vector3d::Zero();
  Eigen::Vector3d lin_acc = Eigen::Vector3d::Zero();
};

}  // namespace steam_icp