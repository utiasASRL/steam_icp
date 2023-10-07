#pragma once

#include <Eigen/Dense>

namespace steam_icp {

struct PoseData {
  double timestamp = 0;
  Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
};

}  // namespace steam_icp