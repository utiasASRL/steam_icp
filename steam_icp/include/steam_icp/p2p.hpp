#pragma once

#include <Eigen/Dense>

namespace steam_icp {

struct P2PMatch {
  double timestamp = 0;
  Eigen::Vector3d reference = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal = Eigen::Vector3d::Ones();
  Eigen::Vector3d query = Eigen::Vector3d::Zero();
};

}  // namespace steam_icp