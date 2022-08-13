#pragma once

#include <Eigen/Dense>

namespace steam_icp {

// A Point3D
struct Point3D {
  Eigen::Vector3d raw_pt;  // Raw point read from the sensor
  Eigen::Vector3d pt;      // Corrected point taking into account the motion of the sensor during frame acquisition
  double radial_velocity = 0.0;  // Radial velocity of the point
  double alpha_timestamp = 0.0;  // Relative timestamp in the frame in [0.0, 1.0]
  double timestamp = 0.0;        // The absolute timestamp (if applicable)
  int beam_id = -1;              // The beam id of the point
};

}  // namespace steam_icp
