#pragma once

#include <Eigen/Core>

#include "steam_icp/dataset.hpp"

namespace steam_icp {

Sequence::SeqError evaluateOdometry(const std::string &filename, const ArrayPoses &poses_gt,
                                    const ArrayPoses &poses_estimated, int step_size = 10);

}  // namespace steam_icp