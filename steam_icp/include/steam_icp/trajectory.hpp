#pragma once

#include "steam.hpp"

#include "steam_icp/point.hpp"

namespace steam_icp {

using ArrayMatrix4d = std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>;
using ArrayPoses = ArrayMatrix4d;

struct TrajectoryFrame {
  TrajectoryFrame() = default;

  Eigen::Matrix4d getBeginPose() const {
    Eigen::Matrix4d begin_pose = Eigen::Matrix4d::Identity();
    begin_pose.block<3, 3>(0, 0) = begin_R;
    begin_pose.block<3, 1>(0, 3) = begin_t;
    return begin_pose;
  }

  Eigen::Matrix4d getMidPose() const {
    Eigen::Matrix4d mid_pose = Eigen::Matrix4d::Identity();
    auto q_begin = Eigen::Quaterniond(begin_R);
    auto q_end = Eigen::Quaterniond(end_R);
    Eigen::Vector3d t_begin = begin_t;
    Eigen::Vector3d t_end = end_t;
    Eigen::Quaterniond q = q_begin.slerp(0.5, q_end);
    q.normalize();
    mid_pose.block<3, 3>(0, 0) = q.toRotationMatrix();
    mid_pose.block<3, 1>(0, 3) = 0.5 * t_begin + 0.5 * t_end;
    return mid_pose;
  }

  double begin_timestamp = 0.0;
  double end_timestamp = 1.0;

  Eigen::Matrix3d begin_R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d begin_t = Eigen::Vector3d::Zero();
  Eigen::Matrix3d end_R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d end_t = Eigen::Vector3d::Zero();

  Eigen::Matrix<double, 6, 6> end_T_rm_cov = Eigen::Matrix<double, 6, 6>::Identity();
  Eigen::Matrix<double, 6, 6> end_w_mr_inr_cov = Eigen::Matrix<double, 6, 6>::Identity();
  Eigen::Matrix<double, 12, 12> end_state_cov = Eigen::Matrix<double, 12, 12>::Identity();

  std::vector<Point3D> points;
};

using Trajectory = std::vector<TrajectoryFrame>;

}  // namespace steam_icp