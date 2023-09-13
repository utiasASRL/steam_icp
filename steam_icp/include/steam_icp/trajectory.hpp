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
    if (mid_pose_init) {
      return mid_pose_;
    } else {
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
  }

  // if we want to evaluate the trajectory at a specific timestamp other than
  // the middle of (begin_timestamp, end_timestamp), use this function.
  void setEvalTime(double eval_timestamp) {
    eval_timestamp_ = eval_timestamp;
    eval_time_init = true;
  }

  double getEvalTime() const {
    if (eval_time_init) {
      return eval_timestamp_;
    } else {
      return (begin_timestamp + end_timestamp) / 2.0;
    }
  }

  void setMidPose(Eigen::Matrix4d mid_pose) {
    mid_pose_ = mid_pose;
    mid_pose_init = true;
  }

  double begin_timestamp = 0.0;
  double end_timestamp = 1.0;
  bool eval_time_init = false;
  bool mid_pose_init = false;

  Eigen::Matrix3d begin_R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d begin_t = Eigen::Vector3d::Zero();
  Eigen::Matrix3d end_R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d end_t = Eigen::Vector3d::Zero();

  Eigen::Matrix<double, 6, 6> end_T_rm_cov = Eigen::Matrix<double, 6, 6>::Identity();
  Eigen::Matrix<double, 6, 6> end_w_mr_inr_cov = Eigen::Matrix<double, 6, 6>::Identity();
  Eigen::Matrix<double, 6, 6> end_dw_mr_inr_cov = Eigen::Matrix<double, 6, 6>::Identity();
  Eigen::Matrix<double, 18, 18> end_state_cov = Eigen::Matrix<double, 18, 18>::Identity();

  Eigen::Matrix<double, 6, 1> mid_w = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 1> mid_dw = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 1> mid_b = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 18, 18> mid_state_cov = Eigen::Matrix<double, 18, 18>::Identity();
  Eigen::Matrix4d mid_T_mi = Eigen::Matrix4d::Identity();

  std::vector<Point3D> points;

 private:
  Eigen::Matrix4d mid_pose_ = Eigen::Matrix4d::Identity();
  double eval_timestamp_ = 0.5;
};

using Trajectory = std::vector<TrajectoryFrame>;

}  // namespace steam_icp