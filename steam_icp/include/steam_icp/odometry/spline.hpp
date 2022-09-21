#pragma once

#include <fstream>

#include "steam.hpp"
#include "steam_icp/odometry.hpp"

namespace steam_icp {

class SplineOdometry : public Odometry {
 public:
  enum class STEAM_LOSS_FUNC { L2, DCS, CAUCHY, GM };

  struct Options : public Odometry::Options {
    // sensor vehicle transformation
    Eigen::Matrix<double, 4, 4> T_sr = Eigen::Matrix<double, 4, 4>::Identity();
    // trajectory
    double knot_spacing = 0.1;
    double window_delay = 0.0;
    // regularization
    double c_cov = 1.0;
    // velocity prior (no side slipping)
    Eigen::Matrix<double, 6, 6> vp_cov = Eigen::Matrix<double, 6, 6>::Identity();
    double vp_spacing = 0.1;
    // radial velocity
    STEAM_LOSS_FUNC rv_loss_func = STEAM_LOSS_FUNC::L2;
    double rv_cov_inv = 1.0;
    double rv_loss_threshold = 1.0;
    // optimization
    bool verbose = false;
    int max_iterations = 1;
    unsigned int num_threads = 1;
  };

  SplineOdometry(const Options &options);
  ~SplineOdometry();

  Trajectory trajectory() override { return trajectory_; }

  RegistrationSummary registerFrame(const std::vector<Point3D> &frame) override;

 private:
  void initializeTimestamp(int index_frame, const std::vector<Point3D> &const_frame);
  std::vector<Point3D> initializeFrame(int index_frame, const std::vector<Point3D> &const_frame);
  bool estimateMotion(int index_frame, std::vector<Point3D> &keypoints);

 private:
  const Options options_;

  // steam variables
  steam::se3::SE3StateVar::Ptr T_sr_var_ = nullptr;  // robot to sensor transformation as a steam variable

  // trajectory variables
  struct TrajectoryVar {
    TrajectoryVar(const steam::traj::Time &t, const steam::vspace::VSpaceStateVar<6>::Ptr &w) : time(t), c(w) {}
    steam::traj::Time time;
    steam::vspace::VSpaceStateVar<6>::Ptr c;
  };
  std::vector<TrajectoryVar> trajectory_vars_;
  int curr_active_idx_ = 0;

  double curr_prior_time_ = 0.0;

  steam::traj::bspline::Interface::Ptr spline_trajectory_;
  steam::SlidingWindowFilter::Ptr sliding_window_filter_;

  STEAM_ICP_REGISTER_ODOMETRY("SPLINE", SplineOdometry);
};

}  // namespace steam_icp