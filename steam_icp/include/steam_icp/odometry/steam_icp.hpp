#pragma once

#include <fstream>

#include "steam.hpp"
#include "steam_icp/odometry.hpp"

namespace steam_icp {

class SteamOdometry : public Odometry {
 public:
  enum class STEAM_LOSS_FUNC { L2, DCS, CAUCHY, GM };

  struct Options : public Odometry::Options {
    // sensor vehicle transformation
    Eigen::Matrix<double, 4, 4> T_sr = Eigen::Matrix<double, 4, 4>::Identity();
    // trajectory
    Eigen::Matrix<double, 6, 1> qc_diag = Eigen::Matrix<double, 6, 1>::Ones();
    int num_extra_states = 0;
    //
    bool add_prev_state = false;
    int num_extra_prev_states = 0;
    bool lock_prev_pose = false;
    bool lock_prev_vel = false;
    bool prev_pose_as_prior = false;
    bool prev_vel_as_prior = false;
    //
    int no_prev_state_iters = 0;
    bool association_after_adding_prev_state = true;
    // velocity prior (no side slipping)
    bool use_vp = false;
    Eigen::Matrix<double, 6, 6> vp_cov = Eigen::Matrix<double, 6, 6>::Identity();
    // p2p
    double power_planarity = 2.0;
    int p2p_initial_iters = 0;
    double p2p_initial_max_dist = 0.3;
    double p2p_refined_max_dist = 0.3;
    STEAM_LOSS_FUNC p2p_loss_func = STEAM_LOSS_FUNC::L2;
    double p2p_loss_sigma = 1.0;
    // radial velocity
    bool use_rv = false;
    bool merge_p2p_rv = false;
    double rv_max_error = 2.0;
    STEAM_LOSS_FUNC rv_loss_func = STEAM_LOSS_FUNC::GM;
    double rv_cov_inv = 1.0;
    double rv_loss_threshold = 1.0;
    // optimization
    bool verbose = false;
    int max_iterations = 1;
    unsigned int num_threads = 1;

    //
    int delay_adding_points = 1;
    bool use_final_state_value = false;
  };

  SteamOdometry(const Options &options);
  ~SteamOdometry();

  Trajectory trajectory() override;

  RegistrationSummary registerFrame(const std::vector<Point3D> &frame) override;

 private:
  void initializeTimestamp(int index_frame, const std::vector<Point3D> &const_frame);
  void initializeMotion(int index_frame);
  std::vector<Point3D> initializeFrame(int index_frame, const std::vector<Point3D> &const_frame);
  void updateMap(int index_frame, int update_frame);
  bool icp(int index_frame, std::vector<Point3D> &keypoints);

 private:
  const Options options_;

  // steam variables
  steam::se3::SE3StateVar::Ptr T_sr_var_ = nullptr;  // robot to sensor transformation as a steam variable

  // trajectory variables
  struct TrajectoryVar {
    TrajectoryVar(const steam::traj::Time &t, const steam::se3::SE3StateVar::Ptr &T,
                  const steam::vspace::VSpaceStateVar<6>::Ptr &w)
        : time(t), T_rm(T), w_mr_inr(w) {}
    steam::traj::Time time;
    steam::se3::SE3StateVar::Ptr T_rm;
    steam::vspace::VSpaceStateVar<6>::Ptr w_mr_inr;
  };
  std::vector<TrajectoryVar> trajectory_vars_;
  size_t to_marginalize_ = 0;

  steam::SlidingWindowFilter::Ptr sliding_window_filter_;

  STEAM_ICP_REGISTER_ODOMETRY("STEAM", SteamOdometry);
};

}  // namespace steam_icp