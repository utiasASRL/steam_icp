#pragma once

#include <fstream>

#include "steam.hpp"
#include "steam_icp/odometry.hpp"

namespace steam_icp {

class SteamRoOdometry : public Odometry {
 public:
  enum class STEAM_LOSS_FUNC { L2, DCS, CAUCHY, GM, HUBER };

  struct Options : public Odometry::Options {
    // sensor vehicle transformation
    Eigen::Matrix<double, 4, 4> T_sr = Eigen::Matrix<double, 4, 4>::Identity();
    // trajectory
    Eigen::Matrix<double, 6, 1> qc_diag = Eigen::Matrix<double, 6, 1>::Ones();
    int num_extra_states = 0;
    // p2p
    double power_planarity = 2.0;
    double p2p_max_dist = 0.5;
    STEAM_LOSS_FUNC p2p_loss_func = STEAM_LOSS_FUNC::CAUCHY;
    double p2p_loss_sigma = 0.1;
    // radial velocity
    bool use_rv = false;
    bool merge_p2p_rv = false;
    double rv_max_error = 2.0;
    STEAM_LOSS_FUNC rv_loss_func = STEAM_LOSS_FUNC::CAUCHY;
    double rv_cov_inv = 0.1;
    double rv_loss_threshold = 0.05;
    // optimization
    bool verbose = false;
    int max_iterations = 5;
    unsigned int num_threads = 1;
    //
    int delay_adding_points = 4;
    bool use_final_state_value = false;
    // radar-only option
    double beta = 0.049;  // used for Doppler correction
    bool voxel_downsample = false;
  };

  SteamRoOdometry(const Options &options);
  ~SteamRoOdometry();

  Trajectory trajectory() override;

  RegistrationSummary registerFrame(const DataFrame &frame) override;

 private:
  void initializeTimestamp(int index_frame, const DataFrame &const_frame);
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

  STEAM_ICP_REGISTER_ODOMETRY("STEAMRO", SteamRoOdometry);
};

}  // namespace steam_icp