#pragma once

#include <fstream>

#include "steam.hpp"
#include "steam/evaluable/imu/bias_interpolator.hpp"
#include "steam/problem/cost_term/imu_super_cost_term.hpp"
#include "steam/problem/cost_term/p2p_doppler_const_acc_super_cost_term.hpp"
#include "steam/solver/gauss_newton_solver_nva.hpp"
#include "steam_icp/odometry.hpp"

namespace steam_icp {

class SteamRioOdometry : public Odometry {
 public:
  using Matrix18d = Eigen::Matrix<double, 18, 18>;

  enum class STEAM_LOSS_FUNC { L2, DCS, CAUCHY, GM, HUBER };

  struct Options : public Odometry::Options {
    // sensor vehicle transformation
    Eigen::Matrix<double, 4, 4> T_sr = Eigen::Matrix<double, 4, 4>::Identity();
    // trajectory
    Eigen::Matrix<double, 6, 1> qc_diag = Eigen::Matrix<double, 6, 1>::Ones();
    // p2p
    double p2p_max_dist = 0.5;
    STEAM_LOSS_FUNC p2p_loss_func = STEAM_LOSS_FUNC::CAUCHY;
    double p2p_loss_sigma = 0.1;
    // optimization
    bool verbose = false;
    int max_iterations = 5;
    unsigned int num_threads = 1;
    //
    int delay_adding_points = 4;
    bool use_final_state_value = false;
    // IMU
    double gravity = -9.8042;
    Eigen::Matrix<double, 3, 1> r_imu_acc = Eigen::Matrix<double, 3, 1>::Zero();
    Eigen::Matrix<double, 3, 1> r_imu_ang = Eigen::Matrix<double, 3, 1>::Zero();
    Eigen::Matrix<double, 3, 1> p0_bias_accel = Eigen::Matrix<double, 3, 1>::Zero();
    double pk_bias_accel = 0.0001;
    Eigen::Matrix<double, 3, 1> q_bias_accel = Eigen::Matrix<double, 3, 1>::Ones();
    double p0_bias_gyro = 0.0001;
    double pk_bias_gyro = 0.0001;
    double q_bias_gyro = 0.0001;
    bool use_imu = true;
    // T_mi:
    Eigen::Matrix<double, 6, 1> p0_pose = Eigen::Matrix<double, 6, 1>::Ones();
    Eigen::Matrix<double, 6, 1> p0_vel = Eigen::Matrix<double, 6, 1>::Ones();
    Eigen::Matrix<double, 6, 1> p0_accel = Eigen::Matrix<double, 6, 1>::Ones();
    bool use_bias_prior_after_init = false;
    std::string acc_loss_func = "CAUCHY";
    double acc_loss_sigma = 1.0;
    std::string gyro_loss_func = "L2";
    double gyro_loss_sigma = 1.0;
    // radar-only option
    double beta = 0.049;  // used for Doppler correction
    bool voxel_downsample = false;
  };

  SteamRioOdometry(const Options &options);
  ~SteamRioOdometry();

  Trajectory trajectory() override;

  RegistrationSummary registerFrame(const DataFrame &frame) override;

 private:
  void initializeTimestamp(int index_frame, const DataFrame &const_frame);
  void initializeMotion(int index_frame);
  std::vector<Point3D> initializeFrame(int index_frame, const std::vector<Point3D> &const_frame);
  Eigen::Matrix<double, 6, 1> initialize_gravity(const std::vector<steam::IMUData> &imu_data_vec);
  void updateMap(int index_frame, int update_frame);
  bool icp(int index_frame, std::vector<Point3D> &keypoints, const std::vector<steam::IMUData> &imu_data_vec);

 private:
  const Options options_;

  // steam variables
  steam::se3::SE3StateVar::Ptr T_sr_var_ = nullptr;  // robot to sensor transformation as a steam variable
  steam::se3::SE3StateVar::Ptr T_mi_var_ = nullptr;  // transform from gravity-down to map frame

  // trajectory variables
  struct TrajectoryVar {
    TrajectoryVar(const steam::traj::Time &t, const steam::se3::SE3StateVar::Ptr &T,
                  const steam::vspace::VSpaceStateVar<6>::Ptr &w, const steam::vspace::VSpaceStateVar<6>::Ptr &dw,
                  const steam::vspace::VSpaceStateVar<6>::Ptr &b)
        : time(t), T_rm(T), w_mr_inr(w), dw_mr_inr(dw), imu_biases(b) {}
    steam::traj::Time time;
    steam::se3::SE3StateVar::Ptr T_rm;
    steam::vspace::VSpaceStateVar<6>::Ptr w_mr_inr;
    steam::vspace::VSpaceStateVar<6>::Ptr dw_mr_inr;
    steam::vspace::VSpaceStateVar<6>::Ptr imu_biases;
  };
  std::vector<TrajectoryVar> trajectory_vars_;
  size_t to_marginalize_ = 0;

  std::map<double, std::pair<Matrix18d, Matrix18d>> interp_mats_;

  steam::SlidingWindowFilter::Ptr sliding_window_filter_;

  STEAM_ICP_REGISTER_ODOMETRY("STEAMRIO", SteamRioOdometry);
};

}  // namespace steam_icp