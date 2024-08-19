#pragma once

#include <fstream>

#include "steam.hpp"
#include "steam/problem/cost_term/gyro_super_cost_term.hpp"
#include "steam/problem/cost_term/imu_super_cost_term.hpp"
#include "steam/problem/cost_term/p2p_const_vel_super_cost_term.hpp"
#include "steam/problem/cost_term/preintegrated_accel_cost_term.hpp"
#include "steam_icp/odometry.hpp"

namespace steam_icp {

class SteamLoCVOdometry : public Odometry {
 public:
  using Matrix12d = Eigen::Matrix<double, 12, 12>;
  enum class STEAM_LOSS_FUNC { L2, DCS, CAUCHY, GM };

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
    // IMU
    bool use_imu = false;
    bool use_accel = false;
    Eigen::Matrix<double, 3, 1> r_imu_ang = Eigen::Matrix<double, 3, 1>::Zero();
    double p0_bias_gyro = 0.0001;
    double q_bias_gyro = 0.0001;
    // Accelerometer
    double gravity = -9.8042;
    Eigen::Matrix<double, 3, 1> r_imu_acc = Eigen::Matrix<double, 3, 1>::Ones();
    Eigen::Matrix<double, 3, 1> p0_bias_accel = Eigen::Matrix<double, 3, 1>::Ones();
    Eigen::Matrix<double, 3, 1> q_bias_accel = Eigen::Matrix<double, 3, 1>::Ones();
    bool T_mi_init_only = true;
    Eigen::Matrix<double, 6, 1> qg_diag = Eigen::Matrix<double, 6, 1>::Ones();
    Eigen::Matrix<double, 6, 1> T_mi_init_cov = Eigen::Matrix<double, 6, 1>::Ones();
    std::string acc_loss_func = "L2";
    double acc_loss_sigma = 1.0;
    //
    bool filter_lifetimes = false;
    bool swf_inside_icp_at_begin = true;
    bool break_icp_early = false;
    bool use_elastic_initialization = false;
    double keyframe_translation_threshold_m = 0.0;
    double keyframe_rotation_threshold_deg = 0.0;
    bool use_line_search = false;
  };

  SteamLoCVOdometry(const Options &options);
  ~SteamLoCVOdometry();

  Trajectory trajectory() override;

  RegistrationSummary registerFrame(const DataFrame &frame) override;

 private:
  void initializeTimestamp(int index_frame, const DataFrame &const_frame);
  Eigen::Matrix<double, 6, 1> initialize_gravity(const std::vector<steam::IMUData> &imu_data_vec);
  void initializeMotion(int index_frame);
  std::vector<Point3D> initializeFrame(int index_frame, const std::vector<Point3D> &const_frame);
  void updateMap(int index_frame, int update_frame);
  bool icp(int index_frame, std::vector<Point3D> &keypoints, const std::vector<steam::IMUData> &imu_data_vec);

 private:
  const Options options_;

  // steam variables
  steam::se3::SE3StateVar::Ptr T_sr_var_ = nullptr;  // robot to sensor transformation as a steam variable

  // trajectory variables
  struct TrajectoryVar {
    TrajectoryVar(const steam::traj::Time &t, const steam::se3::SE3StateVar::Ptr &T,
                  const steam::vspace::VSpaceStateVar<6>::Ptr &w, const steam::vspace::VSpaceStateVar<6>::Ptr &b,
                  const steam::se3::SE3StateVar::Ptr &T_m_i)
        : time(t), T_rm(T), w_mr_inr(w), imu_biases(b), T_mi(T_m_i) {}
    steam::traj::Time time;
    steam::se3::SE3StateVar::Ptr T_rm;
    steam::vspace::VSpaceStateVar<6>::Ptr w_mr_inr;
    steam::vspace::VSpaceStateVar<6>::Ptr imu_biases;
    steam::se3::SE3StateVar::Ptr T_mi;
  };
  std::vector<TrajectoryVar> trajectory_vars_;
  size_t to_marginalize_ = 0;

  std::map<double, std::pair<Matrix12d, Matrix12d>> interp_mats_;

  steam::SlidingWindowFilter::Ptr sliding_window_filter_;

  Eigen::Vector3d t_prev_ = Eigen::Vector3d::Zero();
  Eigen::Matrix3d r_prev_ = Eigen::Matrix3d::Identity();

  STEAM_ICP_REGISTER_ODOMETRY("STEAMLOCV", SteamLoCVOdometry);
};

}  // namespace steam_icp