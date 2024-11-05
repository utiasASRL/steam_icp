#pragma once

#include <fstream>

#include "steam.hpp"
#include "steam/problem/cost_term/gyro_super_cost_term.hpp"
#include "steam/problem/cost_term/imu_super_cost_term.hpp"
#include "steam/problem/cost_term/preintegrated_imu_cost_term.hpp"
#include "steam/problem/cost_term/p2p_const_vel_super_cost_term.hpp"
#include "steam/problem/cost_term/preintegrated_accel_cost_term.hpp"
#include "steam/problem/cost_term/p2p_global_perturb_super_cost_term.hpp"
#include "steam_icp/odometry.hpp"

namespace steam_icp {

class DiscreteLIOOdometry : public Odometry {
 public:
  enum class STEAM_LOSS_FUNC { L2, DCS, CAUCHY, GM };

  struct Options : public Odometry::Options {
    // sensor vehicle transformation
    Eigen::Matrix<double, 4, 4> T_sr = Eigen::Matrix<double, 4, 4>::Identity();
    // p2p
    double power_planarity = 2.0;
    double p2p_max_dist = 0.5;
    STEAM_LOSS_FUNC p2p_loss_func = STEAM_LOSS_FUNC::L2;
    double p2p_loss_sigma = 0.1;
    // optimization
    bool verbose = false;
    int max_iterations = 5;
    unsigned int num_threads = 1;
    //
    int delay_adding_points = 4;
    // Gyro
    Eigen::Matrix<double, 3, 1> r_imu_ang = Eigen::Matrix<double, 3, 1>::Zero();
    double q_bias_gyro = 0.0001;
    // Accelerometer
    double gravity = -9.8042;
    Eigen::Matrix<double, 3, 1> r_imu_acc = Eigen::Matrix<double, 3, 1>::Ones();
    Eigen::Matrix<double, 3, 1> q_bias_accel = Eigen::Matrix<double, 3, 1>::Ones();
    Eigen::Matrix<double, 6, 1> T_mi_init_cov = Eigen::Matrix<double, 6, 1>::Ones();
    // IMU
    double imu_loss_sigma = 1.0;
    std::string imu_loss_func = "L2"; 
    // initial cov
    Eigen::Matrix<double, 3, 1> p0_bias_accel = Eigen::Matrix<double, 3, 1>::Ones();
    double p0_bias_gyro = 0.0001;
    Eigen::Matrix<double, 3, 1> p0_vel = Eigen::Matrix<double, 3, 1>::Ones();
    Eigen::Matrix<double, 6, 1> p0_pose = Eigen::Matrix<double, 6, 1>::Ones();

    double pk_bias_accel = 0.0001;
    double pk_bias_gyro = 0.0001;
    bool use_bias_prior_after_init = false;
    //
    bool filter_lifetimes = false;
    bool swf_inside_icp_at_begin = true;
    bool break_icp_early = false;
    double keyframe_translation_threshold_m = 0.0;
    double keyframe_rotation_threshold_deg = 0.0;
    bool use_line_search = false;
    Eigen::Matrix<double, 6, 1> r_pose = Eigen::Matrix<double, 6, 1>::Zero();
  };

  DiscreteLIOOdometry(const Options &options);
  ~DiscreteLIOOdometry();

  Trajectory trajectory() override;

  RegistrationSummary registerFrame(const DataFrame &frame) override;

 private:
  void initializeTimestamp(int index_frame, const DataFrame &const_frame);
  Eigen::Matrix<double, 6, 1> initialize_gravity(const std::vector<steam::IMUData> &imu_data_vec);
  void initializeMotion(int index_frame);
  std::vector<Point3D> initializeFrame(int index_frame, const std::vector<Point3D> &const_frame);
  void updateMap(int index_frame, int update_frame);
  bool icp(int index_frame, std::vector<Point3D> &keypoints, const std::vector<steam::IMUData> &imu_data_vec, const std::vector<PoseData> &pose_data_vec);
  void transform_keypoints(
    const std::vector<double> &unique_point_times,
    std::vector<Point3D> &keypoints,
    const std::vector<steam::IMUData> &imu_data_vec,
    const double curr_time,
    const uint trajectory_vars_index,
    bool undistort_only,
    Eigen::Matrix4d T_rs
    );

 private:
  const Options options_;

  // trajectory variables
  struct TrajectoryVar {
    TrajectoryVar(const steam::traj::Time &t, const steam::se3::SE3StateVarGlobalPerturb::Ptr &T,
                  const steam::vspace::PreIntVelocityStateVar<3>::Ptr &v, const steam::vspace::VSpaceStateVar<6>::Ptr &b)
        : time(t), T_mr(T), v_rm_inm(v), imu_biases(b) {}
    steam::traj::Time time;
    steam::se3::SE3StateVarGlobalPerturb::Ptr T_mr;
    steam::vspace::PreIntVelocityStateVar<3>::Ptr v_rm_inm;
    steam::vspace::VSpaceStateVar<6>::Ptr imu_biases;
  };
  std::vector<TrajectoryVar> trajectory_vars_;
  size_t to_marginalize_ = 0;


  steam::SlidingWindowFilter::Ptr sliding_window_filter_;
  std::vector<steam::IMUData> prev_imu_data_vec_;

  Eigen::Vector3d t_prev_ = Eigen::Vector3d::Zero();
  Eigen::Matrix3d r_prev_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d gravity_ = {0, 0, -9.8042};

  STEAM_ICP_REGISTER_ODOMETRY("DiscreteLIO", DiscreteLIOOdometry);
};

}  // namespace steam_icp