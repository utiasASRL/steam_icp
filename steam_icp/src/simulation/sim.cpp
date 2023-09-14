#include <filesystem>
namespace fs = std::filesystem;

#include "glog/logging.h"

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "tf2/convert.h"
#include "tf2_eigen/tf2_eigen.h"
#include "tf2_ros/static_transform_broadcaster.h"
#include "tf2_ros/transform_broadcaster.h"

#include <Eigen/Core>
#include <iostream>

struct SimulationOptions {
  std::string output_dir = "/sim_output";  // output path (relative or absolute) to save simulation data
  Eigen::Matrix4d T_sr = Eigen::Matrix4d::Identity();
  double imu_rate = 200.0;
  double dt_imu = 0.0025;  // offset bewteen first imu meas and first lidar meas
  double min_dist_sensor_center = 0.1;
  double max_dist_sensor_center = 200.0;
  bool noisy_measurements = false;
  // approximate from spec sheet of velodyne
  double lidar_range_std = 0.02;
  // approximated from actual applanix data
  // note: these are covariances
  Eigen::Matrix<double, 3, 1> r_accel = Eigen::Matrix<double, 3, 1>::Ones();
  Eigen::Matrix<double, 3, 1> r_gyro = Eigen::Matrix<double, 3, 1>::Ones();
  double gravity = -9.8042;
  // todo: simulate bias P0 and Qc
  double p0_bias = 0.01;
  double q_bias = 0.01;
  // learned from Boreas data
  Eigen::Matrix<double, 6, 1> qc_diag = Eigen::Matrix<double, 6, 1>::Ones();
  Eigen::Matrix<double, 6, 1> ad_diag = Eigen::Matrix<double, 6, 1>::Ones();
  Eigen::Matrix<double, 18, 1> x0 = Eigen::Matrix<double, 18, 1>::Zero();
};

#define ROS2_PARAM_NO_LOG(node, receiver, prefix, param, type) \
  receiver = node->declare_parameter<type>(prefix + #param, receiver);
#define ROS2_PARAM(node, receiver, prefix, param, type)   \
  ROS2_PARAM_NO_LOG(node, receiver, prefix, param, type); \
  LOG(WARNING) << "Parameter " << prefix + #param << " = " << receiver << std::endl;
#define ROS2_PARAM_CLAUSE(node, config, prefix, param, type)                   \
  config.param = node->declare_parameter<type>(prefix + #param, config.param); \
  LOG(WARNING) << "Parameter " << prefix + #param << " = " << config.param << std::endl;

SimulationOptions loadOptions(const rclcpp::Node::SharedPtr &node) {
  SimulationOptions options;
  std::string prefix = "";
  ROS2_PARAM_CLAUSE(node, options, prefix, output_dir, std::string);
  if (!options.output_dir.empty() && options.output_dir[options.output_dir.size() - 1] != '/')
    options.output_dir += '/';
  ROS2_PARAM_CLAUSE(node, options, prefix, imu_rate, double);
  ROS2_PARAM_CLAUSE(node, options, prefix, dt_imu, double);
  ROS2_PARAM_CLAUSE(node, options, prefix, min_dist_sensor_center, double);
  ROS2_PARAM_CLAUSE(node, options, prefix, max_dist_sensor_center, double);
  ROS2_PARAM_CLAUSE(node, options, prefix, noisy_measurements, bool);
  ROS2_PARAM_CLAUSE(node, options, prefix, lidar_range_std, double);
  ROS2_PARAM_CLAUSE(node, options, prefix, gravity, double);
  ROS2_PARAM_CLAUSE(node, options, prefix, p0_bias, double);
  ROS2_PARAM_CLAUSE(node, options, prefix, q_bias, double);

  std::vector<double> r_accel;
  ROS2_PARAM_NO_LOG(node, r_accel, prefix, r_accel, std::vector<double>);
  if ((r_accel.size() != 3) && (r_accel.size() != 0))
    throw std::invalid_argument{"r_accel malformed. Must be 3 elements!"};
  if (r_accel.size() == 3) options.r_accel << r_accel[0], r_accel[1], r_accel[2];
  LOG(WARNING) << "Parameter " << prefix + "r_accel"
               << " = " << options.r_accel.transpose() << std::endl;

  std::vector<double> r_gyro;
  ROS2_PARAM_NO_LOG(node, r_gyro, prefix, r_gyro, std::vector<double>);
  if ((r_gyro.size() != 3) && (r_gyro.size() != 0))
    throw std::invalid_argument{"r_gyro malformed. Must be 3 elements!"};
  if (r_gyro.size() == 3) options.r_gyro << r_gyro[0], r_gyro[1], r_gyro[2];
  LOG(WARNING) << "Parameter " << prefix + "r_gyro"
               << " = " << options.r_gyro.transpose() << std::endl;

  std::vector<double> T_sr;
  ROS2_PARAM_NO_LOG(node, T_sr, prefix, T_sr, std::vector<double>);
  if ((T_sr.size() != 16) && (T_sr.size() != 0)) throw std::invalid_argument{"T_sr malformed. Must be 16 elements!"};
  if (T_sr.size() == 16) {
    int k = 0;
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        options.T_sr(i, j) = T_sr[k];
        k++;
      }
    }
  }
  LOG(WARNING) << "(YAML)Parameter " << prefix + "T_sr"
               << " = " << std::endl
               << options.T_sr << std::endl;

  std::vector<double> x0;
  ROS2_PARAM_NO_LOG(node, x0, prefix, x0, std::vector<double>);
  if ((x0.size() != 18) && (x0.size() != 0)) throw std::invalid_argument{"x0 malformed. Must be 18 elements!"};
  if (x0.size() == 18)
    options.x0 << x0[0], x0[1], x0[2], x0[3], x0[4], x0[5], x0[6], x0[7], x0[8], x0[9], x0[10], x0[11], x0[12], x0[13],
        x0[14], x0[15], x0[16], x0[17];
  LOG(WARNING) << "Parameter " << prefix + "x0"
               << " = " << options.x0.transpose() << std::endl;

  return options;
}

// int main(int argc, char **argv) {
//   // todo: parameters for singer prior
//   // todo: sample from multidimensional gaussian
//   // todo: create sparse matrix for P_check_inv
//   // todo: sample Gaussian noise vector Don't allow samples outside of 4-sigma (sample from truncated Gaussian)
//   // todo: load VLS128 config parameters
//   // todo: load extrinsics from yaml files
//   // todo: create ordered list of timestamps required by IMU, lidar
//   // todo: step through simulation, generating pointclouds and IMU measurements
//   // todo: save pointclouds as .bin files and imu measurements in same formatted csv file
//   // todo: add options for adding Gaussian noise to the lidar and IMU measurements

//   // todo: x0

//   // Simulation Parameters (move to config file)

//   return 0;
// }