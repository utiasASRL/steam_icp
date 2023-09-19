#include <filesystem>
namespace fs = std::filesystem;

#include "glog/logging.h"

#include "nav_msgs/msg/odometry.hpp"
#include "pcl_conversions/pcl_conversions.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "tf2/convert.h"
#include "tf2_eigen/tf2_eigen.h"
#include "tf2_ros/static_transform_broadcaster.h"
#include "tf2_ros/transform_broadcaster.h"

#define PCL_NO_PRECOMPILE
#include <pcl/io/pcd_io.h>
#include <pcl/pcl_macros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "lgmath.hpp"

#include "steam_icp/point.hpp"

#include <Eigen/Core>
#include <iostream>

const uint64_t VLS128_CHANNEL_TDURATION_NS = 2665;
const uint64_t VLS128_SEQ_TDURATION_NS = 53300;
const uint64_t VLS128_FIRING_SEQUENCE_PER_REV = 1876;
const double AZIMUTH_STEP = 2 * M_PI / VLS128_FIRING_SEQUENCE_PER_REV;
const double INTER_AZM_STEP = AZIMUTH_STEP / 20;
// firing sequences per revolution: 1876

using namespace steam_icp;

#define PCL_ADD_FLEXIBLE     \
  union EIGEN_ALIGN16 {      \
    __uint128_t raw_flex1;   \
    float data_flex1[4];     \
    struct {                 \
      float flex11;          \
      float flex12;          \
      float flex13;          \
      float flex14;          \
    };                       \
    struct {                 \
      float alpha_timestamp; \
      float timestamp;       \
      float radial_velocity; \
    };                       \
  };

struct EIGEN_ALIGN16 PCLPoint3D {
  PCL_ADD_POINT4D;
  PCL_ADD_FLEXIBLE;
  PCL_MAKE_ALIGNED_OPERATOR_NEW

  inline PCLPoint3D() {
    x = y = z = 0.0f;
    data[3] = 1.0f;
    raw_flex1 = 0;
  }

  inline PCLPoint3D(const PCLPoint3D &p) {
    x = p.x;
    y = p.y;
    z = p.z;
    data[3] = 1.0f;
    raw_flex1 = p.raw_flex1;
  }

  inline PCLPoint3D(const Point3D &p) {
    x = (float)p.pt[0];
    y = (float)p.pt[1];
    z = (float)p.pt[2];
    data[3] = 1.0f;
    alpha_timestamp = p.alpha_timestamp;
    timestamp = p.timestamp;
    radial_velocity = p.radial_velocity;
  }

  inline PCLPoint3D(const Eigen::Vector3d &p) {
    x = (float)p[0];
    y = (float)p[1];
    z = (float)p[2];
    data[3] = 1.0f;
  }
};

struct SimulationOptions {
  std::string output_dir = "/sim_output";  // output path (relative or absolute) to save simulation data
  std::string root_path = "";
  std::string sequence = "";
  std::string lidar_config = "";
  Eigen::Matrix4d T_sr = Eigen::Matrix4d::Identity();
  int num_threads = 20;
  bool verbose = false;
  double imu_rate = 200.0;
  double offset_imu = 0.0025;  // offset bewteen first imu meas and first lidar meas
  double min_dist_sensor_center = 0.1;
  double max_dist_sensor_center = 200.0;
  bool noisy_measurements = false;
  double sim_length = 5.0;
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
  ROS2_PARAM_CLAUSE(node, options, prefix, root_path, std::string);
  ROS2_PARAM_CLAUSE(node, options, prefix, sequence, std::string);
  ROS2_PARAM_CLAUSE(node, options, prefix, lidar_config, std::string);
  ROS2_PARAM_CLAUSE(node, options, prefix, num_threads, int);
  ROS2_PARAM_CLAUSE(node, options, prefix, verbose, bool);
  ROS2_PARAM_CLAUSE(node, options, prefix, imu_rate, double);
  ROS2_PARAM_CLAUSE(node, options, prefix, offset_imu, double);
  ROS2_PARAM_CLAUSE(node, options, prefix, min_dist_sensor_center, double);
  ROS2_PARAM_CLAUSE(node, options, prefix, max_dist_sensor_center, double);
  ROS2_PARAM_CLAUSE(node, options, prefix, noisy_measurements, bool);
  ROS2_PARAM_CLAUSE(node, options, prefix, sim_length, double);
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

  std::vector<double> qc_diag;
  ROS2_PARAM_NO_LOG(node, qc_diag, prefix, qc_diag, std::vector<double>);
  if ((qc_diag.size() != 6) && (qc_diag.size() != 0))
    throw std::invalid_argument{"qc_diag malformed. Must be 6 elements!"};
  if (qc_diag.size() == 6) options.qc_diag << qc_diag[0], qc_diag[1], qc_diag[2], qc_diag[3], qc_diag[4], qc_diag[5];
  LOG(WARNING) << "Parameter " << prefix + "qc_diag"
               << " = " << options.qc_diag.transpose() << std::endl;

  std::vector<double> ad_diag;
  ROS2_PARAM_NO_LOG(node, ad_diag, prefix, ad_diag, std::vector<double>);
  if ((ad_diag.size() != 6) && (ad_diag.size() != 0))
    throw std::invalid_argument{"ad_diag malformed. Must be 6 elements!"};
  if (ad_diag.size() == 6) options.ad_diag << ad_diag[0], ad_diag[1], ad_diag[2], ad_diag[3], ad_diag[4], ad_diag[5];
  LOG(WARNING) << "Parameter " << prefix + "ad_diag"
               << " = " << options.ad_diag.transpose() << std::endl;

  Eigen::Matrix4d yfwd2xfwd;
  yfwd2xfwd << 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
  fs::path root_path{options.root_path};
  std::ifstream ifs(root_path / options.sequence / "calib" / "T_applanix_lidar.txt", std::ios::in);
  Eigen::Matrix4d T_applanix_lidar_mat;
  for (size_t row = 0; row < 4; row++)
    for (size_t col = 0; col < 4; col++) ifs >> T_applanix_lidar_mat(row, col);
  options.T_sr = (yfwd2xfwd * T_applanix_lidar_mat).inverse();
  LOG(WARNING) << "(BOREAS)Parameter T_sr = " << std::endl << options.T_sr << std::endl;

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

Eigen::Matrix<double, 128, 3> loadVLS128Config(const std::string &file_path) {
  Eigen::Matrix<double, 128, 3> output = Eigen::Matrix<double, 128, 3>::Zero();
  std::ifstream calib_file(file_path);
  if (calib_file.is_open()) {
    std::string line;
    std::getline(calib_file, line); // header
    int k = 0;
    for (; std::getline(calib_file, line);) {
      if (line.empty()) continue;
      if (k >= 128) break;
      std::stringstream ss(line);

      double rot_correction = 0;
      double vert_correction = 0;
      int laser_id = 0;

      for (int i = 0; i < 9; ++i) {
        std::string value;
        std::getline(ss, value, ',');
        if (i == 1)
          rot_correction = std::stod(value);
        else if (i == 2)
          vert_correction = std::stod(value);
        else if (i == 7)
          laser_id = std::stol(value);
      }
      output(k, 0) = laser_id;
      output(k, 1) = rot_correction;
      output(k, 2) = vert_correction;
      k++;
    }
  }
  return output;
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("simulation");
  auto odometry_publisher = node->create_publisher<nav_msgs::msg::Odometry>("/simulation_odometry", 10);
  auto tf_static_bc = std::make_shared<tf2_ros::StaticTransformBroadcaster>(node);
  auto tf_bc = std::make_shared<tf2_ros::TransformBroadcaster>(node);
  auto raw_points_publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>("/simulation_raw", 2);

  auto to_pc2_msg = [](const auto &points, const std::string &frame_id = "map") {
    pcl::PointCloud<PCLPoint3D> points_pcl;
    points_pcl.reserve(points.size());
    for (auto &pt : points) points_pcl.emplace_back(pt);
    sensor_msgs::msg::PointCloud2 points_msg;
    pcl::toROSMsg(points_pcl, points_msg);
    points_msg.header.frame_id = frame_id;
    // points_msg.header.stamp = rclcpp::Time(stamp);
    return points_msg;
  };

  // Logging
  FLAGS_log_dir = node->declare_parameter<std::string>("log_dir", "/tmp");
  FLAGS_alsologtostderr = 1;
  fs::create_directories(FLAGS_log_dir);
  google::InitGoogleLogging(argv[0]);
  LOG(WARNING) << "Logging to " << FLAGS_log_dir;

  // Read parameters
  auto options = loadOptions(node);

  // Publish sensor vehicle transformations
    auto T_rs_msg = tf2::eigenToTransform(Eigen::Affine3d(options.T_sr.inverse()));
    T_rs_msg.header.frame_id = "vehicle";
    T_rs_msg.child_frame_id = "sensor";
    tf_static_bc->sendTransform(T_rs_msg);

  // Build the Output_dir
  LOG(WARNING) << "Creating directory " << options.output_dir << std::endl;
  fs::create_directories(options.output_dir);

  // Load VLS128 Config
  const auto lidar_config = loadVLS128Config(options.lidar_config);
  LOG(WARNING) << "lidar config" << std::endl << lidar_config << std::endl;

  // Generate a sorted list of all timestamps in the simulation...
  uint64_t tns = 0;
  std::vector<uint64_t> sim_times;
  const uint64_t sim_length_ns = options.sim_length * 1.0e9;
  while (tns < sim_length_ns) {
    t += 2.665e-6;
  }

  t = options.offset_imu;
  const double dt_imu = 1.0 / options.imu_rate;
  while (t < sim_length) {
    sim_times.insert(std::upper_bound(sim_times.begin(), sim_times.end(), t), t);
    t += dt_imu;
  }

  LOG(INFO) << "simulation times: " << std::endl;
  for (auto ts : sim_times) {
    std::cout << ts << std::endl;
  }

  // TODO: pick a linear and angular jerk, and then step through simulation

  uint64_t tns = 0;
  Eigen::Matrix4d T_ri = Eigen::Matrix4d::Identity();
  const uint64_t delta_ns = VLS128_FIRING_SEQUENCE_PER_REV * VLS128_SEQ_TDURATION_NS;
  const double delta_s = delta_ns * 1.0e-9;
  Eigen::Matrix<double, 6, 1> dw = options.x0.block<6, 1>(12, 0);
  Eigen::Matrix<double, 6, 1> w = options.x0.block<6, 1>(6, 0);
  while (tns < sim_length_ns) {

    // build pointcloud
    uint64_t seq_index
    
    for (uint64_t seq_index = 0; seq_index < VLS128_FIRING_SEQUENCE_PER_REV; seq_index++) {
      // double sensor_azimuth = seq_index * AZIMUTH_STEP;
      // double sensor_tns = tns + seq_index * VLS128_SEQ_TDURATION_NS;

      for (int group = 0; group < 16; group++) {

        uint64_t sensor_tns = tns + seq_index * VLS128_SEQ_TDURATION_NS + group * VLS128_CHANNEL_TDURATION_NS;
        double sensor_azimuth = seq_index * AZIMUTH_STEP + group * INTER_AZM_STEP;
        if (group >= 8) {
          sensor_tns += VLS128_CHANNEL_TDURATION_NS;
          sensor_azimuth += INTER_AZM_STEP;
        }

        const double dtg = (sensor_tns - tns) * 1.0e-9;
        Eigen::Matrix4d T_ri_local = lgmath::se3::Transformation(w * dtg + 0.5 * pow(dtg, 2) * dw).matrix() * T_ri;
        
        for (int beam_id = group * 8; beam_id < group * 8 + 8; beam_id++) {
          const double beam_azimuth = sensor_azimuth - options.lidar_config(beam_id, 1);
          const double beam_elevation = options.lidar_config(beam_id, 2);
          // get vehicle position, orientation and compute the beam direction, sensor center.
          

        }


      }
    }
    // publish pointcloud
    tns += delta_ns;
    T_ri = lgmath::se3::Transformation(w * delta_s + 0.5 * pow(delta_s, 2) * dw).matrix() * T_ri;
    w += delta_s * dw;
  }

  LOG(INFO) << "hello simulation world" << std::endl;
  // [x]: parameters for singer prior 
  // [x]: load VLS128 config parameters
  // [x]: load extrinsics from yaml files
  // [ ]: create ordered list of timestamps required by IMU, lidar
  // [ ]: (sample from multidimensional gaussian)
    // [ ]: create sparse matrix for P_check_inv
    // [ ]: sample Gaussian noise vector Don't allow samples outside of 4-sigma (sample from truncated Gaussian)
  // [ ]: step through simulation, generating pointclouds and IMU measurements
  // [ ]: save pointclouds as .bin files and imu measurements in same formatted csv file
  // [ ]: add options for adding Gaussian noise to the lidar and IMU measurements

  // todo: x0

  // Simulation Parameters (move to config file)

  return 0;
}