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

#include "steam_icp/dataset.hpp"
#include "steam_icp/odometry.hpp"
#include "steam_icp/point.hpp"
#include "steam_icp/utils/stopwatch.hpp"

namespace steam_icp {

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

// Parameters to run the SLAM
struct SLAMOptions {
  bool save_trajectory = true;           // whether to save the trajectory
  bool suspend_on_failure = false;       // Whether to suspend the execution once an error is detected
  std::string output_dir = "./outputs";  // The output path (relative or absolute) to save the pointclouds

  struct {
    bool odometry = true;
    bool raw_points = true;
    bool sampled_points = true;
    bool map_points = true;
    Eigen::Matrix4d T_sr = Eigen::Matrix4d::Identity();
  } visualization_options;

  std::string dataset;
  Dataset::Options dataset_options;

  std::string odometry;
  Odometry::Options::Ptr odometry_options;
};

#define ROS2_PARAM_NO_LOG(node, receiver, prefix, param, type) \
  receiver = node->declare_parameter<type>(prefix + #param, receiver);
#define ROS2_PARAM(node, receiver, prefix, param, type)   \
  ROS2_PARAM_NO_LOG(node, receiver, prefix, param, type); \
  LOG(WARNING) << "Parameter " << prefix + #param << " = " << receiver << std::endl;
#define ROS2_PARAM_CLAUSE(node, config, prefix, param, type)                   \
  config.param = node->declare_parameter<type>(prefix + #param, config.param); \
  LOG(WARNING) << "Parameter " << prefix + #param << " = " << config.param << std::endl;

steam_icp::SLAMOptions loadOptions(const rclcpp::Node::SharedPtr &node) {
  steam_icp::SLAMOptions options;

  /// slam options
  {
    std::string prefix = "";
    ROS2_PARAM_CLAUSE(node, options, prefix, save_trajectory, bool);
    ROS2_PARAM_CLAUSE(node, options, prefix, suspend_on_failure, bool);
    ROS2_PARAM_CLAUSE(node, options, prefix, output_dir, std::string);
    if (!options.output_dir.empty() && options.output_dir[options.output_dir.size() - 1] != '/')
      options.output_dir += '/';
  }

  /// visualization options
  {
    auto &visualization_options = options.visualization_options;
    std::string prefix = "visualization_options.";

    ROS2_PARAM_CLAUSE(node, visualization_options, prefix, odometry, bool);
    ROS2_PARAM_CLAUSE(node, visualization_options, prefix, raw_points, bool);
    ROS2_PARAM_CLAUSE(node, visualization_options, prefix, sampled_points, bool);
    ROS2_PARAM_CLAUSE(node, visualization_options, prefix, map_points, bool);

    std::vector<double> T_sr_vec;
    ROS2_PARAM_NO_LOG(node, T_sr_vec, prefix, T_sr_vec, std::vector<double>);
    if ((T_sr_vec.size() != 6) && (T_sr_vec.size() != 0))
      throw std::invalid_argument{"T_sr malformed. Must be 6 elements!"};
    if (T_sr_vec.size() == 6)
      visualization_options.T_sr = lgmath::se3::vec2tran(Eigen::Matrix<double, 6, 1>(T_sr_vec.data()));
    LOG(WARNING) << "Parameter " << prefix + "T_sr"
                 << " = " << std::endl
                 << visualization_options.T_sr << std::endl;
  }

  /// dataset options
  {
    std::string prefix = "";
    ROS2_PARAM_CLAUSE(node, options, prefix, dataset, std::string);
    auto &dataset_options = options.dataset_options;
    prefix = "dataset_options.";

    ROS2_PARAM_CLAUSE(node, dataset_options, prefix, all_sequences, bool);
    ROS2_PARAM_CLAUSE(node, dataset_options, prefix, root_path, std::string);
    ROS2_PARAM_CLAUSE(node, dataset_options, prefix, sequence, std::string);
    ROS2_PARAM_CLAUSE(node, dataset_options, prefix, init_frame, int);
    ROS2_PARAM_CLAUSE(node, dataset_options, prefix, last_frame, int);
    ROS2_PARAM_CLAUSE(node, dataset_options, prefix, min_dist_lidar_center, float);
    ROS2_PARAM_CLAUSE(node, dataset_options, prefix, max_dist_lidar_center, float);
  }

  /// odometry options
  {
    std::string prefix = "";
    ROS2_PARAM_CLAUSE(node, options, prefix, odometry, std::string);
    if (options.odometry == "Elastic")
      options.odometry_options = std::make_shared<ElasticOdometry::Options>();
    else if (options.odometry == "CeresElastic")
      options.odometry_options = std::make_shared<CeresElasticOdometry::Options>();
    else if (options.odometry == "STEAM")
      options.odometry_options = std::make_shared<SteamOdometry::Options>();
    else
      throw std::invalid_argument{"Unknown odometry type!"};

    auto &odometry_options = *options.odometry_options;
    prefix = "odometry_options.";

    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, init_num_frames, int);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, init_voxel_size, double);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, voxel_size, double);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, init_sample_voxel_size, double);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, sample_voxel_size, double);

    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, size_voxel_map, double);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, min_distance_points, double);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, max_num_points_in_voxel, int);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, max_distance, double);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, min_number_neighbors, int);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, max_number_neighbors, int);

    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, num_iters_icp, int);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, threshold_orientation_norm, double);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, threshold_translation_norm, double);

    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, debug_print, bool);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, debug_path, std::string);

    if (options.odometry == "Elastic") {
      auto &elastic_icp_options = dynamic_cast<ElasticOdometry::Options &>(odometry_options);
      prefix = "odometry_options.elastic.";

      ROS2_PARAM_CLAUSE(node, elastic_icp_options, prefix, power_planarity, double);
      ROS2_PARAM_CLAUSE(node, elastic_icp_options, prefix, beta_location_consistency, double);
      ROS2_PARAM_CLAUSE(node, elastic_icp_options, prefix, beta_constant_velocity, double);
      ROS2_PARAM_CLAUSE(node, elastic_icp_options, prefix, max_dist_to_plane, double);
      ROS2_PARAM_CLAUSE(node, elastic_icp_options, prefix, convergence_threshold, double);
      ROS2_PARAM_CLAUSE(node, elastic_icp_options, prefix, num_threads, int);
    } else if (options.odometry == "CeresElastic") {
      auto &ceres_elastic_icp_options = dynamic_cast<CeresElasticOdometry::Options &>(odometry_options);
      prefix = "odometry_options.elastic.";

      ROS2_PARAM_CLAUSE(node, ceres_elastic_icp_options, prefix, power_planarity, double);
      ROS2_PARAM_CLAUSE(node, ceres_elastic_icp_options, prefix, beta_location_consistency, double);
      ROS2_PARAM_CLAUSE(node, ceres_elastic_icp_options, prefix, beta_orientation_consistency, double);
      ROS2_PARAM_CLAUSE(node, ceres_elastic_icp_options, prefix, beta_constant_velocity, double);
      ROS2_PARAM_CLAUSE(node, ceres_elastic_icp_options, prefix, beta_small_velocity, double);
      ROS2_PARAM_CLAUSE(node, ceres_elastic_icp_options, prefix, max_dist_to_plane, double);
      std::string loss_function;
      ROS2_PARAM(node, loss_function, prefix, loss_function, std::string);
      if (loss_function == "L2")
        ceres_elastic_icp_options.loss_function = CeresElasticOdometry::CERES_LOSS_FUNC::L2;
      else if (loss_function == "CAUCHY")
        ceres_elastic_icp_options.loss_function = CeresElasticOdometry::CERES_LOSS_FUNC::CAUCHY;
      else if (loss_function == "HUBER")
        ceres_elastic_icp_options.loss_function = CeresElasticOdometry::CERES_LOSS_FUNC::HUBER;
      else if (loss_function == "TOLERANT")
        ceres_elastic_icp_options.loss_function = CeresElasticOdometry::CERES_LOSS_FUNC::TOLERANT;
      else {
        LOG(WARNING) << "Parameter " << prefix + "loss_function"
                     << " not specified. Using default value: "
                     << "L2";
      }
      ROS2_PARAM_CLAUSE(node, ceres_elastic_icp_options, prefix, sigma, double);
      ROS2_PARAM_CLAUSE(node, ceres_elastic_icp_options, prefix, tolerant_min_threshold, double);
      ROS2_PARAM_CLAUSE(node, ceres_elastic_icp_options, prefix, max_iterations, int);
      ROS2_PARAM_CLAUSE(node, ceres_elastic_icp_options, prefix, weight_alpha, double);
      ROS2_PARAM_CLAUSE(node, ceres_elastic_icp_options, prefix, weight_neighborhood, double);
      ROS2_PARAM_CLAUSE(node, ceres_elastic_icp_options, prefix, num_threads, int);

    } else if (options.odometry == "STEAM") {
      auto &steam_icp_options = dynamic_cast<SteamOdometry::Options &>(odometry_options);
      prefix = "odometry_options.steam.";

      std::vector<double> T_sr_vec;
      ROS2_PARAM_NO_LOG(node, T_sr_vec, prefix, T_sr_vec, std::vector<double>);
      if ((T_sr_vec.size() != 6) && (T_sr_vec.size() != 0))
        throw std::invalid_argument{"T_sr malformed. Must be 6 elements!"};
      if (T_sr_vec.size() == 6)
        steam_icp_options.T_sr = lgmath::se3::vec2tran(Eigen::Matrix<double, 6, 1>(T_sr_vec.data()));
      LOG(WARNING) << "Parameter " << prefix + "T_sr"
                   << " = " << std::endl
                   << steam_icp_options.T_sr << std::endl;

      std::vector<double> qc_diag;
      ROS2_PARAM_NO_LOG(node, qc_diag, prefix, qc_diag, std::vector<double>);
      if ((qc_diag.size() != 6) && (qc_diag.size() != 0))
        throw std::invalid_argument{"Qc diagonal malformed. Must be 6 elements!"};
      if (qc_diag.size() == 6)
        steam_icp_options.qc_diag << qc_diag[0], qc_diag[1], qc_diag[2], qc_diag[3], qc_diag[4], qc_diag[5];
      LOG(WARNING) << "Parameter " << prefix + "qc_diag"
                   << " = " << steam_icp_options.qc_diag.transpose() << std::endl;

      ROS2_PARAM_CLAUSE(node, steam_icp_options, prefix, num_extra_states, int);
      ROS2_PARAM_CLAUSE(node, steam_icp_options, prefix, add_prev_state, bool);
      ROS2_PARAM_CLAUSE(node, steam_icp_options, prefix, num_extra_prev_states, int);
      ROS2_PARAM_CLAUSE(node, steam_icp_options, prefix, lock_prev_pose, bool);
      ROS2_PARAM_CLAUSE(node, steam_icp_options, prefix, lock_prev_vel, bool);
      ROS2_PARAM_CLAUSE(node, steam_icp_options, prefix, prev_pose_as_prior, bool);
      ROS2_PARAM_CLAUSE(node, steam_icp_options, prefix, prev_vel_as_prior, bool);
      ROS2_PARAM_CLAUSE(node, steam_icp_options, prefix, no_prev_state_iters, int);
      ROS2_PARAM_CLAUSE(node, steam_icp_options, prefix, association_after_adding_prev_state, bool);

      ROS2_PARAM_CLAUSE(node, steam_icp_options, prefix, use_vp, bool);

      std::vector<double> vp_cov_diag;
      ROS2_PARAM_NO_LOG(node, vp_cov_diag, prefix, vp_cov_diag, std::vector<double>);
      if ((vp_cov_diag.size() != 6) && (vp_cov_diag.size() != 0))
        throw std::invalid_argument{"Velocity prior cov malformed. Must be 6 elements!"};
      if (vp_cov_diag.size() == 6)
        steam_icp_options.vp_cov.diagonal() << vp_cov_diag[0], vp_cov_diag[1], vp_cov_diag[2], vp_cov_diag[3],
            vp_cov_diag[4], vp_cov_diag[5];
      LOG(WARNING) << "Parameter " << prefix + "vp_cov_diag"
                   << " = " << steam_icp_options.vp_cov.diagonal().transpose() << std::endl;

      ROS2_PARAM_CLAUSE(node, steam_icp_options, prefix, power_planarity, double);
      ROS2_PARAM_CLAUSE(node, steam_icp_options, prefix, p2p_initial_iters, int);
      ROS2_PARAM_CLAUSE(node, steam_icp_options, prefix, p2p_initial_max_dist, double);
      ROS2_PARAM_CLAUSE(node, steam_icp_options, prefix, p2p_refined_max_dist, double);
      std::string p2p_loss_func;
      ROS2_PARAM(node, p2p_loss_func, prefix, p2p_loss_func, std::string);
      if (p2p_loss_func == "L2")
        steam_icp_options.p2p_loss_func = SteamOdometry::STEAM_LOSS_FUNC::L2;
      else if (p2p_loss_func == "DCS")
        steam_icp_options.p2p_loss_func = SteamOdometry::STEAM_LOSS_FUNC::DCS;
      else if (p2p_loss_func == "CAUCHY")
        steam_icp_options.p2p_loss_func = SteamOdometry::STEAM_LOSS_FUNC::CAUCHY;
      else if (p2p_loss_func == "GM")
        steam_icp_options.p2p_loss_func = SteamOdometry::STEAM_LOSS_FUNC::GM;
      else {
        LOG(WARNING) << "Parameter " << prefix + "p2p_loss_func"
                     << " not specified. Using default value: "
                     << "L2";
      }
      ROS2_PARAM_CLAUSE(node, steam_icp_options, prefix, p2p_loss_sigma, double);

      ROS2_PARAM_CLAUSE(node, steam_icp_options, prefix, use_rv, bool);
      ROS2_PARAM_CLAUSE(node, steam_icp_options, prefix, merge_p2p_rv, bool);
      std::string rv_loss_func;
      ROS2_PARAM(node, rv_loss_func, prefix, rv_loss_func, std::string);
      if (rv_loss_func == "L2")
        steam_icp_options.rv_loss_func = SteamOdometry::STEAM_LOSS_FUNC::L2;
      else if (rv_loss_func == "DCS")
        steam_icp_options.rv_loss_func = SteamOdometry::STEAM_LOSS_FUNC::DCS;
      else if (rv_loss_func == "CAUCHY")
        steam_icp_options.rv_loss_func = SteamOdometry::STEAM_LOSS_FUNC::CAUCHY;
      else if (rv_loss_func == "GM")
        steam_icp_options.rv_loss_func = SteamOdometry::STEAM_LOSS_FUNC::GM;
      else {
        LOG(WARNING) << "Parameter " << prefix + "rv_loss_func"
                     << " not specified. Using default value: "
                     << "GM";
      }
      ROS2_PARAM_CLAUSE(node, steam_icp_options, prefix, rv_cov_inv, double);
      ROS2_PARAM_CLAUSE(node, steam_icp_options, prefix, rv_loss_threshold, double);

      ROS2_PARAM_CLAUSE(node, steam_icp_options, prefix, verbose, bool);
      ROS2_PARAM_CLAUSE(node, steam_icp_options, prefix, max_iterations, int);
      ROS2_PARAM_CLAUSE(node, steam_icp_options, prefix, num_threads, int);

      ROS2_PARAM_CLAUSE(node, steam_icp_options, prefix, delay_adding_points, int);
      ROS2_PARAM_CLAUSE(node, steam_icp_options, prefix, use_final_state_value, bool);
    }
  }

  return options;
}

}  // namespace steam_icp

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(
    steam_icp::PCLPoint3D,
    // cartesian coordinates
    (float, x, x)
    (float, y, y)
    (float, z, z)
    // random stuff
    (float, flex11, flex11)
    (float, flex12, flex12)
    (float, flex13, flex13)
    (float, flex14, flex14))
// clang-format on

int main(int argc, char **argv) {
  using namespace steam_icp;

  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("steam_icp");
  auto odometry_publisher = node->create_publisher<nav_msgs::msg::Odometry>("/steam_icp_odometry", 10);
  auto tf_static_bc = std::make_shared<tf2_ros::StaticTransformBroadcaster>(node);
  auto tf_bc = std::make_shared<tf2_ros::TransformBroadcaster>(node);
  auto raw_points_publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>("/steam_icp_raw", 2);
  auto sampled_points_publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>("/steam_icp_sampled", 2);
  auto map_points_publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>("/steam_icp_map", 2);

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
  const auto options = loadOptions(node);

  // Publish sensor vehicle transformations
  auto T_rs_msg = tf2::eigenToTransform(Eigen::Affine3d(options.visualization_options.T_sr.inverse()));
  T_rs_msg.header.frame_id = "vehicle";
  T_rs_msg.child_frame_id = "lidar";
  tf_static_bc->sendTransform(T_rs_msg);

  // Build the Output_dir
  LOG(WARNING) << "Creating directory " << options.output_dir << std::endl;
  fs::create_directories(options.output_dir);

  // Get dataset
  const auto dataset = Dataset::Get(options.dataset, options.dataset_options);

  // Error report in case there is ground truth
  std::vector<Sequence::SeqError> sequence_errors;

  while (auto seq = dataset->next()) {
    LOG(WARNING) << "Running odometry on sequence: " << seq->name() << std::endl;

    // timers
    std::vector<std::pair<std::string, std::unique_ptr<Stopwatch<>>>> timer;
    timer.emplace_back("loading ..................... ", std::make_unique<Stopwatch<>>(false));
    timer.emplace_back("registration ................ ", std::make_unique<Stopwatch<>>(false));
    timer.emplace_back("visualization ............... ", std::make_unique<Stopwatch<>>(false));

    const auto odometry = Odometry::Get(options.odometry, *options.odometry_options);
    bool odometry_success = true;
    while (seq->hasNext()) {
      LOG(INFO) << "Processing frame " << seq->currFrame() << std::endl;

      timer[0].second->start();
      auto frame = seq->next();
      timer[0].second->stop();

      timer[2].second->start();
      if (options.visualization_options.raw_points) {
        auto &raw_points = frame;
        auto raw_points_msg = to_pc2_msg(raw_points, "lidar");
        raw_points_publisher->publish(raw_points_msg);
      }
      timer[2].second->stop();

      timer[1].second->start();
      const auto summary = odometry->registerFrame(frame);
      timer[1].second->stop();
      if (!summary.success) {
        LOG(ERROR) << "Error running odometry for sequence " << seq->name() << ", at frame index " << seq->currFrame()
                   << std::endl;
        if (options.suspend_on_failure) return 1;

        odometry_success = false;
        break;
      }

      timer[2].second->start();
      if (options.visualization_options.odometry) {
        Eigen::Matrix4d T_ws = Eigen::Matrix4d::Identity();
        T_ws.block<3, 3>(0, 0) = summary.R_ms;
        T_ws.block<3, 1>(0, 3) = summary.t_ms;
        Eigen::Matrix4d T_wr = T_ws * options.visualization_options.T_sr;

        /// odometry
        nav_msgs::msg::Odometry odometry;
        odometry.header.frame_id = "map";
        // odometry.header.stamp = rclcpp::Time(stamp);
        odometry.pose.pose = tf2::toMsg(Eigen::Affine3d(T_wr));
        odometry_publisher->publish(odometry);

        /// tf
        auto T_wr_msg = tf2::eigenToTransform(Eigen::Affine3d(T_wr));
        T_wr_msg.header.frame_id = "map";
        // T_wr_msg.header.stamp = rclcpp::Time(stamp);
        T_wr_msg.child_frame_id = "vehicle";
        tf_bc->sendTransform(T_wr_msg);
      }
      if (options.visualization_options.sampled_points) {
        /// sampled points
        auto &sampled_points = summary.corrected_points;
        auto sampled_points_msg = to_pc2_msg(sampled_points, "map");
        sampled_points_publisher->publish(sampled_points_msg);
      }
      if (options.visualization_options.map_points) {
        /// map points
        auto map_points = odometry->map();
        auto map_points_msg = to_pc2_msg(map_points, "map");
        map_points_publisher->publish(map_points_msg);
      }
      timer[2].second->stop();

      if (!rclcpp::ok()) {
        LOG(WARNING) << "Shutting down due to ctrl-c." << std::endl;
        return 0;
      }
    }

    if (!odometry_success) {
      LOG(ERROR) << "Failed on sequence " << seq->name() << " after " << odometry->trajectory().size() << " frames."
                 << std::endl;
      break;
    }

    // dump timing information
    for (size_t i = 0; i < timer.size(); i++) {
      LOG(WARNING) << "Average " << timer[i].first << (timer[i].second->count() / (double)seq->numFrames()) << " ms"
                   << std::endl;
    }

    // transform and save the estimated trajectory
    seq->save(options.output_dir, odometry->trajectory());

    // ground truth
    if (seq->hasGroundTruth()) {
      const auto seq_error = seq->evaluate(options.output_dir, odometry->trajectory());
      LOG(WARNING) << "Mean RPE : " << seq_error.mean_t_rpe << std::endl;
      LOG(WARNING) << "Mean RPE 2D : " << seq_error.mean_t_rpe_2d << std::endl;
      LOG(WARNING) << "Mean APE : " << seq_error.mean_ape << std::endl;
      LOG(WARNING) << "Max APE : " << seq_error.max_ape << std::endl;
      LOG(WARNING) << "Mean Local Error : " << seq_error.mean_local_err << std::endl;
      LOG(WARNING) << "Max Local Error : " << seq_error.max_local_err << std::endl;
      LOG(WARNING) << "Index Max Local Error : " << seq_error.index_max_local_err << std::endl;
      // clang-format off
      LOG(WARNING) << "KITTI Summary : "
                   << std::fixed << std::setprecision(2) << seq_error.mean_t_rpe_2d << " & "
                   << std::fixed << std::setprecision(4) << seq_error.mean_r_rpe_2d << " & "
                   << std::fixed << std::setprecision(2) << seq_error.mean_t_rpe << " & "
                   << std::fixed << std::setprecision(4) << seq_error.mean_r_rpe << "\\\\" << std::endl;
      // clang-format on
      LOG(WARNING) << std::endl;

      sequence_errors.emplace_back(seq_error);
    }
  }

  // error report in case there is ground truth
  if (sequence_errors.size() > 0) {
    LOG(WARNING) << std::endl;
    double all_seq_rpe_t = 0.0;
    double all_seq_rpe_r = 0.0;
    double num_total_errors = 0.0;
    for (const auto &seq_error : sequence_errors) {
      for (const auto &tab_error : seq_error.tab_errors) {
        all_seq_rpe_t += tab_error.t_err;
        all_seq_rpe_r += tab_error.r_err;
        num_total_errors += 1.0;
      }
    }
    LOG(WARNING) << "KITTI metric translation/rotation : " << (all_seq_rpe_t / num_total_errors) * 100 << " "
                 << (all_seq_rpe_r / num_total_errors) * 180.0 / M_PI << std::endl;
  }

  rclcpp::shutdown();

  return 0;
}