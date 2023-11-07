#pragma once

#include "steam/problem/cost_term/imu_super_cost_term.hpp"
#include "steam_icp/dataframe.hpp"
#include "steam_icp/map.hpp"
#include "steam_icp/pose.hpp"
#include "steam_icp/trajectory.hpp"

namespace steam_icp {

class Odometry {
 public:
  using Ptr = std::shared_ptr<Odometry>;
  using ConstPtr = std::shared_ptr<const Odometry>;

  ArrayPoses T_i_r_gt_poses;

  struct Options {
    using Ptr = std::shared_ptr<Options>;
    using ConstPtr = std::shared_ptr<const Options>;

    virtual ~Options() = default;

    // frame preprocessing
    int init_num_frames = 20;  // The number of frames defining the initialization of the map
    double init_voxel_size = 0.2;
    double voxel_size = 0.5;
    double init_sample_voxel_size = 1.0;
    double sample_voxel_size = 1.5;

    // map
    double size_voxel_map = 1.0;       // Max Voxel : -32767 to 32767 then 32km map for SIZE_VOXEL_MAP = 1m
    double min_distance_points = 0.1;  // The minimal distance between points in the map
    int max_num_points_in_voxel = 20;  // The maximum number of points in a voxel
    double max_distance = 100.0;       // The threshold on the voxel size to remove points from the map
    int min_number_neighbors = 20;     // The minimum number of neighbors to be considered in the map
    int max_number_neighbors = 20;
    int voxel_lifetime = 10;

    // common icp options
    int num_iters_icp = 10;                      // The Maximum number of ICP iterations performed
    double threshold_orientation_norm = 0.0001;  // Threshold on rotation (deg) for ICP's stopping criterion
    double threshold_translation_norm = 0.001;   // Threshold on translation (deg) for ICP's stopping criterion
    int min_number_keypoints = 100;

    //
    bool debug_print = false;  // Whether to output debug information to std::cout
    std::string debug_path = "/tmp/";
  };

  static Odometry::Ptr Get(const std::string &odometry, const Options &options) {
    return name2Ctor().at(odometry)(options);
  }

  Odometry(const Options &options) : options_(options) { map_.setDefaultLifeTime(options_.voxel_lifetime); }
  virtual ~Odometry() = default;

  // trajectory
  // virtual Trajectory trajectory() { return trajectory_; }
  virtual Trajectory trajectory() = 0;

  // map
  size_t size() const { return map_.size(); }
  ArrayVector3d map() const { return map_.pointcloud(); }

  // The Output of a registration, including metrics,
  struct RegistrationSummary {
    std::vector<Point3D> keypoints;                      // Last Keypoints selected
    std::vector<Point3D> corrected_points;               // Sampled points expressed in the initial frame
    std::vector<Point3D> all_corrected_points;           // Initial points expressed in the initial frame
    Eigen::Matrix3d R_ms = Eigen::Matrix3d::Identity();  // The rotation between the initial frame and the new frame
    Eigen::Vector3d t_ms = Eigen::Vector3d::Zero();      // The translation between the initial frame and the new frame
    bool success = true;                                 // Whether the registration was a success
  };
  // Registers a new Frame to the Map with an initial estimate
  virtual RegistrationSummary registerFrame(const DataFrame &frame) = 0;

 protected:
  Trajectory trajectory_;
  Map map_;

 private:
  const Options options_;

 private:
  using CtorFunc = std::function<Ptr(const Options &)>;
  using Name2Ctor = std::unordered_map<std::string, CtorFunc>;
  static Name2Ctor &name2Ctor() {
    static Name2Ctor name2ctor;
    return name2ctor;
  }

  template <typename T>
  friend class OdometryRegister;
};

template <typename T>
struct OdometryRegister {
  OdometryRegister() {
    bool success = Odometry::name2Ctor()
                       .try_emplace(T::odometry_name_, Odometry::CtorFunc([](const Odometry::Options &options) {
                                      return std::make_shared<T>(dynamic_cast<const typename T::Options &>(options));
                                    }))
                       .second;
    if (!success) throw std::runtime_error{"OdometryRegister failed - duplicated name"};
  }
};

#define STEAM_ICP_REGISTER_ODOMETRY(NAME, TYPE)       \
 public:                                              \
  inline static constexpr auto odometry_name_ = NAME; \
                                                      \
 private:                                             \
  inline static steam_icp::OdometryRegister<TYPE> odometry_reg_;

}  // namespace steam_icp

///
#include "steam_icp/odometry/ceres_elastic_icp.hpp"
#include "steam_icp/odometry/elastic_icp.hpp"
#include "steam_icp/odometry/steam_icp.hpp"
#include "steam_icp/odometry/steam_lio.hpp"
#include "steam_icp/odometry/steam_rio.hpp"
#include "steam_icp/odometry/steam_ro.hpp"