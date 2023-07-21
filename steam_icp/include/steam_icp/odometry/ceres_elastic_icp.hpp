#pragma once

#include <fstream>

#include "steam_icp/odometry.hpp"

namespace steam_icp {

class CeresElasticOdometry : public Odometry {
 public:
  enum class CERES_LOSS_FUNC { L2, CAUCHY, HUBER, TOLERANT };

  struct Options : public Odometry::Options {
    double power_planarity = 2.0;
    double beta_location_consistency = 0.001;
    double beta_orientation_consistency = 0.0;
    double beta_constant_velocity = 0.001;
    double beta_small_velocity = 0.0;
    double max_dist_to_plane = 0.3;
    CERES_LOSS_FUNC loss_function = CERES_LOSS_FUNC::CAUCHY;
    double sigma = 0.1;
    double tolerant_min_threshold = 0.05;
    int max_iterations = 5;
    double weight_alpha = 0.9;
    double weight_neighborhood = 0.1;
    int num_threads = 1;
  };

  CeresElasticOdometry(const Options &options);
  ~CeresElasticOdometry();

  Trajectory trajectory() override;

  RegistrationSummary registerFrame(const std::pair<double, std::vector<Point3D>> &frame) override;

 private:
  void initializeTimestamp(int index_frame, const std::pair<double, std::vector<Point3D>> &const_frame);
  void initializeMotion(int index_frame);
  std::vector<Point3D> initializeFrame(int index_frame, const std::vector<Point3D> &const_frame);
  void updateMap(int index_frame, int update_frame);
  bool icp(int index_frame, std::vector<Point3D> &keypoints);

 private:
  const Options options_;

  STEAM_ICP_REGISTER_ODOMETRY("CeresElastic", CeresElasticOdometry);
};

}  // namespace steam_icp