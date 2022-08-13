#pragma once

#include <fstream>

#include "steam_icp/odometry.hpp"

namespace steam_icp {

class ElasticOdometry : public Odometry {
 public:
  struct Options : public Odometry::Options {
    double power_planarity = 2.0;
    double beta_location_consistency = 0.001;
    double beta_constant_velocity = 0.001;
    double max_dist_to_plane = 0.3;
    double convergence_threshold = 0.0001;
    int num_threads = 1;
  };

  ElasticOdometry(const Options &options);
  ~ElasticOdometry();

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

  STEAM_ICP_REGISTER_ODOMETRY("Elastic", ElasticOdometry);
};

}  // namespace steam_icp