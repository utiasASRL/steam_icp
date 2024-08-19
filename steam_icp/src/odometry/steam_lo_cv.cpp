#include "steam_icp/odometry/steam_lo_cv.hpp"

#include <iomanip>
#include <random>

#include <glog/logging.h>

#include "steam.hpp"

#include "steam_icp/utils/stopwatch.hpp"

namespace steam_icp {

namespace {

inline double AngularDistance(const Eigen::Matrix3d &rota, const Eigen::Matrix3d &rotb) {
  double d = 0.5 * ((rota * rotb.transpose()).trace() - 1);
  return std::acos(std::max(std::min(d, 1.0), -1.0)) * 180.0 / M_PI;
}

/* -------------------------------------------------------------------------------------------------------------- */
// Subsample to keep one (random) point in every voxel of the current frame
// Run std::shuffle() first in order to retain a random point for each voxel.
void sub_sample_frame(std::vector<Point3D> &frame, double size_voxel, int num_threads) {
  using VoxelMap = tsl::robin_map<Voxel, Point3D>;
  VoxelMap voxel_map;

  for (unsigned int i = 0; i < frame.size(); i++) {
    const auto kx = static_cast<short>(frame[i].pt[0] / size_voxel);
    const auto ky = static_cast<short>(frame[i].pt[1] / size_voxel);
    const auto kz = static_cast<short>(frame[i].pt[2] / size_voxel);
    const auto voxel = Voxel(kx, ky, kz);
    voxel_map.try_emplace(voxel, frame[i]);
  }
  frame.clear();
  std::transform(voxel_map.begin(), voxel_map.end(), std::back_inserter(frame),
                 [](const auto &pair) { return pair.second; });
  frame.shrink_to_fit();
}

/* -------------------------------------------------------------------------------------------------------------- */
void grid_sampling(const std::vector<Point3D> &frame, std::vector<Point3D> &keypoints, double size_voxel_subsampling,
                   int num_threads) {
  keypoints.clear();
  std::vector<Point3D> frame_sub;
  frame_sub.resize(frame.size());
#pragma omp parallel for num_threads(2)
  for (int i = 0; i < (int)frame_sub.size(); i++) {
    frame_sub[i] = frame[i];
  }
  sub_sample_frame(frame_sub, size_voxel_subsampling, num_threads);
  keypoints.reserve(frame_sub.size());
  std::transform(frame_sub.begin(), frame_sub.end(), std::back_inserter(keypoints), [](const auto c) { return c; });
  keypoints.shrink_to_fit();
}

/* -------------------------------------------------------------------------------------------------------------- */

struct Neighborhood {
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal = Eigen::Vector3d::Zero();
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();
  double a2D = 1.0;  // Planarity coefficient
};
// Computes normal and planarity coefficient
Neighborhood compute_neighborhood_distribution(const ArrayVector3d &points) {
  Neighborhood neighborhood;
  // Compute the normals
  Eigen::Vector3d barycenter(Eigen::Vector3d(0, 0, 0));
  for (auto &point : points) {
    barycenter += point;
  }
  barycenter /= (double)points.size();
  neighborhood.center = barycenter;

  Eigen::Matrix3d covariance_Matrix(Eigen::Matrix3d::Zero());
  for (auto &point : points) {
    for (int k = 0; k < 3; ++k)
      for (int l = k; l < 3; ++l) covariance_Matrix(k, l) += (point(k) - barycenter(k)) * (point(l) - barycenter(l));
  }
  covariance_Matrix(1, 0) = covariance_Matrix(0, 1);
  covariance_Matrix(2, 0) = covariance_Matrix(0, 2);
  covariance_Matrix(2, 1) = covariance_Matrix(1, 2);
  neighborhood.covariance = covariance_Matrix;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(covariance_Matrix);
  Eigen::Vector3d normal(es.eigenvectors().col(0).normalized());
  neighborhood.normal = normal;

  // Compute planarity from the eigen values
  double sigma_1 = sqrt(std::abs(es.eigenvalues()[2]));  // Be careful, the eigenvalues are not correct with the
                                                         // iterative way to compute the covariance matrix
  double sigma_2 = sqrt(std::abs(es.eigenvalues()[1]));
  double sigma_3 = sqrt(std::abs(es.eigenvalues()[0]));
  neighborhood.a2D = (sigma_2 - sigma_3) / sigma_1;

  if (neighborhood.a2D != neighborhood.a2D) {
    LOG(ERROR) << "FOUND NAN!!!";
    throw std::runtime_error("error");
  }
  return neighborhood;
}

}  // namespace

SteamLoCVOdometry::SteamLoCVOdometry(const Options &options) : Odometry(options), options_(options) {
  T_sr_var_ = steam::se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(options_.T_sr));
  T_sr_var_->locked() = true;
  sliding_window_filter_ = steam::SlidingWindowFilter::MakeShared(options_.num_threads);
}

SteamLoCVOdometry::~SteamLoCVOdometry() { }

Trajectory SteamLoCVOdometry::trajectory() {
  return trajectory_;
}

auto SteamLoCVOdometry::registerFrame(const DataFrame &const_frame) -> RegistrationSummary {
  RegistrationSummary summary;

  int index_frame = trajectory_.size();
  trajectory_.emplace_back();
  initializeTimestamp(index_frame, const_frame);
  auto frame = initializeFrame(index_frame, const_frame.pointcloud);

  std::vector<Point3D> keypoints;
  if (index_frame > 0) {
    double sample_voxel_size =
        index_frame < options_.init_num_frames ? options_.init_sample_voxel_size : options_.sample_voxel_size;

    grid_sampling(frame, keypoints, sample_voxel_size, options_.num_threads);

    summary.success = icp(index_frame, keypoints, const_frame.imu_data_vec);
    summary.keypoints = keypoints;
    if (!summary.success) return summary;
  } else {
    using namespace steam;
    using namespace steam::se3;
    using namespace steam::vspace;
    using namespace steam::traj;
    // initial state
    lgmath::se3::Transformation T_rm;
    lgmath::se3::Transformation T_mi;
    lgmath::se3::Transformation T_sr(options_.T_sr);
    Eigen::Matrix<double, 6, 1> w_mr_inr = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 1> b_zero = Eigen::Matrix<double, 6, 1>::Zero();
    // initialize frame (the beginning of the trajectory)
    const double begin_time = trajectory_[index_frame].getEvalTime();
    Time begin_steam_time(begin_time);
    const auto begin_T_rm_var = SE3StateVar::MakeShared(T_rm);
    const auto begin_w_mr_inr_var = VSpaceStateVar<6>::MakeShared(w_mr_inr);
    const auto begin_b_var = VSpaceStateVar<6>::MakeShared(b_zero);
    const auto begin_T_mi_var = SE3StateVar::MakeShared(T_mi);
    trajectory_vars_.emplace_back(begin_steam_time, begin_T_rm_var, begin_w_mr_inr_var, begin_b_var, begin_T_mi_var);
    trajectory_[index_frame].end_T_rm_cov = Eigen::Matrix<double, 6, 6>::Identity() * 1e-4;
    trajectory_[index_frame].end_w_mr_inr_cov = Eigen::Matrix<double, 6, 6>::Identity() * 1e-4;
    trajectory_[index_frame].end_state_cov = Eigen::Matrix<double, 18, 18>::Identity() * 1e-4;
    summary.success = true;
  }
  trajectory_[index_frame].points = frame;

  const Eigen::Vector3d t = trajectory_[index_frame].end_t;
  const Eigen::Matrix3d r = trajectory_[index_frame].end_R;

  // add points
  if (index_frame == 0) {
    updateMap(index_frame, index_frame);
  } else if ((index_frame - options_.delay_adding_points) > 0) {
    if ((t - t_prev_).norm() > options_.keyframe_translation_threshold_m || fabs(AngularDistance(r, r_prev_)) > options_.keyframe_rotation_threshold_deg) {
      updateMap(index_frame, (index_frame - options_.delay_adding_points));
      t_prev_ = t;
      r_prev_ = r;
    }
  }

  summary.corrected_points = keypoints;
  summary.R_ms = trajectory_[index_frame].end_R;
  summary.t_ms = trajectory_[index_frame].end_t;
  return summary;
}

void SteamLoCVOdometry::initializeTimestamp(int index_frame, const DataFrame &const_frame) {
  double min_timestamp = std::numeric_limits<double>::max();
  double max_timestamp = std::numeric_limits<double>::min();
#pragma omp parallel for num_threads(2) reduction(min : min_timestamp) reduction(max : max_timestamp)
  for (const auto &point : const_frame.pointcloud) {
    if (point.timestamp > max_timestamp) max_timestamp = point.timestamp;
    if (point.timestamp < min_timestamp) min_timestamp = point.timestamp;
  }
  trajectory_[index_frame].begin_timestamp = min_timestamp;
  trajectory_[index_frame].end_timestamp = max_timestamp;
  // purpose: eval trajectory at the exact file stamp to match ground truth
  trajectory_[index_frame].setEvalTime(const_frame.timestamp);
}

void SteamLoCVOdometry::initializeMotion(int index_frame) {
  if (index_frame <= 1) {
    // Initialize first pose at Identity
    const Eigen::Matrix4d T_rs = options_.T_sr.inverse();
    trajectory_[index_frame].begin_R = T_rs.block<3, 3>(0, 0);
    trajectory_[index_frame].begin_t = T_rs.block<3, 1>(0, 3);
    trajectory_[index_frame].end_R = T_rs.block<3, 3>(0, 0);
    trajectory_[index_frame].end_t = T_rs.block<3, 1>(0, 3);
  } else {
    // Different regimen for the second frame due to the bootstrapped elasticity
    Eigen::Matrix3d R_next_end = trajectory_[index_frame - 1].end_R * trajectory_[index_frame - 2].end_R.inverse() *
                                 trajectory_[index_frame - 1].end_R;
    Eigen::Vector3d t_next_end = trajectory_[index_frame - 1].end_t +
                                 trajectory_[index_frame - 1].end_R * trajectory_[index_frame - 2].end_R.inverse() *
                                     (trajectory_[index_frame - 1].end_t - trajectory_[index_frame - 2].end_t);

    trajectory_[index_frame].begin_R = trajectory_[index_frame - 1].end_R;
    trajectory_[index_frame].begin_t = trajectory_[index_frame - 1].end_t;
    trajectory_[index_frame].end_R = R_next_end;
    trajectory_[index_frame].end_t = t_next_end;
  }
}

std::vector<Point3D> SteamLoCVOdometry::initializeFrame(int index_frame, const std::vector<Point3D> &const_frame) {
  std::vector<Point3D> frame(const_frame);
  double sample_size = index_frame < options_.init_num_frames ? options_.init_voxel_size : options_.voxel_size;
  std::mt19937_64 g;
  std::shuffle(frame.begin(), frame.end(), g);
  // Subsample the scan with voxels taking one random in every voxel
  sub_sample_frame(frame, sample_size, options_.num_threads);
  std::shuffle(frame.begin(), frame.end(), g);
  return frame;
}

void SteamLoCVOdometry::updateMap(int index_frame, int update_frame) {
  using namespace steam::se3;
  using namespace steam::traj;
  const double kSizeVoxelMap = options_.size_voxel_map;
  const double kMinDistancePoints = options_.min_distance_points;
  const int kMaxNumPointsInVoxel = options_.max_num_points_in_voxel;
  // update frame
  auto &frame = trajectory_[update_frame].points;
  Time begin_steam_time = trajectory_[update_frame].begin_timestamp;
  Time end_steam_time = trajectory_[update_frame].end_timestamp;

  const auto T_rm = trajectory_vars_.back().T_rm;
  const double curr_time = trajectory_vars_.back().time.seconds();
  const auto T_mr = T_rm->value().inverse().matrix();

  LOG(INFO) << "Adding points to map between (inclusive): " << begin_steam_time.seconds() << " - "
            << end_steam_time.seconds() << ", with num states: 1" << std::endl;

  std::set<double> unique_point_times_;
  for (const auto &keypoint : frame) {
    unique_point_times_.insert(keypoint.timestamp);
  }
  std::vector<double> unique_point_times(unique_point_times_.begin(), unique_point_times_.end());
  std::map<double, Eigen::Matrix4d> T_ms_cache_map;
  const Eigen::Matrix4d T_rs = options_.T_sr.inverse();
  const Eigen::Matrix<double, 6, 1> w_mr_inr = trajectory_vars_.back().w_mr_inr->value();
  for (int jj = 0; jj < (int)unique_point_times.size(); jj++) {
    const auto &ts = unique_point_times[jj];
    const lgmath::se3::Transformation T_kj(Eigen::Matrix<double, 6, 1>(w_mr_inr * (curr_time - ts)));
    T_ms_cache_map[ts] = T_mr * T_kj.matrix() * T_rs;
  }
#pragma omp parallel for num_threads(options_.num_threads)
  for (unsigned i = 0; i < frame.size(); i++) {
    const Eigen::Matrix4d &T_ms = T_ms_cache_map[frame[i].timestamp];
    frame[i].pt = T_ms.block<3, 3>(0, 0) * frame[i].raw_pt + T_ms.block<3, 1>(0, 3);
  }

  map_.add(frame, kSizeVoxelMap, kMaxNumPointsInVoxel, kMinDistancePoints);
  if (options_.filter_lifetimes) map_.update_and_filter_lifetimes();
  frame.clear();
  frame.shrink_to_fit();

  // remove points
  const double kMaxDistance = options_.max_distance;
  const Eigen::Vector3d location = trajectory_[index_frame].end_t;
  map_.remove(location, kMaxDistance);
}

bool SteamLoCVOdometry::icp(int index_frame, std::vector<Point3D> &keypoints,
                          const std::vector<steam::IMUData> &imu_data_vec) {
  using namespace steam;
  using namespace steam::se3;
  using namespace steam::traj;
  using namespace steam::vspace;
  using namespace steam::imu;

  bool icp_success = true;

  std::vector<StateVarBase::Ptr> steam_state_vars;
  std::vector<BaseCostTerm::ConstPtr> meas_cost_terms;
  const size_t prev_trajectory_var_index = trajectory_vars_.size() - 1;
  size_t curr_trajectory_var_index = trajectory_vars_.size() - 1;

  const double prev_time = trajectory_[index_frame - 1].getEvalTime();
  LOG(INFO) << "[CT_ICP_STEAM] prev scan end time: " << prev_time << std::endl;
  if (trajectory_vars_.back().time != Time(static_cast<double>(prev_time)))
    throw std::runtime_error{"missing previous scan end variable"};
  const lgmath::se3::Transformation T_rm_prev = trajectory_vars_.back().T_rm->value();
  
  Eigen::Matrix<double, 6, 1> prev_imu_biases = trajectory_vars_.back().imu_biases->value();
  lgmath::se3::Transformation prev_T_mi = trajectory_vars_.back().T_mi->value();

  LOG(INFO) << "[CT_ICP_STEAM] curr scan end time: " << trajectory_[index_frame].getEvalTime() << std::endl;
  const double curr_time = trajectory_[index_frame].getEvalTime();
  const double time_diff = curr_time - prev_time;

  Eigen::Matrix<double, 6, 1> prev_w_mr_inr = Eigen::Matrix<double, 6, 1>::Zero();
  if (index_frame > 2) {
    const lgmath::se3::Transformation T_rm_prev_prev = trajectory_vars_[trajectory_vars_.size() - 2].T_rm->value();
    prev_w_mr_inr = (T_rm_prev * T_rm_prev_prev.inverse()).vec() / (trajectory_vars_[trajectory_vars_.size() - 1].time - trajectory_vars_[trajectory_vars_.size() - 2].time).seconds();
  }
  LOG(INFO) << "prev_w_mr_in_r: " << prev_w_mr_inr.transpose() << std::endl;
      
  const Eigen::Matrix<double, 6, 1> xi_mr_inr_odo(time_diff * prev_w_mr_inr);
  const auto T_rm_prediction = lgmath::se3::Transformation(xi_mr_inr_odo) * T_rm_prev;
  const auto T_rm_var =  SE3StateVar::MakeShared(T_rm_prediction);
  const auto T_mr_eval = InverseEvaluator::MakeShared(T_rm_var);
  
  // dummy variables for compatibility with trajectory_vars_
  const auto w_mr_inr_var = VSpaceStateVar<6>::MakeShared(prev_w_mr_inr);
  const auto imu_biases_var = VSpaceStateVar<6>::MakeShared(prev_imu_biases);
  const auto T_mi_var = SE3StateVar::MakeShared(prev_T_mi);
  // the only variable to be optimized
  steam_state_vars.emplace_back(T_rm_var);

  trajectory_vars_.emplace_back(Time(curr_time), T_rm_var, w_mr_inr_var, imu_biases_var, T_mi_var);
  curr_trajectory_var_index++;

  auto transform_keypoints = [&]() {
    const auto T_mr = T_rm_var->value().inverse().matrix();
#pragma omp parallel for num_threads(options_.num_threads)
    for (int jj = 0; jj < (int)keypoints.size(); jj++) {
      auto &keypoint = keypoints[jj];
      keypoint.pt = T_mr.block<3, 3>(0, 0) * keypoint.raw_pt + T_mr.block<3, 1>(0, 3);
    }
  };

  // For the N first frames, visit 2 voxels
  const short nb_voxels_visited = index_frame < options_.init_num_frames ? 2 : 1;
  const int kMinNumNeighbors = options_.min_number_neighbors;

  auto &current_estimate = trajectory_.at(index_frame);

  // timers
  std::vector<std::pair<std::string, std::unique_ptr<Stopwatch<>>>> timer;
  timer.emplace_back("Update Transform ............... ", std::make_unique<Stopwatch<>>(false));
  timer.emplace_back("Association .................... ", std::make_unique<Stopwatch<>>(false));
  timer.emplace_back("Optimization ................... ", std::make_unique<Stopwatch<>>(false));
  timer.emplace_back("Alignment ...................... ", std::make_unique<Stopwatch<>>(false));

  auto p2p_loss_func = [this]() -> BaseLossFunc::Ptr {
      switch (options_.p2p_loss_func) {
        case SteamLoCVOdometry::STEAM_LOSS_FUNC::L2:
          return L2LossFunc::MakeShared();
        case SteamLoCVOdometry::STEAM_LOSS_FUNC::DCS:
          return DcsLossFunc::MakeShared(options_.p2p_loss_sigma);
        case SteamLoCVOdometry::STEAM_LOSS_FUNC::CAUCHY:
          return CauchyLossFunc::MakeShared(options_.p2p_loss_sigma);
        case SteamLoCVOdometry::STEAM_LOSS_FUNC::GM:
          return GemanMcClureLossFunc::MakeShared(options_.p2p_loss_sigma);
        default:
          return nullptr;
      }
      return nullptr;
    }();

  // Transform points into the robot frame just once:
  timer[0].second->start();
  const Eigen::Matrix4d T_rs_mat = options_.T_sr.inverse();
#pragma omp parallel for num_threads(options_.num_threads)
  for (int i = 0; i < (int)keypoints.size(); i++) {
    auto &keypoint = keypoints[i];
    keypoint.raw_pt = T_rs_mat.block<3, 3>(0, 0) * keypoint.raw_pt + T_rs_mat.block<3, 1>(0, 3);
  }
  timer[0].second->stop();

  // De-skew points just once:
  std::set<double> unique_point_times_;
  for (const auto &keypoint : keypoints) {
    unique_point_times_.insert(keypoint.timestamp);
  }
  std::vector<double> unique_point_times(unique_point_times_.begin(), unique_point_times_.end());
  std::map<double, Eigen::Matrix4d> T_kj_cache_map;
  for (int jj = 0; jj < (int)unique_point_times.size(); jj++) {
    const auto &ts = unique_point_times[jj];
    const lgmath::se3::Transformation T_kj(Eigen::Matrix<double, 6, 1>(prev_w_mr_inr * (curr_time - ts)));
    T_kj_cache_map[ts] = T_kj.matrix();
  }
#pragma omp parallel for num_threads(options_.num_threads)
  for (int jj = 0; jj < (int)keypoints.size(); jj++) {
    auto &keypoint = keypoints[jj];
    const Eigen::Matrix4d &T_kj = T_kj_cache_map[keypoint.timestamp];
    keypoint.raw_pt = T_kj.block<3, 3>(0, 0) * keypoint.raw_pt + T_kj.block<3, 1>(0, 3);
  }

  int N_matches = 0;
  const auto noise_model = StaticNoiseModel<1>::MakeShared(Eigen::Matrix<double, 1, 1>::Identity(), NoiseType::INFORMATION);

  for (int iter(0); iter < options_.num_iters_icp; iter++) {
    timer[0].second->start();
    transform_keypoints();
    timer[0].second->stop();

    const auto problem = OptimizationProblem::MakeShared(options_.num_threads);
    for (const auto &var : steam_state_vars) problem->addStateVariable(var);
    meas_cost_terms.clear();
    meas_cost_terms.reserve(keypoints.size());

    timer[1].second->start();

#pragma omp declare reduction(merge_meas : std::vector<BaseCostTerm::ConstPtr> : omp_out.insert( \
        omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for num_threads(options_.num_threads) reduction(merge_meas : meas_cost_terms)
    for (int i = 0; i < (int)keypoints.size(); i++) {
      const auto &keypoint = keypoints[i];
      const auto &pt_keypoint = keypoint.pt;

      // Neighborhood search
      ArrayVector3d vector_neighbors =
          map_.searchNeighbors(pt_keypoint, nb_voxels_visited, options_.size_voxel_map, options_.max_number_neighbors);

      if ((int)vector_neighbors.size() < kMinNumNeighbors) {
        continue;
      }

      // Compute normals from neighbors
      auto neighborhood = compute_neighborhood_distribution(vector_neighbors);

      const double planarity_weight = std::pow(neighborhood.a2D, options_.power_planarity);
      const double weight = planarity_weight;

      const double dist_to_plane = std::abs((keypoint.pt - vector_neighbors[0]).transpose() * neighborhood.normal);
      if (dist_to_plane >= options_.p2p_max_dist)
        continue;

      Eigen::Vector3d closest_pt = vector_neighbors[0];
      Eigen::Vector3d closest_normal = weight * neighborhood.normal;

      const auto error_func = p2p::p2planeError(T_mr_eval, closest_pt, keypoint.raw_pt, closest_normal);
      error_func->setTime(Time(keypoint.timestamp));
      const auto cost = WeightedLeastSqCostTerm<1>::MakeShared(error_func, noise_model, p2p_loss_func);
      meas_cost_terms.emplace_back(cost);
    }

    N_matches = meas_cost_terms.size();
    for (const auto &cost : meas_cost_terms) problem->addCostTerm(cost);

    timer[1].second->stop();

    if (N_matches < options_.min_number_keypoints) {
      LOG(ERROR) << "Error: not enough keypoints selected!" << std::endl;
      LOG(ERROR) << "Number_of_residuals: " << N_matches << std::endl;
      icp_success = false;
      break;
    }

    timer[2].second->start();
    GaussNewtonSolverNVA::Params params;
    params.verbose = options_.verbose;
    params.max_iterations = (unsigned int)options_.max_iterations;
    if (iter >= 3 && options_.use_line_search)
      params.line_search = true;
    else
      params.line_search = false;
    GaussNewtonSolverNVA solver(*problem, params);
    solver.optimize();
    timer[2].second->stop();

    timer[3].second->start();
    // Compute break conditions for ICP
    double diff_trans = 0, diff_rot = 0;
    const auto begin_T_mr = T_mr_eval->evaluate().matrix();
    const auto begin_T_ms = begin_T_mr * options_.T_sr.inverse();
    diff_trans += (current_estimate.begin_t - begin_T_ms.block<3, 1>(0, 3)).norm();
    diff_rot += AngularDistance(current_estimate.begin_R, begin_T_ms.block<3, 3>(0, 0));
    const auto end_T_ms = begin_T_ms;
    const auto mid_T_ms = begin_T_ms;
    current_estimate.setMidPose(mid_T_ms);
    current_estimate.begin_R = begin_T_ms.block<3, 3>(0, 0);
    current_estimate.begin_t = begin_T_ms.block<3, 1>(0, 3);
    current_estimate.end_R = end_T_ms.block<3, 3>(0, 0);
    current_estimate.end_t = end_T_ms.block<3, 1>(0, 3);
    timer[3].second->stop();

    LOG(INFO) << "diff_trans(m): " << diff_trans << " diff_rot(deg): " << diff_rot << std::endl;

    if ((index_frame > 1) &&
        (diff_rot < options_.threshold_orientation_norm && diff_trans < options_.threshold_translation_norm)) {
      if (options_.debug_print) {
        LOG(INFO) << "ICP: Finished with N=" << iter << " iterations" << std::endl;
      }
      if (options_.break_icp_early) break;
    }
  }

  current_estimate.mid_w = prev_w_mr_inr;
  LOG(INFO) << "w(-1) " << prev_w_mr_inr.transpose() << std::endl;
  LOG(INFO) << "Number of keypoints used in ICP: " << N_matches << std::endl;

  /// Debug print
  if (options_.debug_print) {
    for (size_t i = 0; i < timer.size(); i++)
      LOG(INFO) << "Elapsed " << timer[i].first << *(timer[i].second) << std::endl;
    LOG(INFO) << "Number iterations ICP: " << options_.num_iters_icp << std::endl;
    LOG(INFO) << "Translation Begin: " << trajectory_[index_frame].begin_t.transpose() << std::endl;
    LOG(INFO) << "Translation End: " << trajectory_[index_frame].end_t.transpose() << std::endl;
  }

  return icp_success;
}

}  // namespace steam_icp
