#include "steam_icp/odometry/steam_rio.hpp"

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

}  // namespace

SteamRoOdometry::SteamRoOdometry(const Options &options) : Odometry(options), options_(options) {
  // iniitalize steam vars
  T_sr_var_ = steam::se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(options_.T_sr));
  T_sr_var_->locked() = true;
  lgmath::se3::Transformation T_mi;
  T_mi_var_ = steam::se3::SE3StateVar::MakeShared(T_mi);
  T_mi_var_->locked() = true;
  sliding_window_filter_ = steam::SlidingWindowFilter::MakeShared(options_.num_threads);
}

SteamRoOdometry::~SteamRoOdometry() {
  using namespace steam::traj;

  std::ofstream trajectory_file;
  auto now = std::chrono::system_clock::now();
  auto utc = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
  trajectory_file.open(options_.debug_path + "/trajectory_" + std::to_string(utc) + ".txt", std::ios::out);
  // trajectory_file.open(options_.debug_path + "/trajectory.txt", std::ios::out);

  LOG(INFO) << "Building full trajectory." << std::endl;
  auto full_trajectory = steam::traj::const_vel::Interface::MakeShared(options_.qc_diag);
  for (auto &var : trajectory_vars_) {
    full_trajectory->add(var.time, var.T_rm, var.w_mr_inr);
  }

  LOG(INFO) << "Dumping trajectory." << std::endl;
  double begin_time = trajectory_.front().begin_timestamp;
  double end_time = trajectory_.back().end_timestamp;
  double dt = 0.01;
  for (double time = begin_time; time <= end_time; time += dt) {
    Time steam_time(time);
    //
    const auto T_rm = full_trajectory->getPoseInterpolator(steam_time)->evaluate().matrix();
    const auto w_mr_inr = full_trajectory->getVelocityInterpolator(steam_time)->evaluate();
    // clang-format off
    trajectory_file << std::fixed << std::setprecision(12) << (0.0) << " " << steam_time.nanosecs() << " "
                    << T_rm(0, 0) << " " << T_rm(0, 1) << " " << T_rm(0, 2) << " " << T_rm(0, 3) << " "
                    << T_rm(1, 0) << " " << T_rm(1, 1) << " " << T_rm(1, 2) << " " << T_rm(1, 3) << " "
                    << T_rm(2, 0) << " " << T_rm(2, 1) << " " << T_rm(2, 2) << " " << T_rm(2, 3) << " "
                    << T_rm(3, 0) << " " << T_rm(3, 1) << " " << T_rm(3, 2) << " " << T_rm(3, 3) << " "
                    << w_mr_inr(0) << " " << w_mr_inr(1) << " " << w_mr_inr(2) << " " << w_mr_inr(3) << " "
                    << w_mr_inr(4) << " " << w_mr_inr(5) << std::endl;
    // clang-format on
  }
  LOG(INFO) << "Dumping trajectory. - DONE" << std::endl;
}

Trajectory SteamRoOdometry::trajectory() {
  if (options_.use_final_state_value) {
    LOG(INFO) << "Building full trajectory." << std::endl;
    auto full_trajectory = steam::traj::const_vel::Interface::MakeShared(options_.qc_diag);
    for (auto &var : trajectory_vars_) {
      full_trajectory->add(var.time, var.T_rm, var.w_mr_inr);
    }

    LOG(INFO) << "Updating trajectory." << std::endl;
    using namespace steam::se3;
    using namespace steam::traj;
    for (auto &frame : trajectory_) {
      Time begin_steam_time(frame.begin_timestamp);
      const auto begin_T_mr = inverse(full_trajectory->getPoseInterpolator(begin_steam_time))->evaluate().matrix();
      const auto begin_T_ms = begin_T_mr * options_.T_sr.inverse();
      frame.begin_R = begin_T_ms.block<3, 3>(0, 0);
      frame.begin_t = begin_T_ms.block<3, 1>(0, 3);

      Time mid_steam_time(static_cast<double>(frame.getEvalTime()));
      const auto mid_T_mr = inverse(full_trajectory->getPoseInterpolator(mid_steam_time))->evaluate().matrix();
      const auto mid_T_ms = mid_T_mr * options_.T_sr.inverse();
      frame.setMidPose(mid_T_ms);

      Time end_steam_time(frame.end_timestamp);
      const auto end_T_mr = inverse(full_trajectory->getPoseInterpolator(end_steam_time))->evaluate().matrix();
      const auto end_T_ms = end_T_mr * options_.T_sr.inverse();
      frame.end_R = end_T_ms.block<3, 3>(0, 0);
      frame.end_t = end_T_ms.block<3, 1>(0, 3);
    }
  }
  return trajectory_;
}

auto SteamRoOdometry::registerFrame(const DataFrame &const_frame) -> RegistrationSummary {
  RegistrationSummary summary;

  // add a new frame
  int index_frame = trajectory_.size();
  trajectory_.emplace_back();

  //
  initializeTimestamp(index_frame, const_frame);

  //
  initializeMotion(index_frame);

  //
  auto frame = initializeFrame(index_frame, const_frame.pointcloud);

  //
  std::vector<Point3D> keypoints(frame);
  if (index_frame > 0) {
    double sample_voxel_size =
        index_frame < options_.init_num_frames ? options_.init_sample_voxel_size : options_.sample_voxel_size;

    // downsample
    if (options_.voxel_downsample) grid_sampling(frame, keypoints, sample_voxel_size, options_.num_threads);

    // icp
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
    Eigen::Matrix<double, 6, 1> w_mr_inr = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 1> b_zero = Eigen::Matrix<double, 6, 1>::Zero();

    // initialize frame (the beginning of the trajectory)
    const double begin_time = trajectory_[index_frame].begin_timestamp;
    Time begin_steam_time(begin_time);
    const auto begin_T_rm_var = SE3StateVar::MakeShared(T_rm);
    const auto begin_w_mr_inr_var = VSpaceStateVar<6>::MakeShared(w_mr_inr);
    const auto begin_b_var = VSpaceStateVar<6>::MakeShared(b_zero);
    trajectory_vars_.emplace_back(begin_steam_time, begin_T_rm_var, begin_w_mr_inr_var, begin_b_var);

    // the end of current scan (this is the first state that could be optimized)
    const double end_time = trajectory_[index_frame].end_timestamp;
    Time end_steam_time(end_time);
    const auto end_T_rm_var = SE3StateVar::MakeShared(T_rm);
    const auto end_w_mr_inr_var = VSpaceStateVar<6>::MakeShared(w_mr_inr);
    const auto end_b_var = VSpaceStateVar<6>::MakeShared(b_zero);
    trajectory_vars_.emplace_back(end_steam_time, end_T_rm_var, end_w_mr_inr_var, end_b_var);
    to_marginalize_ = 1;  /// The first state is not added to the filter

    trajectory_[index_frame].end_T_rm_cov = Eigen::Matrix<double, 6, 6>::Identity() * 1e-4;
    trajectory_[index_frame].end_w_mr_inr_cov = Eigen::Matrix<double, 6, 6>::Identity() * 1e-4;
    trajectory_[index_frame].end_state_cov = Eigen::Matrix<double, 18, 18>::Identity() * 1e-4;

    summary.success = true;
  }
  trajectory_[index_frame].points = frame;

  // add points
  if (index_frame == 0) {
    updateMap(index_frame, index_frame);
  } else if ((index_frame - options_.delay_adding_points) > 0) {
    updateMap(index_frame, (index_frame - options_.delay_adding_points));
  }

  summary.corrected_points = keypoints;

#if false
  // correct all points
  summary.all_corrected_points = const_frame;
  auto q_begin = Eigen::Quaterniond(trajectory_[index_frame].begin_R);
  auto q_end = Eigen::Quaterniond(trajectory_[index_frame].end_R);
  Eigen::Vector3d t_begin = trajectory_[index_frame].begin_t;
  Eigen::Vector3d t_end = trajectory_[index_frame].end_t;
  for (auto &point : summary.all_corrected_points) {
    double alpha_timestamp = point.alpha_timestamp;
    Eigen::Matrix3d R = q_begin.slerp(alpha_timestamp, q_end).normalized().toRotationMatrix();
    Eigen::Vector3d t = (1.0 - alpha_timestamp) * t_begin + alpha_timestamp * t_end;
    point.pt = R * point.raw_pt + t;
  }
#endif

  summary.R_ms = trajectory_[index_frame].end_R;
  summary.t_ms = trajectory_[index_frame].end_t;

  return summary;
}

void SteamRoOdometry::initializeTimestamp(int index_frame, const DataFrame &const_frame) {
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

void SteamRoOdometry::initializeMotion(int index_frame) {
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

std::vector<Point3D> SteamRoOdometry::initializeFrame(int index_frame, const std::vector<Point3D> &const_frame) {
  std::vector<Point3D> frame(const_frame);

  if (options_.voxel_downsample) {
    double sample_size = index_frame < options_.init_num_frames ? options_.init_voxel_size : options_.voxel_size;
    std::mt19937_64 g;
    std::shuffle(frame.begin(), frame.end(), g);
    // Subsample the scan with voxels taking one random in every voxel
    sub_sample_frame(frame, sample_size, options_.num_threads);
    std::shuffle(frame.begin(), frame.end(), g);
  }
  // initialize points
  auto q_begin = Eigen::Quaterniond(trajectory_[index_frame].begin_R);
  auto q_end = Eigen::Quaterniond(trajectory_[index_frame].end_R);
  Eigen::Vector3d t_begin = trajectory_[index_frame].begin_t;
  Eigen::Vector3d t_end = trajectory_[index_frame].end_t;
#pragma omp parallel for num_threads(options_.num_threads)
  for (unsigned int i = 0; i < frame.size(); ++i) {
    auto &point = frame[i];
    double alpha_timestamp = point.alpha_timestamp;
    Eigen::Matrix3d R = q_begin.slerp(alpha_timestamp, q_end).normalized().toRotationMatrix();
    Eigen::Vector3d t = (1.0 - alpha_timestamp) * t_begin + alpha_timestamp * t_end;
    //
    point.pt = R * point.raw_pt + t;
  }

  return frame;
}

void SteamRoOdometry::updateMap(int index_frame, int update_frame) {
  const double kSizeVoxelMap = options_.size_voxel_map;
  const double kMinDistancePoints = options_.min_distance_points;
  const int kMaxNumPointsInVoxel = options_.max_num_points_in_voxel;

  // update frame
  auto &frame = trajectory_[update_frame].points;
#if false
  auto q_begin = Eigen::Quaterniond(trajectory_[update_frame].begin_R);
  auto q_end = Eigen::Quaterniond(trajectory_[update_frame].end_R);
  Eigen::Vector3d t_begin = trajectory_[update_frame].begin_t;
  Eigen::Vector3d t_end = trajectory_[update_frame].end_t;
  for (auto &point : frame) {
    // modifies the world point of the frame based on the raw_pt
    double alpha_timestamp = point.alpha_timestamp;
    Eigen::Matrix3d R = q_begin.slerp(alpha_timestamp, q_end).normalized().toRotationMatrix();
    Eigen::Vector3d t = (1.0 - alpha_timestamp) * t_begin + alpha_timestamp * t_end;
    //
    point.pt = R * point.raw_pt + t;
  }
#else
  using namespace steam::se3;
  using namespace steam::traj;

  Time begin_steam_time = trajectory_[update_frame].begin_timestamp;
  Time end_steam_time = trajectory_[update_frame].end_timestamp;

  // construct the trajectory for interpolation
  int num_states = 0;
  const auto update_trajectory = const_vel::Interface::MakeShared(options_.qc_diag);
  for (size_t i = (to_marginalize_ - 1); i < trajectory_vars_.size(); i++) {
    const auto &var = trajectory_vars_.at(i);
    update_trajectory->add(var.time, var.T_rm, var.w_mr_inr);
    num_states++;
    if (var.time == end_steam_time) break;
    if (var.time > end_steam_time) throw std::runtime_error("var.time > end_steam_time, should not happen");
  }

  LOG(INFO) << "Adding points to map between (inclusive): " << begin_steam_time.seconds() << " - "
            << end_steam_time.seconds() << ", with num states: " << num_states << std::endl;

  std::set<double> unique_point_times_;
  for (const auto &point : frame) {
    unique_point_times_.insert(point.timestamp);
  }
  std::vector<double> unique_point_times(unique_point_times_.begin(), unique_point_times_.end());

  std::map<double, std::pair<Eigen::Matrix4d, Eigen::Vector3d>> T_ms_cache_map;
  const Eigen::Matrix4d T_rs = options_.T_sr.inverse();
#pragma omp parallel for num_threads(options_.num_threads)
  for (int jj = 0; jj < (int)unique_point_times.size(); jj++) {
    const auto &ts = unique_point_times[jj];
    const auto T_rm_intp_eval = update_trajectory->getPoseInterpolator(Time(ts));
    const Eigen::Matrix4d T_ms = T_rm_intp_eval->value().inverse().matrix() * T_rs;
    const auto w_mr_inr_intp_eval = update_trajectory->getVelocityInterpolator(Time(ts));
    const auto v_m_s_in_s = compose_velocity(T_sr_var_, w_mr_inr_intp_eval)->value().block<3, 1>(0, 0);
    auto pair = std::make_pair(T_ms, v_m_s_in_s);
#pragma omp critical
    T_ms_cache_map[ts] = pair;
  }

#pragma omp parallel for num_threads(options_.num_threads)
  for (unsigned i = 0; i < frame.size(); i++) {
    // const double query_time = frame[i].timestamp;

    // const auto T_rm_intp_eval = update_trajectory->getPoseInterpolator(Time(query_time));
    // const auto T_ms_intp_eval = inverse(compose(T_sr_var_, T_rm_intp_eval));

    // const auto w_mr_inr_intp_eval = update_trajectory->getVelocityInterpolator(Time(query_time));
    // const auto w_ms_ins_intp_eval = compose_velocity(T_sr_var_, w_mr_inr_intp_eval);

    // const Eigen::Matrix4d T_ms = T_ms_intp_eval->evaluate().matrix();
    // const Eigen::Matrix3d R = T_ms.block<3, 3>(0, 0);
    // const Eigen::Vector3d t = T_ms.block<3, 1>(0, 3);

    const Eigen::Matrix4d &T_ms = T_ms_cache_map[frame[i].timestamp].first;

    //
    if (options_.beta != 0) {
      const auto abar = frame[i].raw_pt.normalized();
      // const auto v_m_s_in_s = w_ms_ins_intp_eval->evaluate().matrix().block<3, 1>(0, 0);
      const Eigen::Vector3d &v_m_s_in_s = T_ms_cache_map[frame[i].timestamp].second;
      frame[i].pt = T_ms.block<3, 3>(0, 0) * (frame[i].raw_pt - options_.beta * abar * abar.transpose() * v_m_s_in_s) +
                    T_ms.block<3, 1>(0, 3);
    } else {
      frame[i].pt = T_ms.block<3, 3>(0, 0) * frame[i].raw_pt + T_ms.block<3, 1>(0, 3);
    }
  }
#endif

  // map_.clear();
  // update the map with new points and refresh their life time and normal
  map_.add(frame, kSizeVoxelMap, kMaxNumPointsInVoxel, kMinDistancePoints);
  if (options_.filter_lifetimes) map_.update_and_filter_lifetimes();
  frame.clear();
  frame.shrink_to_fit();

  // remove points
  const double kMaxDistance = options_.max_distance;
  const Eigen::Vector3d location = trajectory_[index_frame].end_t;
  map_.remove(location, kMaxDistance);
}

bool SteamRoOdometry::icp(int index_frame, std::vector<Point3D> &keypoints,
                          const std::vector<steam::IMUData> &imu_data_vec) {
  using namespace steam;
  using namespace steam::se3;
  using namespace steam::traj;
  using namespace steam::vspace;
  using namespace steam::imu;

  bool icp_success = true;

  ///
  const auto steam_trajectory = const_vel::Interface::MakeShared(options_.qc_diag);
  std::vector<StateVarBase::Ptr> steam_state_vars;
  std::vector<BaseCostTerm::ConstPtr> prior_cost_terms;
  std::vector<BaseCostTerm::ConstPtr> meas_cost_terms;
  std::vector<BaseCostTerm::ConstPtr> imu_cost_terms;
  std::vector<BaseCostTerm::ConstPtr> imu_prior_cost_terms;
  std::vector<BaseCostTerm::ConstPtr> preint_cost_terms;
  const size_t prev_trajectory_var_index = trajectory_vars_.size() - 1;
  size_t curr_trajectory_var_index = trajectory_vars_.size() - 1;

  /// use previous trajectory to initialize steam state variables
  LOG(INFO) << "[CT_ICP_STEAM] prev scan end time: " << trajectory_[index_frame - 1].end_timestamp << std::endl;
  const double prev_time = trajectory_[index_frame - 1].end_timestamp;
  if (trajectory_vars_.back().time != Time(static_cast<double>(prev_time)))
    throw std::runtime_error{"missing previous scan end variable"};
  Time prev_steam_time = trajectory_vars_.back().time;
  lgmath::se3::Transformation prev_T_rm = trajectory_vars_.back().T_rm->value();
  Eigen::Matrix<double, 6, 1> prev_w_mr_inr = trajectory_vars_.back().w_mr_inr->value();
  Eigen::Matrix<double, 6, 1> prev_imu_biases = trajectory_vars_.back().imu_biases->value();
  const auto prev_T_rm_var = trajectory_vars_.back().T_rm;
  const auto prev_w_mr_inr_var = trajectory_vars_.back().w_mr_inr;
  const auto prev_imu_biases_var = trajectory_vars_.back().imu_biases;
  steam_trajectory->add(prev_steam_time, prev_T_rm_var, prev_w_mr_inr_var);
  steam_state_vars.emplace_back(prev_T_rm_var);
  steam_state_vars.emplace_back(prev_w_mr_inr_var);
  if (options_.use_imu) steam_state_vars.emplace_back(prev_imu_biases_var);

  /// New state for this frame
  LOG(INFO) << "[CT_ICP_STEAM] curr scan end time: " << trajectory_[index_frame].end_timestamp << std::endl;
  LOG(INFO) << "[CT_ICP_STEAM] total num new states: " << (options_.num_extra_states + 1) << std::endl;
  const double curr_time = trajectory_[index_frame].end_timestamp;
  const int num_states = options_.num_extra_states + 1;
  const double time_diff = (curr_time - prev_time) / static_cast<double>(num_states);
  std::vector<double> knot_times;
  knot_times.reserve(num_states);
  for (int i = 0; i < options_.num_extra_states; ++i) {
    knot_times.emplace_back(prev_time + (double)(i + 1) * time_diff);
  }
  knot_times.emplace_back(curr_time);

  // add new state variables, initialize with constant velocity
  for (size_t i = 0; i < knot_times.size(); ++i) {
    double knot_time = knot_times[i];
    Time knot_steam_time(knot_time);
    //
    const Eigen::Matrix<double, 6, 1> xi_mr_inr_odo((knot_steam_time - prev_steam_time).seconds() * prev_w_mr_inr);
    const auto knot_T_rm = lgmath::se3::Transformation(xi_mr_inr_odo) * prev_T_rm;
    const auto T_rm_var = SE3StateVar::MakeShared(knot_T_rm);
    const auto w_mr_inr_var = VSpaceStateVar<6>::MakeShared(prev_w_mr_inr);
    const auto imu_biases_var = VSpaceStateVar<6>::MakeShared(prev_imu_biases);
    //
    steam_trajectory->add(knot_steam_time, T_rm_var, w_mr_inr_var);
    steam_state_vars.emplace_back(T_rm_var);
    steam_state_vars.emplace_back(w_mr_inr_var);

    if (options_.use_imu) {
      steam_state_vars.emplace_back(imu_biases_var);
    }

    // cache the end state in full steam trajectory because it will be used again
    trajectory_vars_.emplace_back(knot_steam_time, T_rm_var, w_mr_inr_var, imu_biases_var);
    curr_trajectory_var_index++;
  }

  if (index_frame == 1) {
    const auto &prev_var = trajectory_vars_.at(prev_trajectory_var_index);
    /// add a prior to state at the beginning
    lgmath::se3::Transformation T_rm;
    Eigen::Matrix<double, 6, 1> w_mr_inr = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 12, 12> state_cov = Eigen::Matrix<double, 12, 12>::Identity() * 1e-4;
    steam_trajectory->addStatePrior(prev_var.time, T_rm, w_mr_inr, state_cov);
    if (prev_var.time != Time(trajectory_.at(0).end_timestamp)) throw std::runtime_error{"inconsistent timestamp"};
  }

  if (options_.use_imu && index_frame == 1) {
    Eigen::Matrix<double, 6, 1> b_zero = Eigen::Matrix<double, 6, 1>::Zero();
    const auto &prev_var = trajectory_vars_.at(prev_trajectory_var_index);
    // add a prior to imu bias at the beginning
    Eigen::Matrix<double, 6, 6> init_bias_cov = Eigen::Matrix<double, 6, 6>::Identity();
    init_bias_cov.block<3, 3>(0, 0).diagonal() = options_.p0_bias_accel;
    init_bias_cov.block<3, 3>(3, 3) = Eigen::Matrix<double, 3, 3>::Identity() * options_.p0_bias_gyro;
    auto bias_error = vspace::vspace_error<6>(prev_var.imu_biases, b_zero);
    auto noise_model = StaticNoiseModel<6>::MakeShared(init_bias_cov);
    auto loss_func = L2LossFunc::MakeShared();
    const auto bias_prior_factor = WeightedLeastSqCostTerm<6>::MakeShared(bias_error, noise_model, loss_func);
    imu_prior_cost_terms.emplace_back(bias_prior_factor);
  }

  /// update sliding window variables
  {
    //
    if (index_frame == 1) {
      const auto &prev_var = trajectory_vars_.at(prev_trajectory_var_index);
      sliding_window_filter_->addStateVariable(std::vector<StateVarBase::Ptr>{prev_var.T_rm, prev_var.w_mr_inr});
      if (options_.use_imu) {
        sliding_window_filter_->addStateVariable(std::vector<StateVarBase::Ptr>{prev_var.imu_biases});
      }
    }

    //
    for (size_t i = prev_trajectory_var_index + 1; i <= curr_trajectory_var_index; ++i) {
      const auto &var = trajectory_vars_.at(i);
      sliding_window_filter_->addStateVariable(std::vector<StateVarBase::Ptr>{var.T_rm, var.w_mr_inr});
      if (options_.use_imu) {
        sliding_window_filter_->addStateVariable(std::vector<StateVarBase::Ptr>{var.imu_biases});
      }
    }

    //
    if ((index_frame - options_.delay_adding_points) > 0) {
      //
      const double begin_marg_time = trajectory_vars_.at(to_marginalize_).time.seconds();
      double end_marg_time = trajectory_vars_.at(to_marginalize_).time.seconds();
      std::vector<StateVarBase::Ptr> marg_vars;
      int num_states = 0;
      //
      double marg_time = trajectory_.at(index_frame - options_.delay_adding_points - 1).end_timestamp;
      Time marg_steam_time(marg_time);
      for (size_t i = to_marginalize_; i <= curr_trajectory_var_index; ++i) {
        const auto &var = trajectory_vars_.at(i);
        if (var.time <= marg_steam_time) {
          end_marg_time = var.time.seconds();
          marg_vars.emplace_back(var.T_rm);
          marg_vars.emplace_back(var.w_mr_inr);
          if (options_.use_imu) {
            marg_vars.emplace_back(var.imu_biases);
          }
          num_states++;
        } else {
          to_marginalize_ = i;
          break;
        }
      }
      sliding_window_filter_->marginalizeVariable(marg_vars);
      //
      LOG(INFO) << "Marginalizing time (inclusive): " << begin_marg_time << " - " << end_marg_time
                << ", with num states: " << num_states << std::endl;
    }
  }

  auto imu_options = PreintAccCostTerm::Options();
  imu_options.num_threads = options_.num_threads;
  imu_options.loss_sigma = options_.acc_loss_sigma;
  if (options_.acc_loss_func == "L2") imu_options.loss_func = PreintAccCostTerm::LOSS_FUNC::L2;
  if (options_.acc_loss_func == "DCS") imu_options.loss_func = PreintAccCostTerm::LOSS_FUNC::DCS;
  if (options_.acc_loss_func == "CAUCHY") imu_options.loss_func = PreintAccCostTerm::LOSS_FUNC::CAUCHY;
  if (options_.acc_loss_func == "GM") imu_options.loss_func = PreintAccCostTerm::LOSS_FUNC::GM;
  imu_options.gravity(2, 0) = options_.gravity;
  imu_options.r_imu_acc = options_.r_imu_acc;
  imu_options.se2 = true;

  const auto preint_cost_term = PreintAccCostTerm::MakeShared(
      steam_trajectory, prev_steam_time, knot_times.back(), trajectory_vars_[prev_trajectory_var_index].imu_biases,
      trajectory_vars_[prev_trajectory_var_index + 1].imu_biases, T_mi_var_, T_mi_var_, imu_options);

  if (options_.use_imu) {
    if (options_.use_accel) {
      preint_cost_term->set(imu_data_vec);
      preint_cost_term->init();
    }
    imu_cost_terms.reserve(imu_data_vec.size());
    Eigen::Matrix<double, 1, 1> R_ang = Eigen::Matrix<double, 1, 1>::Identity() * options_.r_imu_ang;
    const auto gyro_noise_model = StaticNoiseModel<1>::MakeShared(R_ang);
    const auto gyro_loss_func = CauchyLossFunc::MakeShared(1.0);
    for (const auto &imu_data : imu_data_vec) {
      size_t i = prev_trajectory_var_index;
      for (; i < trajectory_vars_.size() - 1; i++) {
        if (imu_data.timestamp >= trajectory_vars_[i].time.seconds() &&
            imu_data.timestamp < trajectory_vars_[i + 1].time.seconds())
          break;
      }
      if (imu_data.timestamp < trajectory_vars_[i].time.seconds() ||
          imu_data.timestamp >= trajectory_vars_[i + 1].time.seconds())
        throw std::runtime_error("imu stamp not within knot times");

      const auto bias_intp_eval = VSpaceInterpolator<6>::MakeShared(
          Time(imu_data.timestamp), trajectory_vars_[i].imu_biases, trajectory_vars_[i].time,
          trajectory_vars_[i + 1].imu_biases, trajectory_vars_[i + 1].time);

      const auto w_mr_inr_intp_eval = steam_trajectory->getVelocityInterpolator(Time(imu_data.timestamp));
      const auto gyro_error_func = imu::GyroErrorSE2(w_mr_inr_intp_eval, bias_intp_eval, imu_data.ang_vel);
      const auto gyro_cost = WeightedLeastSqCostTerm<1>::MakeShared(gyro_error_func, gyro_noise_model, gyro_loss_func);
      imu_cost_terms.emplace_back(gyro_cost);
    }

    {
      Eigen::Matrix<double, 6, 6> bias_cov = Eigen::Matrix<double, 6, 6>::Identity();
      bias_cov.block<3, 3>(0, 0).diagonal() = options_.q_bias_accel;
      bias_cov.block<3, 3>(3, 3) = Eigen::Matrix<double, 3, 3>::Identity() * options_.q_bias_gyro;
      auto noise_model = StaticNoiseModel<6>::MakeShared(bias_cov);
      auto loss_func = L2LossFunc::MakeShared();
      size_t i = prev_trajectory_var_index;
      for (; i < trajectory_vars_.size() - 1; i++) {
        const auto nbk = vspace::NegationEvaluator<6>::MakeShared(trajectory_vars_[i + 1].imu_biases);
        auto bias_error = vspace::AdditionEvaluator<6>::MakeShared(trajectory_vars_[i].imu_biases, nbk);
        const auto bias_prior_factor = WeightedLeastSqCostTerm<6>::MakeShared(bias_error, noise_model, loss_func);
        imu_prior_cost_terms.emplace_back(bias_prior_factor);
      }
    }
  }
  // Get evaluator for query points
  std::set<double> unique_point_times_;
  for (const auto &keypoint : keypoints) {
    unique_point_times_.insert(keypoint.timestamp);
  }
  std::vector<double> unique_point_times(unique_point_times_.begin(), unique_point_times_.end());

  interp_mats_.clear();

  const auto &time1 = prev_steam_time.seconds();
  const auto &time2 = knot_times.back();
  const double T = time2 - time1;
  const Eigen::Matrix<double, 6, 1> ones = Eigen::Matrix<double, 6, 1>::Ones();
  const auto Qinv_T = steam::traj::const_vel::getQinv(T, ones);
  const auto Tran_T = steam::traj::const_vel::getTran(T);
#pragma omp parallel for num_threads(options_.num_threads)
  for (unsigned int i = 0; i < unique_point_times.size(); ++i) {
    const double &time = unique_point_times[i];
    const double tau = time - time1;
    const double kappa = time2 - time;
    const Matrix12d Q_tau = steam::traj::const_vel::getQ(tau, ones);
    const Matrix12d Tran_kappa = steam::traj::const_vel::getTran(kappa);
    const Matrix12d Tran_tau = steam::traj::const_vel::getTran(tau);
    const Matrix12d omega = (Q_tau * Tran_kappa.transpose() * Qinv_T);
    const Matrix12d lambda = (Tran_tau - omega * Tran_T);
#pragma omp critical
    interp_mats_.emplace(time, std::make_pair(omega, lambda));
  }

  // std::vector<Evaluable<const_vel::Interface::PoseType>::ConstPtr> T_ms_intp_eval_vec;
  // std::vector<Evaluable<const_vel::Interface::VelocityType>::ConstPtr> w_ms_ins_intp_eval_vec;
  // T_ms_intp_eval_vec.reserve(keypoints.size());
  // w_ms_ins_intp_eval_vec.reserve(keypoints.size());
  // for (const auto &keypoint : keypoints) {
  //   const double query_time = keypoint.timestamp;
  //   // pose
  //   const auto T_rm_intp_eval = steam_trajectory->getPoseInterpolator(Time(query_time));
  //   const auto T_ms_intp_eval = inverse(compose(T_sr_var_, T_rm_intp_eval));
  //   T_ms_intp_eval_vec.emplace_back(T_ms_intp_eval);
  //   // velocity
  //   const auto w_mr_inr_intp_eval = steam_trajectory->getVelocityInterpolator(Time(query_time));
  //   const auto w_ms_ins_intp_eval = compose_velocity(T_sr_var_, w_mr_inr_intp_eval);
  //   w_ms_ins_intp_eval_vec.emplace_back(w_ms_ins_intp_eval);
  // }

  // For the 50 first frames, visit 2 voxels
  const short nb_voxels_visited = index_frame < options_.init_num_frames ? 2 : 1;
  // const int kMinNumNeighbors = options_.min_number_neighbors;

  auto &current_estimate = trajectory_.at(index_frame);

  // timers
  std::vector<std::pair<std::string, std::unique_ptr<Stopwatch<>>>> timer;
  timer.emplace_back("Update Transform ............... ", std::make_unique<Stopwatch<>>(false));
  timer.emplace_back("Association .................... ", std::make_unique<Stopwatch<>>(false));
  timer.emplace_back("Optimization ................... ", std::make_unique<Stopwatch<>>(false));
  timer.emplace_back("Alignment ...................... ", std::make_unique<Stopwatch<>>(false));
  timer.emplace_back("Sliding Window ................. ", std::make_unique<Stopwatch<>>(false));
  std::vector<std::pair<std::string, std::unique_ptr<Stopwatch<>>>> inner_timer;
  inner_timer.emplace_back("Search Neighbors ............. ", std::make_unique<Stopwatch<>>(false));
  inner_timer.emplace_back("Compute Normal ............... ", std::make_unique<Stopwatch<>>(false));
  inner_timer.emplace_back("Add Cost Term ................ ", std::make_unique<Stopwatch<>>(false));
  bool innerloop_time = (options_.num_threads == 1);

  auto transform_keypoints = [&]() {
    const auto knot1 = steam_trajectory->get(prev_steam_time);
    const auto knot2 = steam_trajectory->get(knot_times.back());
    const auto T1 = knot1->pose()->value();
    const auto w1 = knot1->velocity()->value();
    const auto T2 = knot2->pose()->value();
    const auto w2 = knot2->velocity()->value();

    const auto xi_21 = (T2 / T1).vec();
    const Eigen::Matrix<double, 6, 6> J_21_inv = lgmath::se3::vec2jacinv(xi_21);
    const auto J_21_inv_w2 = J_21_inv * w2;

    std::map<double, std::pair<Eigen::Matrix4d, Eigen::Vector3d>> T_ms_cache_map;
    const Eigen::Matrix4d T_rs = options_.T_sr.inverse();
    const auto Ad_T_s_r = lgmath::se3::tranAd(options_.T_sr);
#pragma omp parallel for num_threads(options_.num_threads)
    for (int jj = 0; jj < (int)unique_point_times.size(); jj++) {
      const auto &ts = unique_point_times[jj];
      const auto &omega = interp_mats_.at(ts).first;
      const auto &lambda = interp_mats_.at(ts).second;
      const Eigen::Matrix<double, 6, 1> xi_i1 =
          lambda.block<6, 6>(0, 6) * w1 + omega.block<6, 6>(0, 0) * xi_21 + omega.block<6, 6>(0, 6) * J_21_inv_w2;
      const Eigen::Matrix<double, 6, 1> xi_j1 =
          lambda.block<6, 6>(6, 6) * w1 + omega.block<6, 6>(6, 0) * xi_21 + omega.block<6, 6>(6, 6) * J_21_inv_w2;
      const lgmath::se3::Transformation T_i1(xi_i1);
      const lgmath::se3::Transformation T_i0 = T_i1 * T1;
      const Eigen::Matrix4d T_ms = T_i0.inverse().matrix() * T_rs;
      const Eigen::Vector3d v_m_s_in_s = (Ad_T_s_r * lgmath::se3::vec2jac(xi_i1) * xi_j1).block<3, 1>(0, 0);
      auto pair = std::make_pair(T_ms, v_m_s_in_s);
#pragma omp critical
      T_ms_cache_map[ts] = pair;
    }

#pragma omp parallel for num_threads(options_.num_threads)
    for (int jj = 0; jj < (int)keypoints.size(); jj++) {
      auto &keypoint = keypoints[jj];
      const Eigen::Matrix4d &T_ms = T_ms_cache_map[keypoint.timestamp].first;
      if (options_.beta != 0) {
        const auto abar = keypoint.raw_pt.normalized();
        const Eigen::Vector3d &v_m_s_in_s = T_ms_cache_map[keypoint.timestamp].second;
        keypoint.pt =
            T_ms.block<3, 3>(0, 0) * (keypoint.raw_pt - options_.beta * abar * abar.transpose() * v_m_s_in_s) +
            T_ms.block<3, 1>(0, 3);
      } else {
        keypoint.pt = T_ms.block<3, 3>(0, 0) * keypoint.raw_pt + T_ms.block<3, 1>(0, 3);
      }
    }
  };

  //   auto transform_keypoints = [&]() {
  // #pragma omp parallel for num_threads(options_.num_threads)
  //     for (int i = 0; i < (int)keypoints.size(); i++) {
  //       auto &keypoint = keypoints[i];
  //       const auto &T_ms_intp_eval = T_ms_intp_eval_vec[i];
  //       const auto &w_ms_ins_intp_eval = w_ms_ins_intp_eval_vec[i];
  //       const auto T_ms = T_ms_intp_eval->evaluate().matrix();
  //       if (options_.beta != 0) {
  //         const auto abar = keypoint.raw_pt.normalized();
  //         const auto v_m_s_in_s = w_ms_ins_intp_eval->evaluate().matrix().block<3, 1>(0, 0);
  //         keypoint.pt =
  //             T_ms.block<3, 3>(0, 0) * (keypoint.raw_pt - options_.beta * abar * abar.transpose() * v_m_s_in_s) +
  //             T_ms.block<3, 1>(0, 3);
  //       } else {
  //         keypoint.pt = T_ms.block<3, 3>(0, 0) * keypoint.raw_pt + T_ms.block<3, 1>(0, 3);
  //       }
  //     }
  //   };

#define USE_P2P_SUPER_COST_TERM true

  auto p2p_options = P2PDopplerCVSuperCostTerm::Options();
  p2p_options.num_threads = options_.num_threads;
  p2p_options.p2p_loss_sigma = options_.p2p_loss_sigma;
  p2p_options.beta = options_.beta;
  p2p_options.T_sr = options_.T_sr;
  if (options_.p2p_loss_func == SteamRoOdometry::STEAM_LOSS_FUNC::L2)
    p2p_options.p2p_loss_func = P2PDopplerCVSuperCostTerm::LOSS_FUNC::L2;
  if (options_.p2p_loss_func == SteamRoOdometry::STEAM_LOSS_FUNC::DCS)
    p2p_options.p2p_loss_func = P2PDopplerCVSuperCostTerm::LOSS_FUNC::DCS;
  if (options_.p2p_loss_func == SteamRoOdometry::STEAM_LOSS_FUNC::CAUCHY)
    p2p_options.p2p_loss_func = P2PDopplerCVSuperCostTerm::LOSS_FUNC::CAUCHY;
  if (options_.p2p_loss_func == SteamRoOdometry::STEAM_LOSS_FUNC::GM)
    p2p_options.p2p_loss_func = P2PDopplerCVSuperCostTerm::LOSS_FUNC::GM;
  const auto p2p_super_cost_term =
      P2PDopplerCVSuperCostTerm::MakeShared(steam_trajectory, prev_steam_time, knot_times.back(), p2p_options);

  //
  const double max_pair_d2 = options_.p2p_max_dist * options_.p2p_max_dist;

  Eigen::Matrix<double, 6, 1> v_begin = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 1> v_end = Eigen::Matrix<double, 6, 1>::Zero();

  auto &p2p_matches = p2p_super_cost_term->get();
  p2p_matches.clear();
  int N_matches = 0;

#define SWF_INSIDE_ICP true

  for (int iter(0); iter < options_.num_iters_icp; iter++) {
    timer[0].second->start();
    transform_keypoints();
    timer[0].second->stop();

    // initialize problem
#if SWF_INSIDE_ICP
    SlidingWindowFilter problem(*sliding_window_filter_);
#else
    OptimizationProblem problem(/* num_threads */ options_.num_threads);
    for (const auto &var : steam_state_vars) problem.addStateVariable(var);
#endif

    // add prior cost terms
    steam_trajectory->addPriorCostTerms(problem);
    for (const auto &prior_cost_term : prior_cost_terms) problem.addCostTerm(prior_cost_term);

    meas_cost_terms.clear();
    p2p_matches.clear();
#if USE_P2P_SUPER_COST_TERM
    p2p_matches.reserve(keypoints.size());
#else
    meas_cost_terms.reserve(keypoints.size());
#endif

    timer[1].second->start();

#pragma omp declare reduction(merge_meas : std::vector<BaseCostTerm::ConstPtr> : omp_out.insert( \
        omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp declare reduction( \
        merge_matches : std::vector<P2PMatch> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for num_threads(options_.num_threads) reduction(merge_meas : meas_cost_terms) \
    reduction(merge_matches : p2p_matches)
    for (int i = 0; i < (int)keypoints.size(); i++) {
      const auto &keypoint = keypoints[i];
      const auto &pt_keypoint = keypoint.pt;

      if (innerloop_time) inner_timer[0].second->start();

      // Neighborhood search
      ArrayVector3d vector_neighbors =
          map_.searchNeighbors(pt_keypoint, nb_voxels_visited, options_.size_voxel_map, options_.max_number_neighbors);

      if ((int)vector_neighbors.size() < options_.min_number_neighbors) continue;

      if (innerloop_time) inner_timer[0].second->stop();

      if (innerloop_time) inner_timer[1].second->start();
      if (innerloop_time) inner_timer[1].second->stop();

      if (innerloop_time) inner_timer[2].second->start();

      const Eigen::Vector3d d_vec = keypoint.pt - vector_neighbors[0];
      if (d_vec.transpose() * d_vec > max_pair_d2) continue;

#if USE_P2P_SUPER_COST_TERM
      Eigen::Vector3d closest_pt = vector_neighbors[0];
      Eigen::Vector3d dummy_normal = Eigen::Vector3d::Zero();
      p2p_matches.emplace_back(P2PMatch(keypoint.timestamp, closest_pt, dummy_normal, keypoint.raw_pt));
#else
      Eigen::Vector3d closest_pt = vector_neighbors[0];
      Eigen::Matrix3d W = Eigen::Matrix3d::Identity();
      const auto noise_model = StaticNoiseModel<3>::MakeShared(W, NoiseType::INFORMATION);

      const auto &T_ms_intp_eval = T_ms_intp_eval_vec[i];
      auto error_func = [&]() -> Evaluable<Eigen::Matrix<double, 3, 1>>::Ptr {
        if (options_.beta != 0) {
          const auto &w_ms_ins_intp_eval = w_ms_ins_intp_eval_vec[i];
          return p2p::p2pErrorDoppler(T_ms_intp_eval, w_ms_ins_intp_eval, closest_pt, keypoint.raw_pt, options_.beta);
        } else {
          return p2p::p2pError(T_ms_intp_eval, closest_pt, keypoint.raw_pt);
        }
      }();
      const auto loss_func = [this]() -> BaseLossFunc::Ptr {
        switch (options_.p2p_loss_func) {
          case STEAM_LOSS_FUNC::L2:
            return L2LossFunc::MakeShared();
          case STEAM_LOSS_FUNC::DCS:
            return DcsLossFunc::MakeShared(options_.p2p_loss_sigma);
          case STEAM_LOSS_FUNC::CAUCHY:
            return CauchyLossFunc::MakeShared(options_.p2p_loss_sigma);
          case STEAM_LOSS_FUNC::GM:
            return GemanMcClureLossFunc::MakeShared(options_.p2p_loss_sigma);
          default:
            return nullptr;
        }
        return nullptr;
      }();

      const auto cost = WeightedLeastSqCostTerm<3>::MakeShared(error_func, noise_model, loss_func);
      meas_cost_terms.emplace_back(cost);
#endif

      if (innerloop_time) inner_timer[2].second->stop();
    }

#if USE_P2P_SUPER_COST_TERM
    N_matches = p2p_matches.size();
    p2p_super_cost_term->initP2PMatches();
#else
    N_matches = meas_cost_terms.size();
#endif

    for (const auto &cost : meas_cost_terms) problem.addCostTerm(cost);
    for (const auto &cost : imu_cost_terms) problem.addCostTerm(cost);
    for (const auto &cost : imu_prior_cost_terms) problem.addCostTerm(cost);
    problem.addCostTerm(p2p_super_cost_term);
    if (options_.use_imu && options_.use_accel) {
      problem.addCostTerm(preint_cost_term);
    }

    timer[1].second->stop();

    if (N_matches < options_.min_number_keypoints) {
      LOG(ERROR) << "[CT_ICP]Error : not enough keypoints selected in ct-icp !" << std::endl;
      LOG(ERROR) << "[CT_ICP]Number_of_residuals : " << N_matches << std::endl;
      icp_success = false;
      break;
    }

    timer[2].second->start();

    // Solve
    GaussNewtonSolverNVA::Params params;
    params.verbose = options_.verbose;
    params.max_iterations = (unsigned int)options_.max_iterations;
#if SWF_INSIDE_ICP
    params.reuse_previous_pattern = false;
#endif
    GaussNewtonSolverNVA solver(problem, params);
    solver.optimize();

    timer[2].second->stop();

    timer[3].second->start();

    // Update (changes trajectory data)
    double diff_trans = 0, diff_rot = 0, diff_vel = 0;

    Time curr_begin_steam_time(static_cast<double>(trajectory_[index_frame].begin_timestamp));
    const auto begin_T_mr = inverse(steam_trajectory->getPoseInterpolator(curr_begin_steam_time))->evaluate().matrix();
    const auto begin_T_ms = begin_T_mr * options_.T_sr.inverse();
    diff_trans += (current_estimate.begin_t - begin_T_ms.block<3, 1>(0, 3)).norm();
    diff_rot += AngularDistance(current_estimate.begin_R, begin_T_ms.block<3, 3>(0, 0));

    Time curr_end_steam_time(static_cast<double>(trajectory_[index_frame].end_timestamp));
    const auto end_T_mr = inverse(steam_trajectory->getPoseInterpolator(curr_end_steam_time))->evaluate().matrix();
    const auto end_T_ms = end_T_mr * options_.T_sr.inverse();
    diff_trans += (current_estimate.end_t - end_T_ms.block<3, 1>(0, 3)).norm();
    diff_rot += AngularDistance(current_estimate.end_R, end_T_ms.block<3, 3>(0, 0));

    const auto vb = steam_trajectory->getVelocityInterpolator(curr_begin_steam_time)->value();
    const auto ve = steam_trajectory->getVelocityInterpolator(curr_end_steam_time)->value();
    diff_vel += (vb - v_begin).norm();
    diff_vel += (ve - v_end).norm();
    v_begin = vb;
    v_end = ve;

    Time curr_mid_steam_time(static_cast<double>(trajectory_[index_frame].getEvalTime()));
    const auto mid_T_mr = inverse(steam_trajectory->getPoseInterpolator(curr_mid_steam_time))->evaluate().matrix();
    const auto mid_T_ms = mid_T_mr * options_.T_sr.inverse();
    current_estimate.setMidPose(mid_T_ms);

    current_estimate.begin_R = begin_T_ms.block<3, 3>(0, 0);
    current_estimate.begin_t = begin_T_ms.block<3, 1>(0, 3);
    current_estimate.end_R = end_T_ms.block<3, 3>(0, 0);
    current_estimate.end_t = end_T_ms.block<3, 1>(0, 3);

    timer[3].second->stop();

    if ((index_frame > 1) &&
        (diff_rot < options_.threshold_orientation_norm && diff_trans < options_.threshold_translation_norm &&
         diff_vel < options_.threshold_translation_norm * 4 + options_.threshold_orientation_norm * 4)) {
      if (options_.debug_print) {
        LOG(INFO) << "CT_ICP: Finished with N=" << iter << " ICP iterations" << std::endl;
      }
      // break;
    }
  }

  /// optimize in a sliding window
  LOG(INFO) << "Optimizing in a sliding window!" << std::endl;
  timer[4].second->start();
  // {
  //
  steam_trajectory->addPriorCostTerms(*sliding_window_filter_);  // ** this includes state priors (like for x_0)
  for (const auto &prior_cost_term : prior_cost_terms) sliding_window_filter_->addCostTerm(prior_cost_term);
  for (const auto &meas_cost_term : meas_cost_terms) sliding_window_filter_->addCostTerm(meas_cost_term);
  for (const auto &imu_cost : imu_cost_terms) sliding_window_filter_->addCostTerm(imu_cost);
  for (const auto &imu_prior_cost : imu_prior_cost_terms) sliding_window_filter_->addCostTerm(imu_prior_cost);
  sliding_window_filter_->addCostTerm(p2p_super_cost_term);
  if (options_.use_imu && options_.use_accel) {
    sliding_window_filter_->addCostTerm(preint_cost_term);
  }

  //
  LOG(INFO) << "number of variables: " << sliding_window_filter_->getNumberOfVariables() << std::endl;
  LOG(INFO) << "number of cost terms: " << sliding_window_filter_->getNumberOfCostTerms() << std::endl;
  if (sliding_window_filter_->getNumberOfVariables() > 100)
    throw std::runtime_error{"too many variables in the filter!"};
  if (sliding_window_filter_->getNumberOfCostTerms() > 100000)
    throw std::runtime_error{"too many cost terms in the filter!"};

  GaussNewtonSolverNVA::Params params;
  params.max_iterations = 20;
  params.reuse_previous_pattern = false;
  GaussNewtonSolverNVA solver(*sliding_window_filter_, params);
#if !SWF_INSIDE_ICP
  solver.optimize();
#endif
  // }
  timer[4].second->stop();

  // clang-format off
  Time curr_begin_steam_time(static_cast<double>(current_estimate.begin_timestamp));
  const auto curr_begin_T_mr = inverse(steam_trajectory->getPoseInterpolator(curr_begin_steam_time))->evaluate().matrix();
  const auto curr_begin_T_ms = curr_begin_T_mr * options_.T_sr.inverse();
  Time curr_end_steam_time(static_cast<double>(current_estimate.end_timestamp));
  const auto curr_end_T_mr = inverse(steam_trajectory->getPoseInterpolator(curr_end_steam_time))->evaluate().matrix();
  const auto curr_end_T_ms = curr_end_T_mr * options_.T_sr.inverse();

  Time curr_mid_steam_time(static_cast<double>(trajectory_[index_frame].getEvalTime()));
  const auto mid_T_mr = inverse(steam_trajectory->getPoseInterpolator(curr_mid_steam_time))->evaluate().matrix();
  const auto mid_T_ms = mid_T_mr * options_.T_sr.inverse();
  current_estimate.setMidPose(mid_T_ms);

  current_estimate.begin_R = curr_begin_T_ms.block<3, 3>(0, 0);
  current_estimate.begin_t = curr_begin_T_ms.block<3, 1>(0, 3);
  current_estimate.end_R = curr_end_T_ms.block<3, 3>(0, 0);
  current_estimate.end_t = curr_end_T_ms.block<3, 1>(0, 3);
  // clang-format on

  // Debug Code (stuff to plot)
  current_estimate.mid_w = steam_trajectory->getVelocityInterpolator(curr_mid_steam_time)->value();
  Covariance covariance(solver);
  current_estimate.mid_state_cov.block<12, 12>(0, 0) =
      steam_trajectory->getCovariance(covariance, trajectory_vars_[prev_trajectory_var_index].time);

  // timer[0].second->start();
  // transform_keypoints();
  // timer[0].second->stop();

  if (options_.use_imu) {
    size_t i = prev_trajectory_var_index;
    for (; i < trajectory_vars_.size() - 1; i++) {
      if (curr_mid_steam_time.seconds() >= trajectory_vars_[i].time.seconds() &&
          curr_mid_steam_time.seconds() < trajectory_vars_[i + 1].time.seconds())
        break;
    }
    if (curr_mid_steam_time.seconds() < trajectory_vars_[i].time.seconds() ||
        curr_mid_steam_time.seconds() >= trajectory_vars_[i + 1].time.seconds())
      throw std::runtime_error("mid time not within knot times");

    const auto bias_intp_eval =
        VSpaceInterpolator<6>::MakeShared(curr_mid_steam_time, trajectory_vars_[i].imu_biases, trajectory_vars_[i].time,
                                          trajectory_vars_[i + 1].imu_biases, trajectory_vars_[i + 1].time);

    current_estimate.mid_b = bias_intp_eval->value();
  }

  LOG(INFO) << "Number of keypoints used in CT-ICP : " << N_matches << std::endl;

  /// Debug print
  if (options_.debug_print) {
    for (size_t i = 0; i < timer.size(); i++)
      LOG(INFO) << "Elapsed " << timer[i].first << *(timer[i].second) << std::endl;
    if (innerloop_time) {
      for (size_t i = 0; i < inner_timer.size(); i++)
        LOG(INFO) << "Elapsed (Inner Loop) " << inner_timer[i].first << *(inner_timer[i].second) << std::endl;
    }
    LOG(INFO) << "Number iterations CT-ICP : " << options_.num_iters_icp << std::endl;
    LOG(INFO) << "Translation Begin: " << trajectory_[index_frame].begin_t.transpose() << std::endl;
    LOG(INFO) << "Translation End: " << trajectory_[index_frame].end_t.transpose() << std::endl;
  }

  return icp_success;
}

}  // namespace steam_icp
