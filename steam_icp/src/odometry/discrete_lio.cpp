#include <set>
#include <iomanip>
#include <random>
#include <glog/logging.h>

#include <steam.hpp>

#include "steam_icp/odometry/discrete_lio.hpp"
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

DiscreteLIOOdometry::DiscreteLIOOdometry(const Options &options) : Odometry(options), options_(options) {
  sliding_window_filter_ = steam::SlidingWindowFilter::MakeShared(options_.num_threads);
}

DiscreteLIOOdometry::~DiscreteLIOOdometry() { }

Trajectory DiscreteLIOOdometry::trajectory() {
  return trajectory_;
}

auto DiscreteLIOOdometry::registerFrame(const DataFrame &const_frame) -> RegistrationSummary {
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
  std::vector<Point3D> keypoints;
  if (index_frame > 0) {
    double sample_voxel_size =
        index_frame < options_.init_num_frames ? options_.init_sample_voxel_size : options_.sample_voxel_size;

    // downsample
    grid_sampling(frame, keypoints, sample_voxel_size, options_.num_threads);

    // icp
    summary.success = icp(index_frame, keypoints, const_frame.imu_data_vec, const_frame.pose_data_vec);
    summary.keypoints = keypoints;
    if (!summary.success) return summary;
  } else {
    using namespace steam;
    using namespace steam::se3;
    using namespace steam::vspace;
    using namespace steam::traj;

    // initial state
    lgmath::se3::Transformation T_mr;
    lgmath::se3::Transformation T_sr(options_.T_sr);
    Eigen::Matrix<double, 3, 1> v_rm_inm = Eigen::Matrix<double, 3, 1>::Zero();
    Eigen::Matrix<double, 6, 1> b_zero = Eigen::Matrix<double, 6, 1>::Zero();

    // initialize frame
    // const double begin_time = trajectory_[index_frame].begin_timestamp;
    const double time = trajectory_[index_frame].getEvalTime();
    Time steam_time(time);

    const auto T_mr_var = SE3StateVarGlobalPerturb::MakeShared(T_mr);
    const auto v_rm_inm_var = PreIntVelocityStateVar<3>::MakeShared(v_rm_inm, T_mr_var);
    const auto b_var = VSpaceStateVar<6>::MakeShared(b_zero);
    trajectory_vars_.emplace_back(steam_time, T_mr_var, v_rm_inm_var, b_var);

    to_marginalize_ = 0;

    // initialize gravity once at start-up (assumes stationary at start-up)
    Eigen::Matrix<double, 6, 1> xi_mi = initialize_gravity(const_frame.imu_data_vec);
    Eigen::Matrix3d C_mi = lgmath::se3::Transformation(xi_mi).matrix().block<3, 3>(0, 0);
    gravity_ << 0, 0, options_.gravity;
    gravity_ = C_mi * gravity_;

    summary.success = true;
  }
  prev_imu_data_vec_.clear();
  for (auto imu_data : const_frame.imu_data_vec) {
    prev_imu_data_vec_.push_back(imu_data);
  }
  trajectory_[index_frame].points = frame;
  trajectory_[index_frame].imu_data_vec.clear();
  for (auto imu_data : const_frame.imu_data_vec) {
    trajectory_[index_frame].imu_data_vec.push_back(imu_data);
  }

  const Eigen::Matrix4d T = trajectory_[index_frame].getMidPose();
  const Eigen::Vector3d t = T.block<3, 1>(0, 3);
  const Eigen::Matrix3d r = T.block<3, 3>(0, 0);

  // add points
  if (index_frame == 0) {
    updateMap(index_frame, index_frame);
  } else if ((index_frame - options_.delay_adding_points) > 0) {
    // if ((t - t_prev_).norm() > options_.keyframe_translation_threshold_m || fabs(AngularDistance(r, r_prev_)) > options_.keyframe_rotation_threshold_deg) {
      updateMap(index_frame, (index_frame - options_.delay_adding_points));
      t_prev_ = t;
      r_prev_ = r;
    // }
  }

  summary.corrected_points = keypoints;

  summary.R_ms = r;
  summary.t_ms = t;

  return summary;
}

void DiscreteLIOOdometry::initializeTimestamp(int index_frame, const DataFrame &const_frame) {
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

void DiscreteLIOOdometry::initializeMotion(int index_frame) {
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

std::vector<Point3D> DiscreteLIOOdometry::initializeFrame(int index_frame, const std::vector<Point3D> &const_frame) {
  std::vector<Point3D> frame(const_frame);

  double sample_size = index_frame < options_.init_num_frames ? options_.init_voxel_size : options_.voxel_size;
  std::mt19937_64 g;
  std::shuffle(frame.begin(), frame.end(), g);
  // Subsample the scan with voxels taking one random in every voxel
  sub_sample_frame(frame, sample_size, options_.num_threads);
  std::shuffle(frame.begin(), frame.end(), g);

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

void DiscreteLIOOdometry::updateMap(int index_frame, int update_frame) {
  using namespace steam::se3;
  using namespace steam::traj;
  Time mid_steam_time = Time(trajectory_[update_frame].getEvalTime());
  const double kSizeVoxelMap = options_.size_voxel_map;
  const double kMinDistancePoints = options_.min_distance_points;
  const int kMaxNumPointsInVoxel = options_.max_num_points_in_voxel;
  // update frame
  auto &frame = trajectory_[update_frame].points;
  if (update_frame > 0) {
    const auto &imu_data_vec = trajectory_[update_frame].imu_data_vec;
    size_t trajectory_vars_index = (to_marginalize_ - 1);
    for (; trajectory_vars_index < trajectory_vars_.size(); trajectory_vars_index++) {
      if (trajectory_vars_[trajectory_vars_index].time == mid_steam_time) break;
      if (trajectory_vars_[trajectory_vars_index].time > mid_steam_time) throw std::runtime_error("var.time > mid_steam_time, should not happen");
    }
    assert(trajectory_vars_[trajectory_vars_index].time == mid_steam_time);
    Time begin_steam_time = trajectory_[update_frame].begin_timestamp;
    Time end_steam_time = trajectory_[update_frame].end_timestamp;
    int num_states = 1;
    LOG(INFO) << "Adding points to map between (inclusive): " << begin_steam_time.seconds() << " - "
              << end_steam_time.seconds() << ", with num states: " << num_states << std::endl;
    std::set<double> unique_point_times_;
    for (const auto &point : frame) {
      unique_point_times_.insert(point.timestamp);
    }
    std::vector<double> unique_point_times(unique_point_times_.begin(), unique_point_times_.end());
    // Undistort using state estimate + IMU measurements
    bool undistort_only = false;
    // bool undistort_only = true;
    const Eigen::Matrix4d T_rs = options_.T_sr.inverse();
    transform_keypoints(unique_point_times, frame, imu_data_vec, mid_steam_time.seconds(), trajectory_vars_index, undistort_only, T_rs);
//         {
//     const auto T_ms = trajectory_vars_[trajectory_vars_index].T_mr->evaluate().matrix() * T_rs;
// #pragma omp parallel for num_threads(options_.num_threads)
//     for (int jj = 0; jj < (int)frame.size(); jj++) {
//       auto &keypoint = frame[jj];
//       keypoint.pt = T_ms.block<3, 3>(0, 0) * keypoint.raw_pt + T_ms.block<3, 1>(0, 3);
//     }
//         }
  }
  // Add the undistorted point to the map
  map_.add(frame, kSizeVoxelMap, kMaxNumPointsInVoxel, kMinDistancePoints);
  if (options_.filter_lifetimes) map_.update_and_filter_lifetimes();
  frame.clear();
  frame.shrink_to_fit();
  // remove points
  const double kMaxDistance = options_.max_distance;
  const Eigen::Vector3d location = trajectory_[index_frame].end_t;
  map_.remove(location, kMaxDistance);
}

Eigen::Matrix<double, 6, 1> DiscreteLIOOdometry::initialize_gravity(const std::vector<steam::IMUData> &imu_data_vec) {
  using namespace steam;
  using namespace steam::se3;
  using namespace steam::traj;
  using namespace steam::vspace;
  using namespace steam::imu;

  std::vector<BaseCostTerm::ConstPtr> cost_terms;
  cost_terms.reserve(imu_data_vec.size());
  Eigen::Matrix<double, 3, 3> R = Eigen::Matrix<double, 3, 3>::Identity();
  R.diagonal() = options_.r_imu_acc;
  const auto noise_model = StaticNoiseModel<3>::MakeShared(R);
  // const auto loss_func = CauchyLossFunc::MakeShared(1.0);
  const auto loss_func = L2LossFunc::MakeShared();
  const auto T_rm_init = SE3StateVar::MakeShared(lgmath::se3::Transformation());
  lgmath::se3::Transformation T_mi;
  const auto T_mi_var = SE3StateVar::MakeShared(T_mi);
  T_rm_init->locked() = true;
  Eigen::Matrix<double, 6, 1> b_zero = Eigen::Matrix<double, 6, 1>::Zero();
  const auto bias = VSpaceStateVar<6>::MakeShared(b_zero);
  bias->locked() = true;
  Eigen::Matrix<double, 6, 1> dw = Eigen::Matrix<double, 6, 1>::Zero();
  const auto dw_mr_inr = VSpaceStateVar<6>::MakeShared(dw);
  dw_mr_inr->locked() = true;
  for (const auto &imu_data : imu_data_vec) {
    const auto acc_error_func = imu::AccelerationError(T_rm_init, dw_mr_inr, bias, T_mi_var, imu_data.lin_acc);
    acc_error_func->setGravity(options_.gravity);
    const auto acc_cost = WeightedLeastSqCostTerm<3>::MakeShared(acc_error_func, noise_model, loss_func);
    cost_terms.emplace_back(acc_cost);
  }

  {
    Eigen::Matrix<double, 6, 6> init_T_mi_cov = Eigen::Matrix<double, 6, 6>::Identity();
    init_T_mi_cov.diagonal() = options_.T_mi_init_cov;
    init_T_mi_cov(3, 3) = 1.0;
    init_T_mi_cov(4, 4) = 1.0;
    init_T_mi_cov(5, 5) = 1.0;
    lgmath::se3::Transformation T_mi_zero;
    auto T_mi_error = se3_error(T_mi_var, T_mi_zero);
    auto noise_model = StaticNoiseModel<6>::MakeShared(init_T_mi_cov);
    auto loss_func = L2LossFunc::MakeShared();
    const auto T_mi_prior_factor = WeightedLeastSqCostTerm<6>::MakeShared(T_mi_error, noise_model, loss_func);
    cost_terms.emplace_back(T_mi_prior_factor);
  }

  // Solve
  OptimizationProblem problem;
  for (const auto &cost : cost_terms) problem.addCostTerm(cost);
  problem.addStateVariable(T_mi_var);
  GaussNewtonSolverNVA::Params params;
  params.verbose = options_.verbose;
  params.max_iterations = (unsigned int)options_.max_iterations;
  GaussNewtonSolverNVA solver(problem, params);
  solver.optimize();
  LOG(INFO) << "Initialization, T_mi:" << std::endl
            << T_mi_var->value().matrix() << std::endl
            << "vec: " << T_mi_var->value().vec() << std::endl;
  return T_mi_var->value().vec();
}

void DiscreteLIOOdometry::transform_keypoints(
  const std::vector<double> &unique_point_times,
  std::vector<Point3D> &keypoints,
  const std::vector<steam::IMUData> &imu_data_vec,
  const double curr_time,
  const uint trajectory_vars_index,
  bool undistort_only,
  Eigen::Matrix4d T_rs
  ) {
  Eigen::Matrix4d T_mr = trajectory_vars_[trajectory_vars_index].T_mr->evaluate().matrix();
  const Eigen::Matrix4d T_r0_m = T_mr.inverse();
  Eigen::Matrix3d C_mr = T_mr.block<3, 3>(0, 0);
  Eigen::Vector3d r_rm_inm = T_mr.block<3, 1>(0, 3);
  Eigen::Vector3d v_rm_inm = trajectory_vars_[trajectory_vars_index].v_rm_inm->evaluate();
  Eigen::Matrix<double, 6, 1> b = trajectory_vars_[trajectory_vars_index].imu_biases->evaluate();
  Eigen::Vector3d ba = b.block<3, 1>(0, 0);
  Eigen::Vector3d bg = b.block<3, 1>(3, 0);
  std::vector<std::pair<double, Eigen::Matrix4d>> T_mr_vec;
  T_mr_vec.push_back(std::make_pair(curr_time, T_mr));

  const double min_time = *std::min_element(unique_point_times.begin(), unique_point_times.end());
  const double max_time = *std::max_element(unique_point_times.begin(), unique_point_times.end());

  // IMU measurements before / after the evaluation time (mid of scan)
  std::vector<steam::IMUData> imu_before;
  std::vector<steam::IMUData> imu_after;
  for (auto imu_data : imu_data_vec) {
    if (imu_data.timestamp < curr_time) {
      imu_before.push_back(imu_data);
    } else {
      imu_after.push_back(imu_data);
    }
  }
  // before:
  if (imu_before.size() > 0) {
    std::reverse(imu_before.begin(), imu_before.end());
    double delta_t = fabs(imu_before[0].timestamp - curr_time);
    if (delta_t > 0) {
      C_mr = C_mr * lgmath::so3::vec2rot(-1 * (imu_before[0].ang_vel - bg) * delta_t);
      v_rm_inm -= (C_mr * (imu_before[0].lin_acc - ba) + gravity_) * delta_t;
      r_rm_inm -= v_rm_inm * delta_t + 0.5 * (C_mr * (imu_before[0].lin_acc - ba) + gravity_) * delta_t * delta_t;
      
      Eigen::Matrix4d T_mr = Eigen::Matrix4d::Identity();
      T_mr.block<3, 3>(0, 0) = C_mr;
      T_mr.block<3, 1>(0, 3) = r_rm_inm;
      T_mr_vec.push_back(std::make_pair(imu_before[0].timestamp, T_mr));
    }
    for (size_t j = 0; j < imu_before.size(); ++j) {
      double delta_t = 0;
      double pose_time = 0;
      if (j < imu_before.size() - 1) {
        delta_t = fabs(imu_before[j + 1].timestamp - imu_before[j].timestamp);
        pose_time = imu_before[j + 1].timestamp;
      } else if (j == imu_before.size() - 1) {
        delta_t = fabs(min_time - imu_before[j].timestamp);
        pose_time = min_time;
      }
      C_mr = C_mr * lgmath::so3::vec2rot(-1 * (imu_before[j].ang_vel - bg) * delta_t);
      v_rm_inm -= (C_mr * (imu_before[j].lin_acc - ba) + gravity_) * delta_t;
      r_rm_inm -= v_rm_inm * delta_t + 0.5 * (C_mr * (imu_before[j].lin_acc - ba) + gravity_) * delta_t * delta_t;
      
      
      Eigen::Matrix4d T_mr = Eigen::Matrix4d::Identity();
      T_mr.block<3, 3>(0, 0) = C_mr;
      T_mr.block<3, 1>(0, 3) = r_rm_inm;
      T_mr_vec.push_back(std::make_pair(pose_time, T_mr));
    }
  }
  // after:
  if (imu_after.size() > 0) {
    T_mr = trajectory_vars_[trajectory_vars_index].T_mr->evaluate().matrix();
    C_mr = T_mr.block<3, 3>(0, 0);
    r_rm_inm = T_mr.block<3, 1>(0, 3);
    v_rm_inm = trajectory_vars_[trajectory_vars_index].v_rm_inm->evaluate();

    double delta_t = imu_after[0].timestamp - curr_time;
    if (delta_t > 0) {
      r_rm_inm += v_rm_inm * delta_t + 0.5 * (C_mr * (imu_after[0].lin_acc - ba) + gravity_) * delta_t * delta_t;
      v_rm_inm += (C_mr * (imu_after[0].lin_acc - ba) + gravity_) * delta_t;
      C_mr = C_mr * lgmath::so3::vec2rot((imu_after[0].ang_vel - bg) * delta_t);
      Eigen::Matrix4d T_mr = Eigen::Matrix4d::Identity();
      T_mr.block<3, 3>(0, 0) = C_mr;
      T_mr.block<3, 1>(0, 3) = r_rm_inm;
      T_mr_vec.push_back(std::make_pair(imu_after[0].timestamp, T_mr));
    }
    for (size_t j = 0; j < imu_after.size(); ++j) {
      double delta_t = 0;
      double pose_time = 0;
      if (j < imu_after.size() - 1) {
        delta_t = imu_after[j + 1].timestamp - imu_after[j].timestamp;
        pose_time = imu_after[j + 1].timestamp;
      } else if (j == imu_after.size() - 1) {
        delta_t = max_time - imu_after[j].timestamp;
        pose_time = max_time;
      }
      assert(delta_t > 0);
      r_rm_inm += v_rm_inm * delta_t + 0.5 * (C_mr * (imu_after[j].lin_acc - ba) + gravity_) * delta_t * delta_t;
      v_rm_inm += (C_mr * (imu_after[j].lin_acc - ba) + gravity_) * delta_t;
      C_mr = C_mr * lgmath::so3::vec2rot(-1 * (imu_after[j].ang_vel - bg) * delta_t);
      Eigen::Matrix4d T_mr = Eigen::Matrix4d::Identity();
      T_mr.block<3, 3>(0, 0) = C_mr;
      T_mr.block<3, 1>(0, 3) = r_rm_inm;
      T_mr_vec.push_back(std::make_pair(pose_time, T_mr));
    }
  }

  std::sort(T_mr_vec.begin(), T_mr_vec.end(), [](auto &left, auto &right) {
    return left.first < right.first;
  });

  std::vector<double> imu_pose_times;
  for (auto pair : T_mr_vec) {
    imu_pose_times.push_back(pair.first);
  }

  std::map<double, Eigen::Matrix4d> T_ms_cache_map;
  const Eigen::Matrix4d T_sr = T_rs.inverse();
  const Eigen::Matrix4d T_s0_m = T_sr * T_r0_m;
#pragma omp parallel for num_threads(options_.num_threads)
  for (int jj = 0; jj < (int)unique_point_times.size(); jj++) {
    const auto &ts = unique_point_times[jj];
    int start_index = 0, end_index = 0;
    for (size_t k = 0; k < imu_pose_times.size(); ++k) {
      if (imu_pose_times[k] > ts) {
        end_index = k;
        break;
      }
      start_index = k;
      end_index = k;
    }
    if ((ts == imu_pose_times[start_index]) || (start_index == end_index) || (imu_pose_times[start_index] == imu_pose_times[end_index])) {
#pragma omp critical
      T_ms_cache_map[ts] = T_mr_vec[start_index].second * T_rs;
    } else if (ts == imu_pose_times[end_index]) {
#pragma omp critical
      T_ms_cache_map[ts] = T_mr_vec[end_index].second * T_rs;
    } else {
      double alpha = (ts - imu_pose_times[start_index]) / (imu_pose_times[end_index] - imu_pose_times[start_index]);
      assert(alpha > 0 && alpha <= 1);
      Eigen::Matrix<double, 6, 1> xi = lgmath::se3::tran2vec(T_mr_vec[start_index].second.inverse() * T_mr_vec[end_index].second);
      Eigen::Matrix4d T_ms = T_mr_vec[start_index].second * lgmath::se3::vec2tran(xi * alpha).matrix() * T_rs;
      if (undistort_only)
        T_ms = T_s0_m * T_ms;  // T_s0_s
#pragma omp critical
      T_ms_cache_map[ts] = T_ms;
    }
  }

#pragma omp parallel for num_threads(options_.num_threads)
  for (int jj = 0; jj < (int)keypoints.size(); jj++) {
    auto &keypoint = keypoints[jj];
    const Eigen::Matrix4d &T_ms = T_ms_cache_map[keypoint.timestamp];
    if (undistort_only) {
      // keypoint.raw_pt = T_ms.block<3, 3>(0, 0) * keypoint.raw_pt + T_ms.block<3, 1>(0, 3);
      keypoint.pt = T_ms.block<3, 3>(0, 0) * keypoint.raw_pt + T_ms.block<3, 1>(0, 3);
    } else {
      keypoint.pt = T_ms.block<3, 3>(0, 0) * keypoint.raw_pt + T_ms.block<3, 1>(0, 3);
    }
    
  }
}

bool DiscreteLIOOdometry::icp(int index_frame, std::vector<Point3D> &keypoints,
                          const std::vector<steam::IMUData> &imu_data_vec,
                          const std::vector<PoseData> &pose_data_vec) {
  using namespace steam;
  using namespace steam::se3;
  using namespace steam::traj;
  using namespace steam::vspace;
  using namespace steam::imu;

  bool icp_success = true;

  std::vector<StateVarBase::Ptr> steam_state_vars;
  std::vector<BaseCostTerm::ConstPtr> meas_cost_terms;
  std::vector<BaseCostTerm::ConstPtr> imu_prior_cost_terms;
  std::vector<BaseCostTerm::ConstPtr> pose_meas_cost_terms;
  std::vector<BaseCostTerm::ConstPtr> prior_cost_terms;
  const size_t prev_trajectory_var_index = trajectory_vars_.size() - 1;
  size_t curr_trajectory_var_index = trajectory_vars_.size() - 1;

  /// use previous trajectory to initialize steam state variables
  LOG(INFO) << "[ICP_STEAM] prev scan eval time: " << trajectory_[index_frame - 1].getEvalTime() << std::endl;
  const double prev_time = trajectory_[index_frame - 1].getEvalTime();
  if (trajectory_vars_.back().time != Time(static_cast<double>(prev_time)))
    throw std::runtime_error{"missing previous scan eval variable"};
  Time prev_steam_time = trajectory_vars_.back().time;
  lgmath::se3::Transformation prev_T_mr = trajectory_vars_.back().T_mr->value();
  Eigen::Vector3d prev_v_rm_inm = trajectory_vars_.back().v_rm_inm->value();
  Eigen::Matrix<double, 6, 1> prev_imu_biases = trajectory_vars_.back().imu_biases->value();
  const auto prev_T_mr_var = trajectory_vars_.back().T_mr;
  const auto prev_v_rm_inm_var = trajectory_vars_.back().v_rm_inm;
  const auto prev_imu_biases_var = trajectory_vars_.back().imu_biases;
  steam_state_vars.emplace_back(prev_T_mr_var);
  steam_state_vars.emplace_back(prev_v_rm_inm_var);
  steam_state_vars.emplace_back(prev_imu_biases_var);

  /// New state for this frame
  LOG(INFO) << "[ICP_STEAM] curr scan eval time: " << trajectory_[index_frame].getEvalTime() << std::endl;
  LOG(INFO) << "[ICP_STEAM] total num new states: " << 1 << std::endl;
  const double curr_time = trajectory_[index_frame].getEvalTime();
  std::vector<double> knot_times = {curr_time};

  // get the set of IMU measurements between the previous scan and the current scan
  std::vector<steam::IMUData> preint_imu_vec;
  for (auto imu_data : prev_imu_data_vec_) {
    if (imu_data.timestamp >= prev_time && imu_data.timestamp < curr_time) {
      preint_imu_vec.push_back(imu_data);
    }
  }
  for (auto imu_data : imu_data_vec) {
    if (imu_data.timestamp >= prev_time && imu_data.timestamp < curr_time) {
      preint_imu_vec.push_back(imu_data);
    }
  }
  for (auto imu_data : preint_imu_vec) {
    std::cout << imu_data.timestamp << " ";
  }
  std::cout << std::endl;

  // Add new state variables, initialize using IMU integration
  {
    double knot_time = knot_times[0];
    Time knot_steam_time(knot_time);

    Eigen::Matrix3d C_mr = prev_T_mr.matrix().block<3, 3>(0, 0);
    Eigen::Vector3d r_rm_inm = prev_T_mr.matrix().block<3, 1>(0, 3);
    Eigen::Vector3d v_rm_inm = prev_v_rm_inm;
    Eigen::Vector3d ba = prev_imu_biases.block<3, 1>(0, 0);
    Eigen::Vector3d bg = prev_imu_biases.block<3, 1>(3, 0);
    double delta_t = preint_imu_vec[0].timestamp - prev_time;
    if (delta_t > 0) {
      r_rm_inm += v_rm_inm * delta_t + 0.5 * (C_mr * (preint_imu_vec[0].lin_acc - ba) + gravity_) * delta_t * delta_t;
      v_rm_inm += (C_mr * (preint_imu_vec[0].lin_acc - ba) + gravity_) * delta_t;
      C_mr = C_mr * lgmath::so3::vec2rot((preint_imu_vec[0].ang_vel - bg) * delta_t);
    }

    for (size_t j = 0; j < preint_imu_vec.size(); ++j) {
      double delta_t = 0;
      if (j < preint_imu_vec.size() - 1) {
        delta_t = preint_imu_vec[j + 1].timestamp - preint_imu_vec[j].timestamp;
      } else if (j == preint_imu_vec.size() - 1) {
        delta_t = curr_time - preint_imu_vec[j].timestamp;
      }
      r_rm_inm += v_rm_inm * delta_t + 0.5 * (C_mr * (preint_imu_vec[j].lin_acc - ba) + gravity_) * delta_t * delta_t;
      v_rm_inm += (C_mr * (preint_imu_vec[j].lin_acc - ba) + gravity_) * delta_t;
      C_mr = C_mr * lgmath::so3::vec2rot((preint_imu_vec[j].ang_vel - bg) * delta_t);
    }
    
    // Initialize new state using IMU integration
    Eigen::Matrix4d T_mr = Eigen::Matrix4d::Identity();
    T_mr.block<3, 3>(0, 0) = C_mr;
    T_mr.block<3, 1>(0, 3) = r_rm_inm;
    std::cout << "new T_mr: " << T_mr << std::endl;
    const auto T_mr_var = SE3StateVarGlobalPerturb::MakeShared(lgmath::se3::Transformation(T_mr));
    const auto v_rm_in_m_var = PreIntVelocityStateVar<3>::MakeShared(v_rm_inm, T_mr_var);
    const auto imu_biases_var = VSpaceStateVar<6>::MakeShared(prev_imu_biases);
    steam_state_vars.emplace_back(T_mr_var);
    steam_state_vars.emplace_back(v_rm_in_m_var);
    steam_state_vars.emplace_back(imu_biases_var);
    // cache the end state in full steam trajectory because it will be used again
    trajectory_vars_.emplace_back(knot_steam_time, T_mr_var, v_rm_in_m_var, imu_biases_var);
    curr_trajectory_var_index++;
  }

  // constant velocity prior
  if (index_frame > 2) {
    // Eigen::Matrix<double, 6, 1> prev_w_mr_inr = Eigen::Matrix<double, 6, 1>::Zero();
    // const lgmath::se3::Transformation T_mr_prev_prev = trajectory_vars_[trajectory_vars_.size() - 2].T_mr->value();
    // prev_w_mr_inr = (prev_T_mr.inverse() * T_mr_prev_prev).vec() / (trajectory_vars_[trajectory_vars_.size() - 1].time - trajectory_vars_[trajectory_vars_.size() - 2].time).seconds();
    // const Eigen::Matrix<double, 6, 1> xi_mr_inr_odo((curr_time - prev_time) * prev_w_mr_inr);
    // const auto T_rm_prediction = lgmath::se3::Transformation(xi_mr_inr_odo) * prev_T_mr.inverse();
    // const auto T_mr_prediction = T_rm_prediction.inverse();
    
    // Eigen::Matrix<double, 6, 6> init_pose_cov = Eigen::Matrix<double, 6, 6>::Identity();
    // init_pose_cov.diagonal() = options_.p0_pose;
    // auto pose_error = se3_global_perturb_error(trajectory_vars_.back().T_mr, T_mr_prediction);
    // auto noise_model = StaticNoiseModel<6>::MakeShared(init_pose_cov);
    // auto loss_func = L2LossFunc::MakeShared();
    // const auto pose_prior_factor = WeightedLeastSqCostTerm<6>::MakeShared(pose_error, noise_model, loss_func);
    // prior_cost_terms.emplace_back(pose_prior_factor);

    // {
    //   Eigen::Matrix3d init_vel_cov = Eigen::Matrix3d::Identity();
    //   init_vel_cov.diagonal() = options_.p0_vel;
    //   const auto nvk = vspace::NegationEvaluator<3>::MakeShared(trajectory_vars_.back().v_rm_inm);
    //   auto velocity_error = vspace::AdditionEvaluator<3>::MakeShared(trajectory_vars_[prev_trajectory_var_index].v_rm_inm, nvk);
    //   auto noise_model = StaticNoiseModel<3>::MakeShared(init_vel_cov);
    //   auto loss_func = L2LossFunc::MakeShared();
    //   const auto velocity_prior_factor = WeightedLeastSqCostTerm<3>::MakeShared(velocity_error, noise_model, loss_func);
    //   prior_cost_terms.emplace_back(velocity_prior_factor);
    // }
  }

  if (index_frame == 1) {
    const auto &prev_var = trajectory_vars_.at(prev_trajectory_var_index);
    {
      Eigen::Matrix<double, 6, 6> init_pose_cov = Eigen::Matrix<double, 6, 6>::Identity();
      init_pose_cov.diagonal() = options_.p0_pose;
      auto pose_error = se3_global_perturb_error(prev_var.T_mr, lgmath::se3::Transformation());
      auto noise_model = StaticNoiseModel<6>::MakeShared(init_pose_cov);
      auto loss_func = L2LossFunc::MakeShared();
      const auto pose_prior_factor = WeightedLeastSqCostTerm<6>::MakeShared(pose_error, noise_model, loss_func);
      prior_cost_terms.emplace_back(pose_prior_factor);
    }
    {
      Eigen::Matrix3d init_vel_cov = Eigen::Matrix3d::Identity();
      init_vel_cov.diagonal() = options_.p0_vel;
      auto velocity_error = vspace::vspace_error<3>(prev_var.v_rm_inm, Eigen::Vector3d::Zero());
      auto noise_model = StaticNoiseModel<3>::MakeShared(init_vel_cov);
      auto loss_func = L2LossFunc::MakeShared();
      const auto velocity_prior_factor = WeightedLeastSqCostTerm<3>::MakeShared(velocity_error, noise_model, loss_func);
      prior_cost_terms.emplace_back(velocity_prior_factor);
    }
    Eigen::Matrix<double, 6, 1> b_zero = Eigen::Matrix<double, 6, 1>::Zero();
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

  if (index_frame > 1 && options_.use_bias_prior_after_init) {
    const auto &prev_var = trajectory_vars_.at(prev_trajectory_var_index);
    Eigen::Matrix<double, 6, 1> b_zero = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 6> init_bias_cov = Eigen::Matrix<double, 6, 6>::Identity();
    init_bias_cov.block<3, 3>(0, 0) = Eigen::Matrix<double, 3, 3>::Identity() * options_.pk_bias_accel;
    init_bias_cov.block<3, 3>(3, 3) = Eigen::Matrix<double, 3, 3>::Identity() * options_.pk_bias_gyro;
    auto bias_error = vspace::vspace_error<6>(prev_var.imu_biases, b_zero);
    auto noise_model = StaticNoiseModel<6>::MakeShared(init_bias_cov);
    auto loss_func = L2LossFunc::MakeShared();
    const auto bias_prior_factor = WeightedLeastSqCostTerm<6>::MakeShared(bias_error, noise_model, loss_func);
    imu_prior_cost_terms.emplace_back(bias_prior_factor);
  }

  /// update sliding window variables
  {
    if (index_frame == 1) {
      const auto &prev_var = trajectory_vars_.at(prev_trajectory_var_index);
      sliding_window_filter_->addStateVariable(std::vector<StateVarBase::Ptr>{prev_var.T_mr, prev_var.v_rm_inm, prev_var.imu_biases});
    }

    for (size_t i = prev_trajectory_var_index + 1; i <= curr_trajectory_var_index; ++i) {
      const auto &var = trajectory_vars_.at(i);
      sliding_window_filter_->addStateVariable(std::vector<StateVarBase::Ptr>{var.T_mr, var.v_rm_inm, var.imu_biases});
    }

    if ((index_frame - options_.delay_adding_points) > 0) {
      const double begin_marg_time = trajectory_vars_.at(to_marginalize_).time.seconds();
      double end_marg_time = trajectory_vars_.at(to_marginalize_).time.seconds();
      std::vector<StateVarBase::Ptr> marg_vars;
      int num_states = 0;
      double marg_time = trajectory_.at(index_frame - options_.delay_adding_points - 1).getEvalTime();
      Time marg_steam_time(marg_time);
      for (size_t i = to_marginalize_; i <= curr_trajectory_var_index; ++i) {
        const auto &var = trajectory_vars_.at(i);
        if (var.time <= marg_steam_time) {
          end_marg_time = var.time.seconds();
          marg_vars.emplace_back(var.T_mr);
          marg_vars.emplace_back(var.v_rm_inm);
          marg_vars.emplace_back(var.imu_biases);
          num_states++;
        } else {
          to_marginalize_ = i;
          break;
        }
      }
      sliding_window_filter_->marginalizeVariable(marg_vars);
      LOG(INFO) << "Marginalizing time (inclusive): " << begin_marg_time << " - " << end_marg_time
                << ", with num states: " << num_states << std::endl;
    }
  }

  // get pose meas cost terms (debug only)
  // {
  //   pose_meas_cost_terms.reserve(pose_data_vec.size());
  //   Eigen::Matrix<double, 6, 6> R_pose = Eigen::Matrix<double, 6, 6>::Zero();
  //   R_pose.diagonal() = options_.r_pose;
  //   const auto pose_noise_model = StaticNoiseModel<6>::MakeShared(R_pose);
  //   const auto pose_loss_func = CauchyLossFunc::MakeShared(1.0);
  //   for (const auto &pose_data : pose_data_vec) {
  //     if (fabs(pose_data.timestamp - trajectory_vars_.back().time.seconds()) > 0.001)
  //       continue;
  //     std::cout << "pose_meas time: " << pose_data.timestamp << " traj time " << trajectory_vars_.back().time.seconds() << std::endl;
  //     auto T_mr_var = trajectory_vars_.back().T_mr;
  //     const auto T_mr_meas = lgmath::se3::Transformation(Eigen::Matrix4d(pose_data.pose.inverse() * options_.T_sr));
  //     auto pose_error = se3_global_perturb_error(T_mr_var, T_mr_meas);
  //     const auto pose_cost = WeightedLeastSqCostTerm<6>::MakeShared(pose_error, pose_noise_model, pose_loss_func);
  //     pose_meas_cost_terms.emplace_back(pose_cost);
  //   }
  // }

  // Preintegrated IMU cost term
  auto imu_options = PreintIMUCostTerm::Options();
  imu_options.loss_sigma = options_.imu_loss_sigma;
  if (options_.imu_loss_func == "L2") imu_options.loss_func = PreintIMUCostTerm::LOSS_FUNC::L2;
  if (options_.imu_loss_func == "DCS") imu_options.loss_func = PreintIMUCostTerm::LOSS_FUNC::DCS;
  if (options_.imu_loss_func == "CAUCHY") imu_options.loss_func = PreintIMUCostTerm::LOSS_FUNC::CAUCHY;
  if (options_.imu_loss_func == "GM") imu_options.loss_func = PreintIMUCostTerm::LOSS_FUNC::GM;
  imu_options.gravity = gravity_;
  imu_options.r_imu_acc = options_.r_imu_acc;
  imu_options.r_imu_ang = options_.r_imu_ang;

  const auto preint_cost_term = PreintIMUCostTerm::MakeShared(
      prev_steam_time,
      trajectory_vars_.back().time,
      trajectory_vars_[prev_trajectory_var_index].T_mr,
      trajectory_vars_[prev_trajectory_var_index + 1].T_mr,
      trajectory_vars_[prev_trajectory_var_index].v_rm_inm,
      trajectory_vars_[prev_trajectory_var_index + 1].v_rm_inm,
      trajectory_vars_[prev_trajectory_var_index + 1].imu_biases, 
      imu_options);
  preint_cost_term->set(preint_imu_vec);

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

  // Get vec of T_mr and times towards interpolating and undistorting the pointcloud
  std::set<double> unique_point_times_;
  for (const auto &keypoint : keypoints) {
    unique_point_times_.insert(keypoint.timestamp);
  }
  std::vector<double> unique_point_times(unique_point_times_.begin(), unique_point_times_.end());

  // For the 50 first frames, visit 2 voxels
  const short nb_voxels_visited = index_frame < options_.init_num_frames ? 2 : 1;
  const int kMinNumNeighbors = options_.min_number_neighbors;

  auto &current_estimate = trajectory_.at(index_frame);

  // timers
  std::vector<std::pair<std::string, std::unique_ptr<Stopwatch<>>>> timer;
  timer.emplace_back("Update Transform ............... ", std::make_unique<Stopwatch<>>(false));
  timer.emplace_back("Association .................... ", std::make_unique<Stopwatch<>>(false));
  timer.emplace_back("Optimization ................... ", std::make_unique<Stopwatch<>>(false));
  timer.emplace_back("Alignment ...................... ", std::make_unique<Stopwatch<>>(false));

#define USE_P2P_SUPER_COST_TERM true
  auto p2p_options = P2PGlobalSuperCostTerm::Options();
  p2p_options.num_threads = options_.num_threads;
  p2p_options.p2p_loss_sigma = options_.p2p_loss_sigma;
  p2p_options.gravity = gravity_;
  if (options_.p2p_loss_func == DiscreteLIOOdometry::STEAM_LOSS_FUNC::L2)
    p2p_options.p2p_loss_func = P2PGlobalSuperCostTerm::LOSS_FUNC::L2;
  if (options_.p2p_loss_func == DiscreteLIOOdometry::STEAM_LOSS_FUNC::DCS)
    p2p_options.p2p_loss_func = P2PGlobalSuperCostTerm::LOSS_FUNC::DCS;
  if (options_.p2p_loss_func == DiscreteLIOOdometry::STEAM_LOSS_FUNC::CAUCHY)
    p2p_options.p2p_loss_func = P2PGlobalSuperCostTerm::LOSS_FUNC::CAUCHY;
  if (options_.p2p_loss_func == DiscreteLIOOdometry::STEAM_LOSS_FUNC::GM)
    p2p_options.p2p_loss_func = P2PGlobalSuperCostTerm::LOSS_FUNC::GM;
  const auto p2p_super_cost_term =
      P2PGlobalSuperCostTerm::MakeShared(trajectory_vars_.back().time,
      trajectory_vars_.back().T_mr,
      trajectory_vars_.back().v_rm_inm,
      trajectory_vars_.back().imu_biases,
      p2p_options,
      imu_data_vec);

  // Transform points into the robot frame just once:
  timer[0].second->start();
  const Eigen::Matrix4d T_rs_mat = options_.T_sr.inverse();
#pragma omp parallel for num_threads(options_.num_threads)
  for (int i = 0; i < (int)keypoints.size(); i++) {
    auto &keypoint = keypoints[i];
    keypoint.raw_pt = T_rs_mat.block<3, 3>(0, 0) * keypoint.raw_pt + T_rs_mat.block<3, 1>(0, 3);
  }
  timer[0].second->stop();
  auto &p2p_matches = p2p_super_cost_term->get();
  p2p_matches.clear();
  int N_matches = 0;

  Eigen::Vector3d v_end = Eigen::Vector3d::Zero();

  bool swf_inside_icp = false;  // kitti-raw : false
  if (index_frame > options_.init_num_frames || options_.swf_inside_icp_at_begin) {
    swf_inside_icp = true;
  }

  const auto p2p_loss_func = [this]() -> BaseLossFunc::Ptr {
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

  // De-skew points just once:
  // transform_keypoints(unique_point_times, keypoints, imu_data_vec, curr_time, trajectory_vars_.size() - 1, true, Eigen::Matrix4d::Identity());

  auto transform_keypoints_simple = [&]() {
    const auto T_mr = trajectory_vars_.back().T_mr->evaluate().matrix();
#pragma omp parallel for num_threads(options_.num_threads)
    for (int jj = 0; jj < (int)keypoints.size(); jj++) {
      auto &keypoint = keypoints[jj];
      keypoint.pt = T_mr.block<3, 3>(0, 0) * keypoint.raw_pt + T_mr.block<3, 1>(0, 3);
    }
  };

  //
  for (int iter(0); iter < options_.num_iters_icp; iter++) {
    timer[0].second->start();
    // transform_keypoints_simple();
    transform_keypoints(unique_point_times, keypoints, imu_data_vec, curr_time, trajectory_vars_.size() - 1, false, Eigen::Matrix4d::Identity());
    // create undistorted copy of keypoints
    std::vector<Point3D> undistorted_points(keypoints);
    transform_keypoints(unique_point_times, undistorted_points, imu_data_vec, curr_time, trajectory_vars_.size() - 1, true, Eigen::Matrix4d::Identity());
    timer[0].second->stop();

    // initialize problem
    const auto problem = [&]() -> Problem::Ptr {
      if (swf_inside_icp) {
        return std::make_shared<SlidingWindowFilter>(*sliding_window_filter_);
      } else {
        auto problem = OptimizationProblem::MakeShared(options_.num_threads);
        for (const auto &var : steam_state_vars) problem->addStateVariable(var);
        return problem;
      }
    }();

    meas_cost_terms.clear();

    timer[1].second->start();

#pragma omp declare reduction(merge_meas : std::vector<BaseCostTerm::ConstPtr> : omp_out.insert( \
        omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp declare reduction( \
        merge_matches : std::vector<P2PMatch> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for num_threads(options_.num_threads) reduction(merge_matches : p2p_matches) reduction(merge_meas : meas_cost_terms)
    for (int i = 0; i < (int)keypoints.size(); i++) {
      const auto &keypoint = keypoints[i];
      const auto &pt_keypoint = keypoint.pt;

      // Neighborhood search
      ArrayVector3d vector_neighbors =
          map_.searchNeighbors(pt_keypoint, nb_voxels_visited, options_.size_voxel_map, options_.max_number_neighbors);

      if ((int)vector_neighbors.size() >= kMinNumNeighbors) {
        // Compute normals from neighbors
        auto neighborhood = compute_neighborhood_distribution(vector_neighbors);

        const double planarity_weight = std::pow(neighborhood.a2D, options_.power_planarity);
        const double weight = planarity_weight;
        Eigen::Vector3d closest_pt = vector_neighbors[0];
        const double dist_to_plane = std::abs((keypoint.pt - closest_pt).transpose() * neighborhood.normal);
        if (dist_to_plane >= options_.p2p_max_dist)
          continue;
        Eigen::Vector3d closest_normal = weight * neighborhood.normal;
#if USE_P2P_SUPER_COST_TERM
        p2p_matches.emplace_back(P2PMatch(keypoint.timestamp, closest_pt, closest_normal, keypoint.raw_pt));
#else
        const auto noise_model = StaticNoiseModel<1>::MakeShared(Eigen::Matrix<double, 1, 1>::Identity());
        const auto error_func = p2p::p2planeGlobalError(trajectory_vars_.back().T_mr, closest_pt, undistorted_points[i].pt, closest_normal);
        // const auto error_func = p2p::p2planeGlobalError(trajectory_vars_.back().T_mr, closest_pt, keypoints[i].raw_pt, closest_normal);
        error_func->setTime(Time(keypoint.timestamp));
        const auto cost = WeightedLeastSqCostTerm<1>::MakeShared(error_func, noise_model, p2p_loss_func);
        meas_cost_terms.emplace_back(cost);
#endif
      }
    }

#if USE_P2P_SUPER_COST_TERM
    N_matches = p2p_matches.size();
    p2p_super_cost_term->initP2PMatches();
    problem->addCostTerm(p2p_super_cost_term);
#else
    N_matches = meas_cost_terms.size();
#endif

    for (const auto &cost : meas_cost_terms) problem->addCostTerm(cost);
    for (const auto &cost : imu_prior_cost_terms) problem->addCostTerm(cost);
    for (const auto &cost : pose_meas_cost_terms) problem->addCostTerm(cost);
    for (const auto &cost : prior_cost_terms) problem->addCostTerm(cost);
    problem->addCostTerm(preint_cost_term);

    timer[1].second->stop();

    if (N_matches < options_.min_number_keypoints) {
      LOG(ERROR) << "[ICP]Error : not enough keypoints selected in ct-icp !" << std::endl;
      LOG(ERROR) << "[ICP]Number_of_residuals : " << N_matches << std::endl;
      icp_success = false;
      break;
    }

    timer[2].second->start();

    // Solve
    GaussNewtonSolverNVA::Params params;
    params.verbose = options_.verbose;
    params.max_iterations = (unsigned int)options_.max_iterations;
    if (iter >= 2 && options_.use_line_search)
    // if (options_.use_line_search)
      params.line_search = true;
    else
      params.line_search = false;
    if (swf_inside_icp) params.reuse_previous_pattern = false;
    GaussNewtonSolverNVA solver(*problem, params);
    solver.optimize();

    timer[2].second->stop();

    timer[3].second->start();

    double diff_trans = 0, diff_rot = 0, diff_vel = 0;
    const auto mid_T_mr = trajectory_vars_.back().T_mr->evaluate().matrix();
    const auto mid_T_ms = mid_T_mr * options_.T_sr.inverse();
    diff_trans += (current_estimate.end_t - mid_T_ms.block<3, 1>(0, 3)).norm();
    diff_rot += AngularDistance(current_estimate.end_R, mid_T_ms.block<3, 3>(0, 0));
    const auto ve = trajectory_vars_.back().v_rm_inm->evaluate();
    diff_vel += (ve - v_end).norm();
    v_end = ve;

    current_estimate.setMidPose(mid_T_ms);

    current_estimate.begin_R = mid_T_ms.block<3, 3>(0, 0);
    current_estimate.begin_t = mid_T_ms.block<3, 1>(0, 3);
    current_estimate.end_R = mid_T_ms.block<3, 3>(0, 0);
    current_estimate.end_t = mid_T_ms.block<3, 1>(0, 3);

    timer[3].second->stop();

    LOG(INFO) << "diff_trans: " << diff_trans << " diff_rot: " << diff_rot << " diff_vel: " << diff_vel << std::endl;
    std::cout << "T_mr: " << trajectory_vars_.back().T_mr->evaluate().matrix() << std::endl;
    std::cout << "v: " << trajectory_vars_.back().v_rm_inm->evaluate().transpose() << std::endl;
    std::cout << "biases: " << trajectory_vars_.back().imu_biases->evaluate().transpose() << std::endl;

    if ((index_frame > 1) &&
        (diff_rot < options_.threshold_orientation_norm && diff_trans < options_.threshold_translation_norm &&
         diff_vel < options_.threshold_translation_norm * 10)) {
      if (options_.debug_print) {
        LOG(INFO) << "ICP: Finished with N=" << iter << " ICP iterations" << std::endl;
      }
      if (options_.break_icp_early) break;
    }
  }

  LOG(INFO) << "Optimizing in a sliding window!" << std::endl;

  for (const auto &meas_cost_term : meas_cost_terms) sliding_window_filter_->addCostTerm(meas_cost_term);
  for (const auto &pose_meas_cost_term : pose_meas_cost_terms) sliding_window_filter_->addCostTerm(pose_meas_cost_term);
  sliding_window_filter_->addCostTerm(preint_cost_term);
  for (const auto &imu_prior_cost : imu_prior_cost_terms) sliding_window_filter_->addCostTerm(imu_prior_cost);
  for (const auto &prior_cost : prior_cost_terms) sliding_window_filter_->addCostTerm(prior_cost);
#if USE_P2P_SUPER_COST_TERM
  sliding_window_filter_->addCostTerm(p2p_super_cost_term);
#endif

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
  if (!swf_inside_icp) solver.optimize();

  const auto mid_T_mr = trajectory_vars_.back().T_mr->evaluate().matrix();
  const auto mid_T_ms = mid_T_mr * options_.T_sr.inverse();
  current_estimate.setMidPose(mid_T_ms);

  current_estimate.begin_R = mid_T_ms.block<3, 3>(0, 0);
  current_estimate.begin_t = mid_T_ms.block<3, 1>(0, 3);
  current_estimate.end_R = mid_T_ms.block<3, 3>(0, 0);
  current_estimate.end_t = mid_T_ms.block<3, 1>(0, 3);

  LOG(INFO) << "Number of keypoints used in ICP : " << N_matches << std::endl;

  if (options_.debug_print) {
    for (size_t i = 0; i < timer.size(); i++)
      LOG(INFO) << "Elapsed " << timer[i].first << *(timer[i].second) << std::endl;
  }

  std::cout << "T_mr: " << trajectory_vars_.back().T_mr->evaluate().matrix() << std::endl;
  std::cout << "v: " << trajectory_vars_.back().v_rm_inm->evaluate().transpose() << std::endl;
  std::cout << "biases: " << trajectory_vars_.back().imu_biases->evaluate().transpose() << std::endl;

  return icp_success;
}

}  // namespace steam_icp
   