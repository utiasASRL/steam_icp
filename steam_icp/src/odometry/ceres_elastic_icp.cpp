#include "steam_icp/odometry/ceres_elastic_icp.hpp"

#include <iomanip>
#include <random>

#include <ceres/ceres.h>
#include <glog/logging.h>

#include "steam_icp/utils/stopwatch.hpp"

namespace steam_icp {

namespace {

Eigen::Matrix4d getMidPose(const Eigen::Matrix3d &begin_R, const Eigen::Matrix3d &end_R, const Eigen::Vector3d &begin_t,
                           const Eigen::Vector3d &end_t) {
  Eigen::Matrix4d mid_pose = Eigen::Matrix4d::Identity();
  const auto q_begin = Eigen::Quaterniond(begin_R);
  const auto q_end = Eigen::Quaterniond(end_R);
  Eigen::Quaterniond q = q_begin.slerp(0.5, q_end);
  q.normalize();
  mid_pose.block<3, 3>(0, 0) = q.toRotationMatrix();
  mid_pose.block<3, 1>(0, 3) = 0.5 * begin_t + 0.5 * end_t;
  return mid_pose;
}

inline double AngularDistance(const Eigen::Matrix3d &rota, const Eigen::Matrix3d &rotb) {
  double norm = ((rota * rotb.transpose()).trace() - 1) / 2;
  norm = std::acos(norm) * 180 / M_PI;
  return norm;
}

/* -------------------------------------------------------------------------------------------------------------- */
// Subsample to keep one random point in every voxel of the current frame
void sub_sample_frame(std::vector<Point3D> &frame, double size_voxel) {
  std::unordered_map<Voxel, std::vector<Point3D>> grid;
  for (int i = 0; i < (int)frame.size(); i++) {
    auto kx = static_cast<short>(frame[i].pt[0] / size_voxel);
    auto ky = static_cast<short>(frame[i].pt[1] / size_voxel);
    auto kz = static_cast<short>(frame[i].pt[2] / size_voxel);
    grid[Voxel(kx, ky, kz)].push_back(frame[i]);
  }
  frame.resize(0);
  int step = 0;  // to take one random point inside each voxel (but with identical results when lunching the SLAM a
                 // second time)
  for (const auto &n : grid) {
    if (n.second.size() > 0) {
      // frame.push_back(n.second[step % (int)n.second.size()]);
      frame.push_back(n.second[0]);
      step++;
    }
  }
}

/* -------------------------------------------------------------------------------------------------------------- */
void grid_sampling(const std::vector<Point3D> &frame, std::vector<Point3D> &keypoints, double size_voxel_subsampling) {
  keypoints.resize(0);
  std::vector<Point3D> frame_sub;
  frame_sub.resize(frame.size());
  for (int i = 0; i < (int)frame_sub.size(); i++) {
    frame_sub[i] = frame[i];
  }
  sub_sample_frame(frame_sub, size_voxel_subsampling);
  keypoints.reserve(frame_sub.size());
  for (int i = 0; i < (int)frame_sub.size(); i++) {
    keypoints.push_back(frame_sub[i]);
  }
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

// A Const Functor for the Continuous time Point-to-Plane
struct CTPointToPlaneFunctor {
  static constexpr int NumResiduals() { return 1; }

  CTPointToPlaneFunctor(const Eigen::Vector3d &reference_point, const Eigen::Vector3d &raw_target,
                        const Eigen::Vector3d &reference_normal, double alpha_timestamp, double weight = 1.0)
      : raw_keypoint_(raw_target),
        reference_point_(reference_point),
        reference_normal_(reference_normal),
        alpha_timestamps_(alpha_timestamp),
        weight_(weight) {}

  template <typename T>
  bool operator()(const T *const begin_rot_params, const T *begin_trans_params, const T *const end_rot_params,
                  const T *end_trans_params, T *residual) const {
    Eigen::Map<Eigen::Quaternion<T>> quat_begin(const_cast<T *>(begin_rot_params));
    Eigen::Map<Eigen::Quaternion<T>> quat_end(const_cast<T *>(end_rot_params));
    Eigen::Quaternion<T> quat_inter = quat_begin.slerp(T(alpha_timestamps_), quat_end);
    quat_inter.normalize();

    Eigen::Matrix<T, 3, 1> transformed = quat_inter * raw_keypoint_.template cast<T>();

    T alpha_m = T(1.0 - alpha_timestamps_);
    transformed(0, 0) += alpha_m * begin_trans_params[0] + alpha_timestamps_ * end_trans_params[0];
    transformed(1, 0) += alpha_m * begin_trans_params[1] + alpha_timestamps_ * end_trans_params[1];
    transformed(2, 0) += alpha_m * begin_trans_params[2] + alpha_timestamps_ * end_trans_params[2];

    residual[0] = weight_ * (reference_point_.template cast<T>() - transformed).transpose() *
                  reference_normal_.template cast<T>();

    return true;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Vector3d raw_keypoint_;
  Eigen::Vector3d reference_point_;
  Eigen::Vector3d reference_normal_;
  double alpha_timestamps_;
  double weight_ = 1.0;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// REGULARISATION COST FUNCTORS
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// A Const Functor which enforces Frame consistency between two poses
struct LocationConsistencyFunctor {
  static constexpr int NumResiduals() { return 3; }

  LocationConsistencyFunctor(const Eigen::Vector3d &previous_location, double beta)
      : previous_location_(previous_location), beta_(beta) {}

  template <typename T>
  bool operator()(const T *const location_params, T *residual) const {
    residual[0] = beta_ * (location_params[0] - previous_location_(0, 0));
    residual[1] = beta_ * (location_params[1] - previous_location_(1, 0));
    residual[2] = beta_ * (location_params[2] - previous_location_(2, 0));
    return true;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  Eigen::Vector3d previous_location_;
  double beta_ = 1.0;
};

// A Functor which enforces frame orientation consistency between two poses
struct OrientationConsistencyFunctor {
  static constexpr int NumResiduals() { return 1; }

  OrientationConsistencyFunctor(const Eigen::Quaterniond &previous_orientation, double beta)
      : previous_orientation_(previous_orientation), beta_(beta) {}

  template <typename T>
  bool operator()(const T *const orientation_params, T *residual) const {
    Eigen::Quaternion<T> quat(orientation_params);
    T scalar_quat = quat.dot(previous_orientation_.template cast<T>());
    residual[0] = T(beta_) * (T(1.0) - scalar_quat * scalar_quat);
    return true;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  Eigen::Quaterniond previous_orientation_;
  double beta_;
};

// A Const Functor which enforces a Constant Velocity constraint on translation
struct ConstantVelocityFunctor {
  static constexpr int NumResiduals() { return 3; }

  ConstantVelocityFunctor(const Eigen::Vector3d &previous_velocity, double beta)
      : previous_velocity_(previous_velocity), beta_(beta) {}

  template <typename T>
  bool operator()(const T *const begin_t, const T *const end_t, T *residual) const {
    residual[0] = beta_ * (end_t[0] - begin_t[0] - previous_velocity_(0, 0));
    residual[1] = beta_ * (end_t[1] - begin_t[1] - previous_velocity_(1, 0));
    residual[2] = beta_ * (end_t[2] - begin_t[2] - previous_velocity_(2, 0));
    return true;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  Eigen::Vector3d previous_velocity_;
  double beta_ = 1.0;
};

// A Const Functor which enforces a Small Velocity constraint
struct SmallVelocityFunctor {
  static constexpr int NumResiduals() { return 3; }

  SmallVelocityFunctor(double beta) : beta_(beta){};

  template <typename T>
  bool operator()(const T *const begin_t, const T *const end_t, T *residual) const {
    residual[0] = beta_ * (begin_t[0] - end_t[0]);
    residual[1] = beta_ * (begin_t[1] - end_t[1]);
    residual[2] = beta_ * (begin_t[2] - end_t[2]);
    return true;
  }

  double beta_;
};
/* -------------------------------------------------------------------------------------------------------------- */
// A Builder to abstract the different configurations of ICP optimization
class ICPOptimizationBuilder {
 public:
  using CTICP_PointToPlaneResidual = ceres::AutoDiffCostFunction<CTPointToPlaneFunctor, 1, 4, 3, 4, 3>;

  explicit ICPOptimizationBuilder(const CeresElasticOdometry::Options &options, const std::vector<Point3D> &points)
      : options_(options), keypoints(points) {
    corrected_raw_points_.reserve(keypoints.size());
    for (const auto &point : keypoints) corrected_raw_points_.emplace_back(point.raw_pt);
  }

  bool InitProblem(int num_residuals) {
    problem = std::make_unique<ceres::Problem>();
    parameter_block_set_ = false;

    // Select Loss function
    switch (options_.loss_function) {
      case CeresElasticOdometry::CERES_LOSS_FUNC::L2:
        break;
      case CeresElasticOdometry::CERES_LOSS_FUNC::CAUCHY:
        loss_function = new ceres::CauchyLoss(options_.sigma);
        break;
      case CeresElasticOdometry::CERES_LOSS_FUNC::HUBER:
        loss_function = new ceres::HuberLoss(options_.sigma);
        break;
      case CeresElasticOdometry::CERES_LOSS_FUNC::TOLERANT:
        loss_function = new ceres::TolerantLoss(options_.tolerant_min_threshold, options_.sigma);
        break;
    }

    // Resize the number of residuals
    vector_ct_icp_residuals_.resize(num_residuals);
    vector_cost_functors_.resize(num_residuals);
    begin_quat_ = nullptr;
    end_quat_ = nullptr;
    begin_t_ = nullptr;
    end_t_ = nullptr;

    return true;
  }

  void AddParameterBlocks(Eigen::Quaterniond &begin_quat, Eigen::Quaterniond &end_quat, Eigen::Vector3d &begin_t,
                          Eigen::Vector3d &end_t) {
    if (parameter_block_set_) throw std::runtime_error{"The parameter block was already set"};
    auto *parameterization = new ceres::EigenQuaternionParameterization();
    begin_t_ = &begin_t.x();
    end_t_ = &end_t.x();
    begin_quat_ = &begin_quat.x();
    end_quat_ = &end_quat.x();

    problem->AddParameterBlock(begin_quat_, 4, parameterization);
    problem->AddParameterBlock(end_quat_, 4, parameterization);
    problem->AddParameterBlock(begin_t_, 3);
    problem->AddParameterBlock(end_t_, 3);

    parameter_block_set_ = true;
  }

  void SetResidualBlock(int keypoint_id, const Eigen::Vector3d &reference_point,
                        const Eigen::Vector3d &reference_normal, double weight = 1.0, double alpha_timestamp = -1.0) {
    if (alpha_timestamp < 0 || alpha_timestamp > 1) throw std::runtime_error("BAD ALPHA TIMESTAMP !");

    CTPointToPlaneFunctor *ct_point_to_plane_functor = new CTPointToPlaneFunctor(
        reference_point, corrected_raw_points_[keypoint_id], reference_normal, alpha_timestamp, weight);
    void *cost_functor = ct_point_to_plane_functor;
    void *cost_function = static_cast<void *>(new CTICP_PointToPlaneResidual(ct_point_to_plane_functor));

    vector_ct_icp_residuals_[keypoint_id] = cost_function;
    vector_cost_functors_[keypoint_id] = cost_functor;
  }

  std::unique_ptr<ceres::Problem> GetProblem() {
    for (auto &pt_to_plane_residual : vector_ct_icp_residuals_) {
      if (pt_to_plane_residual != nullptr) {
        problem->AddResidualBlock(static_cast<CTICP_PointToPlaneResidual *>(pt_to_plane_residual), loss_function,
                                  begin_quat_, begin_t_, end_quat_, end_t_);
      }
    }

    std::fill(vector_cost_functors_.begin(), vector_cost_functors_.end(), nullptr);
    std::fill(vector_ct_icp_residuals_.begin(), vector_ct_icp_residuals_.end(), nullptr);

    return std::move(problem);
  }

 private:
  const CeresElasticOdometry::Options &options_;
  std::unique_ptr<ceres::Problem> problem = nullptr;

  // Pointers managed by ceres
  const std::vector<Point3D> &keypoints;
  std::vector<Eigen::Vector3d> corrected_raw_points_;

  // Parameters block pointers
  bool parameter_block_set_ = false;
  double *begin_quat_ = nullptr;
  double *end_quat_ = nullptr;
  double *begin_t_ = nullptr;
  double *end_t_ = nullptr;

  std::vector<void *> vector_cost_functors_;
  std::vector<void *> vector_ct_icp_residuals_;
  ceres::LossFunction *loss_function = nullptr;
};

}  // namespace

CeresElasticOdometry::CeresElasticOdometry(const Options &options) : Odometry(options), options_(options) {}

CeresElasticOdometry::~CeresElasticOdometry() {
  std::ofstream trajectory_file;
  auto now = std::chrono::system_clock::now();
  auto utc = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
  trajectory_file.open(options_.debug_path + "/trajectory_" + std::to_string(utc) + ".txt", std::ios::out);
  // trajectory_file.open(options_.debug_path + "/trajectory.txt", std::ios::out);

  LOG(INFO) << "Dumping trajectory." << std::endl;
  for (const auto &frame : trajectory_) {
    // clang-format off
    double begin_time = frame.begin_timestamp;
    double end_time = frame.end_timestamp;

    Eigen::Matrix4d begin_T_ms = Eigen::Matrix4d::Identity();
    begin_T_ms.block<3, 3>(0, 0) = frame.begin_R;
    begin_T_ms.block<3, 1>(0, 3) = frame.begin_t;
    Eigen::Matrix4d begin_T_sm = begin_T_ms.inverse();
    Eigen::Matrix4d end_T_ms = Eigen::Matrix4d::Identity();
    end_T_ms.block<3, 3>(0, 0) = frame.end_R;
    end_T_ms.block<3, 1>(0, 3) = frame.end_t;
    Eigen::Matrix4d end_T_sm = end_T_ms.inverse();
    Eigen::Matrix<double, 6, 1> w_ms_ins = lgmath::se3::tran2vec(end_T_sm * begin_T_ms) / (end_time - begin_time);

    trajectory_file << std::fixed << std::setprecision(12) << (0.0) << " " << static_cast<int64_t>(begin_time * 1e6) * 1000 << " "
                    << begin_T_sm(0, 0) << " " << begin_T_sm(0, 1) << " " << begin_T_sm(0, 2) << " " << begin_T_sm(0, 3) << " "
                    << begin_T_sm(1, 0) << " " << begin_T_sm(1, 1) << " " << begin_T_sm(1, 2) << " " << begin_T_sm(1, 3) << " "
                    << begin_T_sm(2, 0) << " " << begin_T_sm(2, 1) << " " << begin_T_sm(2, 2) << " " << begin_T_sm(2, 3) << " "
                    << begin_T_sm(3, 0) << " " << begin_T_sm(3, 1) << " " << begin_T_sm(3, 2) << " " << begin_T_sm(3, 3) << " "
                    << w_ms_ins(0) << " " << w_ms_ins(1) << " " << w_ms_ins(2) << " " << w_ms_ins(3) << " " << w_ms_ins(4) << " " << w_ms_ins(5) << std::endl;
    trajectory_file << std::fixed << std::setprecision(12) << (0.0) << " " << static_cast<int64_t>(end_time * 1e6) * 1000 << " "
                    << end_T_sm(0, 0) << " " << end_T_sm(0, 1) << " " << end_T_sm(0, 2) << " " << end_T_sm(0, 3) << " "
                    << end_T_sm(1, 0) << " " << end_T_sm(1, 1) << " " << end_T_sm(1, 2) << " " << end_T_sm(1, 3) << " "
                    << end_T_sm(2, 0) << " " << end_T_sm(2, 1) << " " << end_T_sm(2, 2) << " " << end_T_sm(2, 3) << " "
                    << end_T_sm(3, 0) << " " << end_T_sm(3, 1) << " " << end_T_sm(3, 2) << " " << end_T_sm(3, 3) << " "
                    << w_ms_ins(0) << " " << w_ms_ins(1) << " " << w_ms_ins(2) << " " << w_ms_ins(3) << " " << w_ms_ins(4) << " " << w_ms_ins(5) << std::endl;
    // clang-format on
  }
  LOG(INFO) << "Dumping trajectory. - DONE" << std::endl;
}

Trajectory CeresElasticOdometry::trajectory() { return trajectory_; }

auto CeresElasticOdometry::registerFrame(const std::pair<double, std::vector<Point3D>> &const_frame)
    -> RegistrationSummary {
  RegistrationSummary summary;

  // add a new frame
  int index_frame = trajectory_.size();
  trajectory_.emplace_back();

  //
  initializeTimestamp(index_frame, const_frame);

  //
  initializeMotion(index_frame);

  //
  auto frame = initializeFrame(index_frame, const_frame.second);

  //
  if (index_frame > 0) {
    double sample_voxel_size =
        index_frame < options_.init_num_frames ? options_.init_sample_voxel_size : options_.sample_voxel_size;

    // downsample
    std::vector<Point3D> keypoints;
    grid_sampling(frame, keypoints, sample_voxel_size);

    // icp
    summary.success = icp(index_frame, keypoints);
    summary.keypoints = keypoints;
    if (!summary.success) return summary;
  } else {
    summary.success = true;
  }
  trajectory_[index_frame].points = frame;

  // add points
  updateMap(index_frame, index_frame);

  summary.corrected_points = frame;

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

void CeresElasticOdometry::initializeTimestamp(int index_frame,
                                               const std::pair<double, std::vector<Point3D>> &const_frame) {
  double min_timestamp = std::numeric_limits<double>::max();
  double max_timestamp = std::numeric_limits<double>::min();
  for (const auto &point : const_frame.second) {
    if (point.timestamp > max_timestamp) max_timestamp = point.timestamp;
    if (point.timestamp < min_timestamp) min_timestamp = point.timestamp;
  }
  trajectory_[index_frame].begin_timestamp = min_timestamp;
  trajectory_[index_frame].end_timestamp = max_timestamp;
}

void CeresElasticOdometry::initializeMotion(int index_frame) {
  if (index_frame <= 1) {
    // Initialize first pose at Identity
    trajectory_[index_frame].begin_R = Eigen::MatrixXd::Identity(3, 3);
    trajectory_[index_frame].begin_t = Eigen::Vector3d(0., 0., 0.);
    trajectory_[index_frame].end_R = Eigen::MatrixXd::Identity(3, 3);
    trajectory_[index_frame].end_t = Eigen::Vector3d(0., 0., 0.);
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

std::vector<Point3D> CeresElasticOdometry::initializeFrame(int index_frame, const std::vector<Point3D> &const_frame) {
  std::vector<Point3D> frame(const_frame);

  double sample_size = index_frame < options_.init_num_frames ? options_.init_voxel_size : options_.voxel_size;
  std::mt19937_64 g;
  std::shuffle(frame.begin(), frame.end(), g);
  // Subsample the scan with voxels taking one random in every voxel
  sub_sample_frame(frame, sample_size);
  std::shuffle(frame.begin(), frame.end(), g);

  // No elastic ICP for first frame because no initialization of ego-motion
  if (index_frame == 1) {
    for (auto &point : frame) point.alpha_timestamp = 1.0;
  }

  // initialize points
  if (index_frame > 1) {
    auto q_begin = Eigen::Quaterniond(trajectory_[index_frame].begin_R);
    auto q_end = Eigen::Quaterniond(trajectory_[index_frame].end_R);
    Eigen::Vector3d t_begin = trajectory_[index_frame].begin_t;
    Eigen::Vector3d t_end = trajectory_[index_frame].end_t;
    for (auto &point : frame) {
      double alpha_timestamp = point.alpha_timestamp;
      Eigen::Matrix3d R = q_begin.slerp(alpha_timestamp, q_end).normalized().toRotationMatrix();
      Eigen::Vector3d t = (1.0 - alpha_timestamp) * t_begin + alpha_timestamp * t_end;
      //
      point.pt = R * point.raw_pt + t;
    }
  }

  return frame;
}

void CeresElasticOdometry::updateMap(int index_frame, int update_frame) {
  const double kSizeVoxelMap = options_.size_voxel_map;
  const double kMinDistancePoints = options_.min_distance_points;
  const int kMaxNumPointsInVoxel = options_.max_num_points_in_voxel;

  // update frame
  auto &frame = trajectory_[update_frame].points;

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

  map_.add(frame, kSizeVoxelMap, kMaxNumPointsInVoxel, kMinDistancePoints);
  frame.clear();
  frame.shrink_to_fit();

  // remove points
  const double kMaxDistance = options_.max_distance;
  const Eigen::Vector3d location = trajectory_[index_frame].end_t;
  map_.remove(location, kMaxDistance);
}

bool CeresElasticOdometry::icp(int index_frame, std::vector<Point3D> &keypoints) {
  bool icp_success = true;

  // For the 50 first frames, visit 2 voxels
  const short nb_voxels_visited = index_frame < options_.init_num_frames ? 2 : 1;
  const double kMaxPointToPlane = options_.max_dist_to_plane;
  const int kMinNumNeighbors = options_.min_number_neighbors;

  ceres::Solver::Options ceres_options;
  ceres_options.max_num_iterations = options_.max_iterations;
  ceres_options.num_threads = options_.num_threads;
  ceres_options.trust_region_strategy_type = ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
  ICPOptimizationBuilder builder(options_, keypoints);

  const auto &previous_estimate = trajectory_.at(index_frame - 1);
  const Eigen::Vector3d previous_velocity = previous_estimate.end_t - previous_estimate.begin_t;
  const Eigen::Quaterniond previous_orientation = Eigen::Quaterniond(previous_estimate.end_R);

  auto &current_estimate = trajectory_.at(index_frame);
  Eigen::Quaterniond begin_quat = Eigen::Quaterniond(current_estimate.begin_R);
  Eigen::Quaterniond end_quat = Eigen::Quaterniond(current_estimate.end_R);
  Eigen::Vector3d begin_t = current_estimate.begin_t;
  Eigen::Vector3d end_t = current_estimate.end_t;

  // timers
  std::vector<std::pair<std::string, std::unique_ptr<Stopwatch<>>>> timer;
  timer.emplace_back("Update Transform ............... ", std::make_unique<Stopwatch<>>(false));
  timer.emplace_back("Association .................... ", std::make_unique<Stopwatch<>>(false));
  timer.emplace_back("Optimization ................... ", std::make_unique<Stopwatch<>>(false));
  timer.emplace_back("Alignment ...................... ", std::make_unique<Stopwatch<>>(false));
  std::vector<std::pair<std::string, std::unique_ptr<Stopwatch<>>>> inner_timer;
  inner_timer.emplace_back("Search Neighbors ............. ", std::make_unique<Stopwatch<>>(false));
  inner_timer.emplace_back("Compute Normal ............... ", std::make_unique<Stopwatch<>>(false));
  inner_timer.emplace_back("Add Cost Term ................ ", std::make_unique<Stopwatch<>>(false));
  bool innerloop_time = (options_.num_threads == 1);

  int number_keypoints_used = 0;

  auto transform_keypoints = [&]() {
#pragma omp parallel for num_threads(options_.num_threads)
    for (auto &keypoint : keypoints) {
      const double &alpha_timestamp = keypoint.alpha_timestamp;
      Eigen::Quaterniond q = begin_quat.slerp(alpha_timestamp, end_quat);
      q.normalize();
      Eigen::Matrix3d R = q.toRotationMatrix();
      Eigen::Vector3d t = (1.0 - alpha_timestamp) * begin_t + alpha_timestamp * end_t;
      keypoint.pt = R * keypoint.raw_pt + t;
    }
  };

  double lambda_weight = std::abs(options_.weight_alpha);
  double lambda_neighborhood = std::abs(options_.weight_neighborhood);
  const double sum = lambda_weight + lambda_neighborhood;
  if (sum <= 0.0) throw std::runtime_error("sum of lambda_weight and lambda_neighborhood must be positive");
  lambda_weight /= sum;
  lambda_neighborhood /= sum;

  //
  int num_iter_icp = index_frame < options_.init_num_frames ? 15 : options_.num_iters_icp;
  for (int iter(0); iter < num_iter_icp; iter++) {
    builder.InitProblem(keypoints.size());
    builder.AddParameterBlocks(begin_quat, end_quat, begin_t, end_t);

    number_keypoints_used = 0;

    timer[0].second->start();
    transform_keypoints();
    timer[0].second->stop();

    timer[1].second->start();

#pragma omp parallel for num_threads(options_.num_threads)
    for (int i = 0; i < (int)keypoints.size(); i++) {
      const auto &keypoint = keypoints[i];
      const auto &pt_keypoint = keypoint.pt;
      const auto &alpha_timestamp = keypoint.alpha_timestamp;

      if (innerloop_time) inner_timer[0].second->start();

      // Neighborhood search
      ArrayVector3d vector_neighbors =
          map_.searchNeighbors(pt_keypoint, nb_voxels_visited, options_.size_voxel_map, options_.max_number_neighbors);

      if (innerloop_time) inner_timer[0].second->stop();

      if ((int)vector_neighbors.size() < kMinNumNeighbors) {
        continue;
      }

      if (innerloop_time) inner_timer[1].second->start();

      // Compute normals from neighbors
      auto neighborhood = compute_neighborhood_distribution(vector_neighbors);

      if (neighborhood.normal.dot(current_estimate.begin_t - pt_keypoint) < 0) {
        neighborhood.normal = -1.0 * neighborhood.normal;
      }

      const double planarity_weight = std::pow(neighborhood.a2D, options_.power_planarity);
      const double weight = lambda_weight * planarity_weight +
                            lambda_neighborhood * std::exp(-(vector_neighbors[0] - pt_keypoint).norm() /
                                                           (kMaxPointToPlane * kMinNumNeighbors));

      if (innerloop_time) inner_timer[1].second->stop();

      if (innerloop_time) inner_timer[2].second->start();

      const double dist_to_plane = std::abs((keypoint.pt - vector_neighbors[0]).transpose() * neighborhood.normal);
      if (dist_to_plane < kMaxPointToPlane) {
        builder.SetResidualBlock(i, vector_neighbors[0], neighborhood.normal, weight, alpha_timestamp);
#pragma omp critical(odometry_cost_term)
        { number_keypoints_used++; }
      }

      if (innerloop_time) inner_timer[2].second->stop();
    }

    timer[1].second->stop();

    if (number_keypoints_used < 100) {
      LOG(ERROR) << "[CT_ICP]Error : not enough keypoints selected in ct-icp !" << std::endl;
      LOG(ERROR) << "[CT_ICP]Number_of_residuals : " << number_keypoints_used << std::endl;
      icp_success = false;
      break;
    }

    timer[2].second->start();

    auto problem = builder.GetProblem();

    // Add constraints in trajectory
    if (index_frame > 1)  // no constraints for frame_index == 1
    {
      // Add Regularisation residuals
      problem->AddResidualBlock(
          new ceres::AutoDiffCostFunction<LocationConsistencyFunctor, LocationConsistencyFunctor::NumResiduals(), 3>(
              new LocationConsistencyFunctor(previous_estimate.end_t,
                                             std::sqrt(number_keypoints_used * options_.beta_location_consistency))),
          nullptr, &begin_t.x());
      problem->AddResidualBlock(
          new ceres::AutoDiffCostFunction<ConstantVelocityFunctor, ConstantVelocityFunctor::NumResiduals(), 3, 3>(
              new ConstantVelocityFunctor(previous_velocity,
                                          std::sqrt(number_keypoints_used * options_.beta_constant_velocity))),
          nullptr, &begin_t.x(), &end_t.x());

      // SMALL VELOCITY
      problem->AddResidualBlock(
          new ceres::AutoDiffCostFunction<SmallVelocityFunctor, SmallVelocityFunctor::NumResiduals(), 3, 3>(
              new SmallVelocityFunctor(std::sqrt(number_keypoints_used * options_.beta_small_velocity))),
          nullptr, &begin_t.x(), &end_t.x());

      // ORIENTATION CONSISTENCY
      problem->AddResidualBlock(
          new ceres::AutoDiffCostFunction<OrientationConsistencyFunctor, OrientationConsistencyFunctor::NumResiduals(),
                                          4>(new OrientationConsistencyFunctor(
              previous_orientation, sqrt(number_keypoints_used * options_.beta_orientation_consistency))),
          nullptr, &begin_quat.x());
    }

    ceres::Solver::Summary summary;
    ceres::Solve(ceres_options, problem.get(), &summary);
    if (!summary.IsSolutionUsable()) {
      std::cout << summary.FullReport() << std::endl;
      throw std::runtime_error("Error During Optimization");
    }
    if (options_.debug_print) {
      std::cout << summary.BriefReport() << std::endl;
    }

    timer[2].second->stop();

    timer[3].second->start();

    // Update (changes trajectory data)
    begin_quat.normalize();
    end_quat.normalize();

    double diff_trans = (current_estimate.begin_t - begin_t).norm() + (current_estimate.end_t - end_t).norm();
    double diff_rot = AngularDistance(current_estimate.begin_R, begin_quat.toRotationMatrix()) +
                      AngularDistance(current_estimate.end_R, end_quat.toRotationMatrix());

    current_estimate.begin_R = begin_quat.toRotationMatrix();
    current_estimate.begin_t = begin_t;
    current_estimate.end_R = end_quat.toRotationMatrix();
    current_estimate.end_t = end_t;
    current_estimate.setMidPose(
        getMidPose(current_estimate.begin_R, current_estimate.end_R, current_estimate.begin_t, current_estimate.end_t));

    timer[3].second->stop();

    if ((index_frame > 1) &&
        (diff_rot < options_.threshold_orientation_norm && diff_trans < options_.threshold_translation_norm)) {
      if (options_.debug_print) {
        LOG(INFO) << "CT_ICP: Finished with N=" << iter << " ICP iterations" << std::endl;
      }
      break;
    }
  }

  timer[0].second->start();
  transform_keypoints();
  timer[0].second->stop();

  LOG(INFO) << "Number of keypoints used in CT-ICP : " << number_keypoints_used << std::endl;

  /// Debug print
  if (options_.debug_print) {
    for (size_t i = 0; i < timer.size(); i++)
      LOG(INFO) << "Elapsed " << timer[i].first << *(timer[i].second) << std::endl;
    for (size_t i = 0; i < inner_timer.size(); i++)
      LOG(INFO) << "Elapsed (Inner Loop) " << inner_timer[i].first << *(inner_timer[i].second) << std::endl;
    LOG(INFO) << "Number iterations CT-ICP : " << options_.num_iters_icp << std::endl;
    LOG(INFO) << "Translation Begin: " << trajectory_[index_frame].begin_t.transpose() << std::endl;
    LOG(INFO) << "Translation End: " << trajectory_[index_frame].end_t.transpose() << std::endl;
  }

  return icp_success;
}

}  // namespace steam_icp
