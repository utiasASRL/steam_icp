#include "steam_icp/odometry/spline.hpp"

#include <iomanip>
#include <random>

#include <glog/logging.h>

#include "steam.hpp"

#include "steam_icp/utils/stopwatch.hpp"

namespace steam_icp {

namespace {

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

}  // namespace

SplineOdometry::SplineOdometry(const Options &options) : Odometry(options), options_(options) {
  // iniitalize steam vars
  T_sr_var_ = steam::se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(options_.T_sr));
  T_sr_var_->locked() = true;

  sliding_window_filter_ = steam::SlidingWindowFilter::MakeShared(options_.num_threads);
  spline_trajectory_ = steam::traj::bspline::Interface::MakeShared(steam::traj::Time(options_.knot_spacing));
}

SplineOdometry::~SplineOdometry() {
  using namespace steam::traj;

  std::ofstream trajectory_file;
  // auto now = std::chrono::system_clock::now();
  // auto utc = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
  // trajectory_file.open(options_.debug_path + "/trajectory_" + std::to_string(utc) + ".txt", std::ios::out);
  trajectory_file.open(options_.debug_path + "/trajectory.txt", std::ios::out);

  LOG(INFO) << "Dumping trajectory." << std::endl;
  double begin_time = trajectory_.front().begin_timestamp;
  double end_time = trajectory_.back().end_timestamp;
  double dt = 0.01;
  for (double time = begin_time; time <= end_time; time += dt) {
    Time steam_time(time);
    //
    Eigen::Matrix4d T_rm = Eigen::Matrix4d::Identity();
    const auto w_mr_inr = spline_trajectory_->getVelocityInterpolator(steam_time)->evaluate();
    //
    trajectory_file << (0.0) << " " << steam_time.nanosecs() << " ";
    trajectory_file << T_rm(0, 0) << " " << T_rm(0, 1) << " " << T_rm(0, 2) << " " << T_rm(0, 3) << " ";
    trajectory_file << T_rm(1, 0) << " " << T_rm(1, 1) << " " << T_rm(1, 2) << " " << T_rm(1, 3) << " ";
    trajectory_file << T_rm(2, 0) << " " << T_rm(2, 1) << " " << T_rm(2, 2) << " " << T_rm(2, 3) << " ";
    trajectory_file << T_rm(3, 0) << " " << T_rm(3, 1) << " " << T_rm(3, 2) << " " << T_rm(3, 3) << " ";
    trajectory_file << w_mr_inr(0) << " " << w_mr_inr(1) << " " << w_mr_inr(2) << " " << w_mr_inr(3) << " ";
    trajectory_file << w_mr_inr(4) << " " << w_mr_inr(5) << std::endl;
  }
  LOG(INFO) << "Dumping trajectory. - DONE" << std::endl;
}

auto SplineOdometry::registerFrame(const std::vector<Point3D> &const_frame) -> RegistrationSummary {
  RegistrationSummary summary;

  // add a new frame
  int index_frame = trajectory_.size();
  trajectory_.emplace_back();

  //
  initializeTimestamp(index_frame, const_frame);

  //
  auto frame = initializeFrame(index_frame, const_frame);

  double sample_voxel_size = options_.sample_voxel_size;

  // downsample
  std::vector<Point3D> keypoints;
  grid_sampling(frame, keypoints, sample_voxel_size);

  // estimate
  summary.success = estimateMotion(index_frame, keypoints);
  summary.keypoints = keypoints;

  return summary;
}

void SplineOdometry::initializeTimestamp(int index_frame, const std::vector<Point3D> &const_frame) {
  double min_timestamp = std::numeric_limits<double>::max();
  double max_timestamp = std::numeric_limits<double>::min();
  for (const auto &point : const_frame) {
    if (point.timestamp > max_timestamp) max_timestamp = point.timestamp;
    if (point.timestamp < min_timestamp) min_timestamp = point.timestamp;
  }
  trajectory_[index_frame].begin_timestamp = min_timestamp;
  trajectory_[index_frame].end_timestamp = max_timestamp;
}

std::vector<Point3D> SplineOdometry::initializeFrame(int /* index_frame */, const std::vector<Point3D> &const_frame) {
  std::vector<Point3D> frame(const_frame);
  double sample_size = options_.voxel_size;
  std::mt19937_64 g;
  std::shuffle(frame.begin(), frame.end(), g);
  // Subsample the scan with voxels taking one random in every voxel
  sub_sample_frame(frame, sample_size);
  std::shuffle(frame.begin(), frame.end(), g);

  return frame;
}

bool SplineOdometry::estimateMotion(int index_frame, std::vector<Point3D> &keypoints) {
  using namespace steam;
  using namespace steam::se3;
  using namespace steam::traj;
  using namespace steam::vspace;

  // timers
  std::vector<std::pair<std::string, std::unique_ptr<Stopwatch<>>>> timer;
  timer.emplace_back("Instantiation .................. ", std::make_unique<Stopwatch<>>(false));
  timer.emplace_back("Optimization ................... ", std::make_unique<Stopwatch<>>(false));

  // cost terms passed to the sliding window filter
  std::vector<BaseCostTerm::ConstPtr> prior_cost_terms;
  std::vector<BaseCostTerm::ConstPtr> meas_cost_terms;

  timer[0].second->start();

  // Get evaluator for query points
  std::vector<Evaluable<const_vel::Interface::VelocityType>::ConstPtr> w_ms_ins_intp_eval_vec;
  w_ms_ins_intp_eval_vec.reserve(keypoints.size());
  for (const auto &keypoint : keypoints) {
    const auto query_time =
        trajectory_[index_frame].begin_timestamp +
        keypoint.alpha_timestamp * (trajectory_[index_frame].end_timestamp - trajectory_[index_frame].begin_timestamp);
    // velocity
    const auto w_mr_inr_intp_eval = spline_trajectory_->getVelocityInterpolator(Time(query_time));
    const auto w_ms_ins_intp_eval = compose_velocity(T_sr_var_, w_mr_inr_intp_eval);
    w_ms_ins_intp_eval_vec.emplace_back(w_ms_ins_intp_eval);
  }

  // add velocity cost terms
  Eigen::Matrix<double, 1, 1> W = 1.0 * Eigen::Matrix<double, 1, 1>::Identity();
  const auto noise_model = StaticNoiseModel<1>::MakeShared(W, NoiseType::INFORMATION);
#pragma omp parallel for num_threads(options_.num_threads)
  for (int i = 0; i < (int)keypoints.size(); i++) {
    const auto &keypoint = keypoints[i];

    const auto &w_ms_ins_intp_eval = w_ms_ins_intp_eval_vec[i];
    const auto error_func = p2p::radialVelError(w_ms_ins_intp_eval, keypoint.raw_pt, keypoint.radial_velocity);

    const auto loss_func = [this]() -> BaseLossFunc::Ptr {
      switch (options_.rv_loss_func) {
        case STEAM_LOSS_FUNC::L2:
          return L2LossFunc::MakeShared();
        case STEAM_LOSS_FUNC::DCS:
          return DcsLossFunc::MakeShared(options_.rv_loss_threshold);
        case STEAM_LOSS_FUNC::CAUCHY:
          return CauchyLossFunc::MakeShared(options_.rv_loss_threshold);
        case STEAM_LOSS_FUNC::GM:
          return GemanMcClureLossFunc::MakeShared(options_.rv_loss_threshold);
        default:
          return nullptr;
      }
      return nullptr;
    }();

    const auto cost = WeightedLeastSqCostTerm<1>::MakeShared(error_func, noise_model, loss_func);
#pragma omp critical(odometry_cost_term)
    { meas_cost_terms.emplace_back(cost); }
  }

  const double begin_timestamp = trajectory_[index_frame].begin_timestamp;
  const double end_timestamp = trajectory_[index_frame].end_timestamp;
  LOG(INFO) << "begin time: " << begin_timestamp << std::endl;
  LOG(INFO) << "end time: " << end_timestamp << std::endl;

  const double window_begin_timestamp = begin_timestamp - options_.window_delay;
  LOG(INFO) << "marginalizing knots before time: " << window_begin_timestamp << std::endl;

  // add new prior cost terms
  double curr_prior_time = curr_prior_time_;
  for (; curr_prior_time < end_timestamp; curr_prior_time += options_.vp_spacing) {
    Time query_time(curr_prior_time);
    const auto w_mr_inr_intp_eval = spline_trajectory_->getVelocityInterpolator(query_time);
    const auto error_func = vspace_error<6>(w_mr_inr_intp_eval, Eigen::Matrix<double, 6, 1>::Zero());
    const auto noise_model = StaticNoiseModel<6>::MakeShared(options_.vp_cov);
    const auto loss_func = std::make_shared<L2LossFunc>();
    const auto cost = WeightedLeastSqCostTerm<6>::MakeShared(error_func, noise_model, loss_func);
    prior_cost_terms.emplace_back(cost);
    LOG(INFO) << "adding prior at time: " << query_time.seconds() << std::endl;
  }
  curr_prior_time_ = curr_prior_time;

  // add new variables
  std::vector<StateVarBase::Ptr> new_variables;
  Time end_steam_time = trajectory_vars_.empty() ? Time((double)-10000.0) : trajectory_vars_.back().time;
  const auto &knot_map = spline_trajectory_->knot_map();
  for (auto it = knot_map.lower_bound(end_steam_time); it != knot_map.end(); ++it) {
    if (trajectory_vars_.empty() || (it->first > trajectory_vars_.back().time)) {
      trajectory_vars_.emplace_back(it->first, it->second->getC());
      new_variables.emplace_back(it->second->getC());
      LOG(INFO) << "adding knot at time: " << it->first.seconds() << std::endl;

      // add a weak prior on c vector
      auto error_func = vspace_error<6>(it->second->getC(), Eigen::Matrix<double, 6, 1>::Zero());
      auto noise_model = StaticNoiseModel<6>::MakeShared(Eigen::Matrix<double, 6, 6>::Identity() * options_.c_cov);
      auto loss_func = std::make_shared<L2LossFunc>();
      auto cost = WeightedLeastSqCostTerm<6>::MakeShared(error_func, noise_model, loss_func);
      prior_cost_terms.emplace_back(cost);
    }
  }
  sliding_window_filter_->addStateVariable(new_variables);

  // add cost terms to the sliding window filter
  for (const auto &cost_term : meas_cost_terms) sliding_window_filter_->addCostTerm(cost_term);
  for (const auto &cost_term : prior_cost_terms) sliding_window_filter_->addCostTerm(cost_term);

  // marginalize some variables
  std::vector<StateVarBase::Ptr> marginalize_variables;
  int curr_active_idx = curr_active_idx_;
  for (; curr_active_idx < (int)trajectory_vars_.size(); curr_active_idx++) {
    if (trajectory_vars_.at(curr_active_idx).time.seconds() < window_begin_timestamp) {
      LOG(INFO) << "marginalizing knot at time: " << trajectory_vars_.at(curr_active_idx).time.seconds() << std::endl;
      marginalize_variables.emplace_back(trajectory_vars_.at(curr_active_idx).c);
    } else {
      break;
    }
  }
  curr_active_idx_ = curr_active_idx;
  sliding_window_filter_->marginalizeVariable(marginalize_variables);

  LOG(INFO) << "number of variables: " << sliding_window_filter_->getNumberOfVariables() << std::endl;
  LOG(INFO) << "number of cost terms: " << sliding_window_filter_->getNumberOfCostTerms() << std::endl;
  if (sliding_window_filter_->getNumberOfVariables() > 100)
    throw std::runtime_error{"too many variables in the filter!"};
  if (sliding_window_filter_->getNumberOfCostTerms() > 500000)
    throw std::runtime_error{"too many cost terms in the filter!"};

  timer[0].second->stop();

  timer[1].second->start();

  // Solve
  GaussNewtonSolver::Params params;
  params.max_iterations = options_.max_iterations;
  GaussNewtonSolver solver(*sliding_window_filter_, params);
  solver.optimize();
  timer[1].second->stop();

  /// Debug print
  if (options_.debug_print) {
    for (size_t i = 0; i < timer.size(); i++)
      LOG(INFO) << "Elapsed " << timer[i].first << *(timer[i].second) << std::endl;
  }

  return true;
}

}  // namespace steam_icp
