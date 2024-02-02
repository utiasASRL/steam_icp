#include "steam_icp/datasets/utils.hpp"

#include <fstream>
#include <iomanip>

#include "lgmath.hpp"

namespace steam_icp {

namespace {

double translationError(const Eigen::Matrix4d &pose_error) { return pose_error.block<3, 1>(0, 3).norm(); }

double translationError2D(const Eigen::Matrix4d &pose_error) { return pose_error.block<2, 1>(0, 3).norm(); }

double rotationError(const Eigen::Matrix4d &pose_error) {
  double a = pose_error(0, 0);
  double b = pose_error(1, 1);
  double c = pose_error(2, 2);
  double d = 0.5 * (a + b + c - 1.0);
  return std::acos(std::max(std::min(d, 1.0), -1.0));
}

double rotationError2D(Eigen::Matrix4d &pose_error) {
  auto pose_error_projected = lgmath::se3::tran2vec(pose_error);
  pose_error_projected.segment<3>(2) = Eigen::Vector3d::Zero();
  return rotationError(lgmath::se3::vec2tran(pose_error_projected));
}

Eigen::Matrix4d projectTo2d(const Eigen::Matrix4d &T_12) {
  Eigen::Matrix4d T_12_out = Eigen::Matrix4d::Identity();
  T_12_out.block<2, 1>(0, 3) = T_12.block<2, 1>(0, 3);
  Eigen::Vector3d xbar;
  xbar << 1, 0, 0;
  xbar = T_12.block<3, 3>(0, 0) * xbar;
  const double phi = -atan2f(xbar(1, 0), xbar(0, 0));
  T_12_out(0, 0) = T_12_out(1, 1) = cos(phi);
  T_12_out(0, 1) = sin(phi);
  T_12_out(1, 0) = -T_12_out(0, 1);
  return T_12_out;
}

std::vector<double> trajectoryDistances(const ArrayPoses &poses) {
  std::vector<double> dist(1, 0.0);
  for (size_t i = 1; i < poses.size(); i++) dist.push_back(dist[i - 1] + translationError(poses[i - 1] - poses[i]));
  return dist;
}

int lastFrameFromSegmentLength(const std::vector<double> &dist, int first_frame, double len) {
  for (int i = first_frame; i < (int)dist.size(); i++)
    if (dist[i] > dist[first_frame] + len) return i;
  return -1;
}

void computeMeanRPE(const ArrayPoses &poses_gt, const ArrayPoses &poses_result, Sequence::SeqError &seq_err,
                    int step_size) {
  // static parameter
  double lengths[] = {100, 200, 300, 400, 500, 600, 700, 800};
  size_t num_lengths = sizeof(lengths) / sizeof(double);

  // pre-compute distances (from ground truth as reference)
  std::vector<double> dist = trajectoryDistances(poses_gt);

  int num_total = 0;
  double mean_t_rpe = 0;
  double mean_t_rpe_2d = 0;
  double mean_r_rpe = 0;
  double mean_r_rpe_2d = 0;
  // for all start positions do
  for (int first_frame = 0; first_frame < (int)poses_gt.size(); first_frame += step_size) {
    // for all segment lengths do
    for (size_t i = 0; i < num_lengths; i++) {
      // current length
      double len = lengths[i];

      // compute last frame
      int last_frame = lastFrameFromSegmentLength(dist, first_frame, len);

      // next frame if sequence not long enough
      if (last_frame == -1) continue;

      // compute translational errors
      Eigen::Matrix4d pose_delta_gt = poses_gt[first_frame].inverse() * poses_gt[last_frame];
      Eigen::Matrix4d pose_delta_result = poses_result[first_frame].inverse() * poses_result[last_frame];
      Eigen::Matrix4d pose_error = pose_delta_result.inverse() * pose_delta_gt;
      double t_err = translationError(pose_error);
      double r_err = rotationError(pose_error);

      /*
      double t_err_2d = translationError2D(pose_error);
      double r_err_2d = rotationError2D(pose_error);
      */

      Eigen::Matrix4d pose_delta_gt_2d = projectTo2d(pose_delta_gt);
      Eigen::Matrix4d pose_delta_result_2d = projectTo2d(pose_delta_result);
      Eigen::Matrix4d pose_error_2d = pose_delta_result_2d.inverse() * pose_delta_gt_2d;
      double t_err_2d = translationError2D(pose_error_2d);
      double r_err_2d = rotationError(pose_error_2d);
      seq_err.tab_errors.emplace_back(t_err / len, r_err / len, t_err_2d, r_err_2d, len);

      mean_t_rpe += t_err / len;
      mean_t_rpe_2d += t_err_2d / len;
      mean_r_rpe += r_err / len;
      mean_r_rpe_2d += r_err_2d / len;
      num_total++;
    }
  }

  seq_err.mean_t_rpe = ((mean_t_rpe / static_cast<double>(num_total)) * 100.0);
  seq_err.mean_t_rpe_2d = ((mean_t_rpe_2d / static_cast<double>(num_total)) * 100.0);
  seq_err.mean_r_rpe = ((mean_r_rpe / static_cast<double>(num_total)) * 180.0 / M_PI);
  seq_err.mean_r_rpe_2d = ((mean_r_rpe_2d / static_cast<double>(num_total)) * 180.0 / M_PI);
}

// void computeMeanRPELocal(const ArrayPoses &poses_gt, const ArrayPoses &poses_result, Sequence::SeqError &seq_err,
//                          int step_size) {
//   // static parameter
//   // pre-compute distances (from ground truth as reference)
//   int num_total = 0;
//   double mean_t_rpe = 0;
//   double mean_t_rpe_2d = 0;
//   double mean_r_rpe = 0;
//   double mean_r_rpe_2d = 0;
//   // for all start positions do
//   for (int first_frame = 0; first_frame < (int)poses_gt.size(); first_frame++) {
//     int last_frame = first_frame + step_size;
//     if (last_frame >= (int)poses_gt.size()) continue;
//     // compute translational errors
//     Eigen::Matrix4d pose_delta_gt = poses_gt[first_frame].inverse() * poses_gt[last_frame];
//     Eigen::Matrix4d pose_delta_result = poses_result[first_frame].inverse() * poses_result[last_frame];
//     Eigen::Matrix4d pose_error = pose_delta_result.inverse() * pose_delta_gt;
//     double t_err = translationError(pose_error);
//     double r_err = rotationError(pose_error);
//     double t_err_2d = translationError2D(pose_error);
//     double r_err_2d = rotationError2D(pose_error);

//     // for all segment lengths do
//     for (size_t i = 0; i < num_lengths; i++) {
//       // current length
//       double len = lengths[i];

//       // compute last frame
//       int last_frame = lastFrameFromSegmentLength(dist, first_frame, len);

//       // next frame if sequence not long enough
//       if (last_frame == -1) continue;

//       // compute translational errors
//       Eigen::Matrix4d pose_delta_gt = poses_gt[first_frame].inverse() * poses_gt[last_frame];
//       Eigen::Matrix4d pose_delta_result = poses_result[first_frame].inverse() * poses_result[last_frame];
//       Eigen::Matrix4d pose_error = pose_delta_result.inverse() * pose_delta_gt;
//       double t_err = translationError(pose_error);
//       double r_err = rotationError(pose_error);
//       double t_err_2d = translationError2D(pose_error);
//       double r_err_2d = rotationError2D(pose_error);

//       mean_t_rpe += t_err / len;
//       mean_t_rpe_2d += t_err_2d / len;
//       mean_r_rpe += r_err / len;
//       mean_r_rpe_2d += r_err_2d / len;
//       num_total++;
//     }
//   }

//   seq_err.mean_t_rpe = ((mean_t_rpe / static_cast<double>(num_total)) * 100.0);
//   seq_err.mean_t_rpe_2d = ((mean_t_rpe_2d / static_cast<double>(num_total)) * 100.0);
//   seq_err.mean_r_rpe = ((mean_r_rpe / static_cast<double>(num_total)) * 180.0 / M_PI);
//   seq_err.mean_r_rpe_2d = ((mean_r_rpe_2d / static_cast<double>(num_total)) * 180.0 / M_PI);
// }

}  // namespace

// step_size: every 10 frame (= every second for LiDAR at 10Hz)
// for the Navtech, use 4 (=every second at 4Hz)
Sequence::SeqError evaluateOdometry(const std::string &filename, const ArrayPoses &poses_gt,
                                    const ArrayPoses &poses_est, int step_size) {
  std::ofstream errorfile(filename);
  if (!errorfile.is_open()) throw std::runtime_error{"failed to open file: " + filename};
  errorfile << std::setprecision(std::numeric_limits<long double>::digits10 + 1);

  Sequence::SeqError seq_err;

  // Compute Mean and Max APE (Mean and Max Absolute Pose Error)
  seq_err.mean_ape = 0.0;
  seq_err.max_ape = 0.0;
  for (size_t i = 0; i < poses_gt.size(); i++) {
    double t_ape_err = translationError(poses_est[i].inverse() * poses_gt[0].inverse() * poses_gt[i]);
    seq_err.mean_ape += t_ape_err;
    if (seq_err.max_ape < t_ape_err) {
      seq_err.max_ape = t_ape_err;
    }
  }
  seq_err.mean_ape /= static_cast<double>(poses_gt.size());

  // Compute Mean and Max Local Error
  seq_err.mean_local_err = 0.0;
  seq_err.max_local_err = 0.0;
  seq_err.index_max_local_err = 0;
  for (int i = 1; i < (int)poses_gt.size(); i++) {
    Eigen::Matrix4d t_local = poses_gt[i].inverse() * poses_gt[i - 1] * poses_est[i - 1].inverse() * poses_est[i];
    const auto t_local_vec = lgmath::se3::tran2vec(t_local);

    double gt_local_norm_2d = (poses_gt[i].block<2, 1>(0, 3) - poses_gt[i - 1].block<2, 1>(0, 3)).norm();
    double est_local_norm_2d = (poses_est[i].block<2, 1>(0, 3) - poses_est[i - 1].block<2, 1>(0, 3)).norm();
    double t_local_err_2d = gt_local_norm_2d - est_local_norm_2d;

    double gt_local_norm = (poses_gt[i].block<3, 1>(0, 3) - poses_gt[i - 1].block<3, 1>(0, 3)).norm();
    double est_local_norm = (poses_est[i].block<3, 1>(0, 3) - poses_est[i - 1].block<3, 1>(0, 3)).norm();
    double t_local_err = gt_local_norm - est_local_norm;

    double abs_t_local_err = fabs(t_local_err);
    seq_err.mean_local_err += abs_t_local_err;
    if (seq_err.max_local_err < abs_t_local_err) {
      seq_err.max_local_err = abs_t_local_err;
      seq_err.index_max_local_err = i;
    }

    errorfile << t_local_err_2d << " " << t_local_err << " " << t_local_vec.transpose() << std::endl;
  }
  seq_err.mean_local_err /= static_cast<double>(poses_gt.size() - 1);

  // Compute sequence mean RPE errors
  computeMeanRPE(poses_gt, poses_est, seq_err, step_size);

  return seq_err;
}

}  // namespace steam_icp