#include "steam_icp/datasets/boreas_velodyne.hpp"

#include <glog/logging.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include "steam_icp/datasets/utils.hpp"
namespace fs = std::filesystem;

namespace steam_icp {

namespace {

inline Eigen::Matrix3d roll(const double &r) {
  Eigen::Matrix3d res;
  res << 1., 0., 0., 0., std::cos(r), std::sin(r), 0., -std::sin(r), std::cos(r);
  return res;
}

inline Eigen::Matrix3d pitch(const double &p) {
  Eigen::Matrix3d res;
  res << std::cos(p), 0., -std::sin(p), 0., 1., 0., std::sin(p), 0., std::cos(p);
  return res;
}

inline Eigen::Matrix3d yaw(const double &y) {
  Eigen::Matrix3d res;
  res << std::cos(y), std::sin(y), 0., -std::sin(y), std::cos(y), 0., 0., 0., 1.;
  return res;
}

inline Eigen::Matrix3d rpy2rot(const double &r, const double &p, const double &y) {
  return roll(r) * pitch(p) * yaw(y);
}

ArrayPoses loadPoses(const std::string &file_path) {
  ArrayPoses poses;
  std::ifstream pose_file(file_path);
  if (pose_file.is_open()) {
    std::string line;
    std::getline(pose_file, line);  // header
    for (; std::getline(pose_file, line);) {
      if (line.empty()) continue;
      std::stringstream ss(line);

      int64_t timestamp = 0;
      Eigen::Matrix4d T_ms = Eigen::Matrix4d::Identity();
      double r = 0, p = 0, y = 0;

      for (int i = 0; i < 10; ++i) {
        std::string value;
        std::getline(ss, value, ',');

        if (i == 0)
          timestamp = std::stol(value);
        else if (i == 1)
          T_ms(0, 3) = std::stod(value);
        else if (i == 2)
          T_ms(1, 3) = std::stod(value);
        else if (i == 3)
          T_ms(2, 3) = std::stod(value);
        else if (i == 7)
          r = std::stod(value);
        else if (i == 8)
          p = std::stod(value);
        else if (i == 9)
          y = std::stod(value);
      }
      T_ms.block<3, 3>(0, 0) = rpy2rot(r, p, y);

      (void)timestamp;
      // LOG(WARNING) << "loaded: " << timestamp << " " << std::fixed << std::setprecision(6)
      //              << T_ms(0, 3) << " " << T_ms(1, 3) << " " << T_ms(2, 3) << " "
      //              << r << " " << p << " " << y << " " << std::endl;

      poses.push_back(T_ms);
    }
  } else {
    throw std::runtime_error{"unable to open file: " + file_path};
  }
  return poses;
}

std::vector<Point3D> readPointCloud(const std::string &path, const std::string &precision_time_path,
                                    const double &time_delta_sec, const double &min_dist, const double &max_dist,
                                    const bool round_timestamps, const double &timestamp_round_hz) {
  std::vector<Point3D> frame;
  // read bin file
  std::ifstream ifs(path, std::ios::binary);
  std::vector<char> buffer(std::istreambuf_iterator<char>(ifs), {});
  const unsigned float_offset = 4;
  const unsigned fields = 6;  // x, y, z, i, r, t
  const unsigned point_step = float_offset * fields;
  const unsigned numPointsIn = std::floor(buffer.size() / point_step);
  const double timestamp_round_dt = 1.0 / timestamp_round_hz;

  auto getFloatFromByteArray = [](char *byteArray, unsigned index) -> float { return *((float *)(byteArray + index)); };
  auto getDoubleFromByteArray = [](char *byteArray, unsigned index) -> double {
    return *((double *)(byteArray + index));
  };

  double frame_last_timestamp = std::numeric_limits<double>::min();
  double frame_first_timestamp = std::numeric_limits<double>::max();
  frame.reserve(numPointsIn);

  bool use_precision_times = false;
  Eigen::VectorXd precision_times;
  if (std::filesystem::directory_entry(precision_time_path).is_regular_file()) {
    std::ifstream ifs2(precision_time_path, std::ios::binary);
    std::vector<char> buffer2(std::istreambuf_iterator<char>(ifs2), {});
    const unsigned double_offset = 8;
    const unsigned numPoints2 = std::floor(buffer2.size() / double_offset);
    if (numPoints2 == numPointsIn) {
      use_precision_times = true;
      precision_times = Eigen::VectorXd::Zero(numPointsIn);
      for (unsigned i(0); i < numPointsIn; i++) {
        const int bufpos = i * double_offset;
        precision_times(i, 0) = getDoubleFromByteArray(buffer2.data(), bufpos);
      }
    } else {
      std::cout << "ERROR loading precision timestamp file..." << std::endl;
    }
  }

  const double min_dist2 = min_dist * min_dist;
  const double max_dist2 = max_dist * max_dist;
  for (unsigned i(0); i < numPointsIn; i++) {
    Point3D new_point;

    const int bufpos = i * point_step;
    int offset = 0;
    new_point.raw_pt[0] = getFloatFromByteArray(buffer.data(), bufpos + offset * float_offset);
    ++offset;
    new_point.raw_pt[1] = getFloatFromByteArray(buffer.data(), bufpos + offset * float_offset);
    ++offset;
    new_point.raw_pt[2] = getFloatFromByteArray(buffer.data(), bufpos + offset * float_offset);

    const double r2 = new_point.raw_pt[0] * new_point.raw_pt[0] + new_point.raw_pt[1] * new_point.raw_pt[1] +
                      new_point.raw_pt[2] * new_point.raw_pt[2];
    if ((r2 <= min_dist2) || (r2 >= max_dist2)) continue;

    new_point.pt = new_point.raw_pt;
    // intensity and ring number skipped
    offset += 3;

    new_point.alpha_timestamp =
        static_cast<double>(getFloatFromByteArray(buffer.data(), bufpos + offset * float_offset));

    if (use_precision_times) {
      new_point.alpha_timestamp = precision_times(i, 0);
    } else {
      new_point.alpha_timestamp = getFloatFromByteArray(buffer.data(), bufpos + offset * float_offset);
    }

    if (round_timestamps)
      new_point.alpha_timestamp = new_point.alpha_timestamp - fmod(new_point.alpha_timestamp, timestamp_round_dt);

    if (new_point.alpha_timestamp < frame_first_timestamp) {
      frame_first_timestamp = new_point.alpha_timestamp;
    }
    if (new_point.alpha_timestamp > frame_last_timestamp) {
      frame_last_timestamp = new_point.alpha_timestamp;
    }
    frame.push_back(new_point);
  }
  frame.shrink_to_fit();

  for (int i(0); i < (int)frame.size(); i++) {
    frame[i].timestamp = frame[i].alpha_timestamp + time_delta_sec;
    frame[i].alpha_timestamp = std::min(1.0, std::max(0.0, 1 - (frame_last_timestamp - frame[i].alpha_timestamp) /
                                                                   (frame_last_timestamp - frame_first_timestamp)));
  }

  return frame;
}
}  // namespace

BoreasVelodyneSequence::BoreasVelodyneSequence(const Options &options) : Sequence(options) {
  dir_path_ = options_.root_path + "/" + options_.sequence + "/lidar/";
  auto dir_iter = std::filesystem::directory_iterator(dir_path_);
  last_frame_ = std::count_if(begin(dir_iter), end(dir_iter), [this](auto &entry) {
    if (entry.is_regular_file()) filenames_.emplace_back(entry.path().filename().string());
    return entry.is_regular_file();
  });
  last_frame_ = std::min(last_frame_, options_.last_frame);
  curr_frame_ = std::max((int)0, options_.init_frame);
  init_frame_ = std::max((int)0, options_.init_frame);
  std::sort(filenames_.begin(), filenames_.end());
  initial_timestamp_ = std::stoll(filenames_[0].substr(0, filenames_[0].find(".")));

  fs::path root_path{options_.root_path};
  std::string ground_truth_file = options_.root_path + "/" + options_.sequence + "/applanix/lidar_poses.csv";
  const auto gt_poses_full = loadPoses(ground_truth_file);
  const ArrayPoses gt_poses(gt_poses_full.begin() + init_frame_, gt_poses_full.begin() + last_frame_);
  std::ifstream ifs(root_path / name() / "calib" / "T_applanix_lidar.txt", std::ios::in);
  Eigen::Matrix4d T_applanix_lidar;
  for (size_t row = 0; row < 4; row++)
    for (size_t col = 0; col < 4; col++) ifs >> T_applanix_lidar(row, col);
  Eigen::Matrix4d T_lidar_applanix = T_applanix_lidar.inverse();
  Eigen::Matrix4d T_robot_applanix;
  T_robot_applanix << 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
  Eigen::Matrix4d T_applanix_robot = T_robot_applanix.inverse();
  T_i_r_gt_poses.clear();
  for (auto pose : gt_poses) {
    T_i_r_gt_poses.push_back(pose * T_lidar_applanix * T_applanix_robot);
  }

  bool use_sbet_imu = false;
  bool use_raw_accel_minus_gravity = false;

  std::string imu_path = options_.root_path + "/" + options_.sequence + "/applanix/imu_raw.csv";
  if (use_sbet_imu) imu_path = options_.root_path + "/" + options_.sequence + "/applanix/imu.csv";
  std::string pose_meas_path = options_.root_path + "/" + options_.sequence + "/applanix/lidar_pose_meas.csv";
  std::string accel_path = options_.root_path + "/" + options_.sequence + "/applanix/accel_raw_minus_gravity.csv";
  std::ifstream imu_file(imu_path);
  std::ifstream pose_meas_file(pose_meas_path);
  std::ifstream acc_file(accel_path);
  Eigen::Matrix3d imu_body_raw_to_applanix, yfwd2xfwd;
  imu_body_raw_to_applanix << 0, -1, 0, -1, 0, 0, 0, 0, -1;
  yfwd2xfwd << 0, 1, 0, -1, 0, 0, 0, 0, 1;
  const std::string time_str = filenames_[0].substr(0, filenames_[0].find("."));
  if (time_str.size() < 10) throw std::runtime_error("filename does not have enough digits to encode epoch time");
  filename_to_time_convert_factor_ = 1.0 / pow(10, time_str.size() - 10);
  const double initial_timestamp_sec = initial_timestamp_ * filename_to_time_convert_factor_;
  if (imu_file.is_open()) {
    std::string line;
    std::string accel_line;
    std::getline(imu_file, line);        // header
    std::getline(acc_file, accel_line);  // header
    for (; std::getline(imu_file, line);) {
      if (line.empty()) continue;
      std::getline(acc_file, accel_line);
      if (accel_line.empty()) continue;
      std::stringstream ss(line);
      std::stringstream ss2(accel_line);
      steam::IMUData imu_data;
      std::string value;
      std::getline(ss, value, ',');
      imu_data.timestamp = std::stod(value) - initial_timestamp_sec;
      std::getline(ss, value, ',');
      imu_data.ang_vel[2] = std::stod(value);
      std::getline(ss, value, ',');
      imu_data.ang_vel[1] = std::stod(value);
      std::getline(ss, value, ',');
      imu_data.ang_vel[0] = std::stod(value);
      // IMU data is transformed into the "robot" frame, coincident with applanix
      // with x-forwards, y-left, z-up.

      std::getline(ss, value, ',');
      imu_data.lin_acc[2] = std::stod(value);
      std::getline(ss, value, ',');
      imu_data.lin_acc[1] = std::stod(value);
      std::getline(ss, value, ',');
      imu_data.lin_acc[0] = std::stod(value);
      if (use_sbet_imu) {
        imu_data.ang_vel = yfwd2xfwd * imu_data.ang_vel;
        imu_data.lin_acc = yfwd2xfwd * imu_data.lin_acc;
      } else {
        imu_data.ang_vel = yfwd2xfwd * imu_body_raw_to_applanix * imu_data.ang_vel;
        if (use_raw_accel_minus_gravity) {
          std::getline(ss2, value, ',');
          std::getline(ss2, value, ',');
          imu_data.lin_acc[0] = std::stod(value);
          std::getline(ss2, value, ',');
          imu_data.lin_acc[1] = std::stod(value);
          std::getline(ss2, value, ',');
          imu_data.lin_acc[2] = std::stod(value);
        } else {
          imu_data.lin_acc = yfwd2xfwd * imu_body_raw_to_applanix * imu_data.lin_acc;
        }
      }
      imu_data_vec_.push_back(imu_data);
    }
  }
  if (pose_meas_file.is_open()) {
    std::string line;
    std::getline(pose_meas_file, line);  // header
    for (; std::getline(pose_meas_file, line);) {
      if (line.empty()) continue;
      std::stringstream ss(line);
      std::string value;
      std::getline(ss, value, ',');
      PoseData pose_data;
      pose_data.timestamp = std::stod(value) - initial_timestamp_sec;
      pose_data.pose = Eigen::Matrix4d::Identity();
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
          std::getline(ss, value, ',');
          pose_data.pose(i, j) = std::stod(value);
        }
      }
      pose_data_vec_.push_back(pose_data);
    }
  }
  LOG(INFO) << "Loaded IMU Data: " << imu_data_vec_.size() << std::endl;
  LOG(INFO) << "Loaded Pose Meas Data: " << pose_data_vec_.size() << std::endl;
}

DataFrame BoreasVelodyneSequence::next() {
  if (!hasNext()) throw std::runtime_error("No more frames in sequence");
  int curr_frame = curr_frame_++;
  auto filename = filenames_.at(curr_frame);
  const std::string time_str = filename.substr(0, filename.find("."));
  // filenames are epoch times --> at least 9 digits to encode the seconds
  if (time_str.size() < 10) throw std::runtime_error("filename does not have enough digits to encode epoch time");
  filename_to_time_convert_factor_ = 1.0 / pow(10, time_str.size() - 10);
  DataFrame frame;
  int64_t time_delta = std::stoll(time_str) - initial_timestamp_;
  double time_delta_sec = static_cast<double>(time_delta) * filename_to_time_convert_factor_;
  frame.timestamp = time_delta_sec;
  const auto precision_time_file = options_.root_path + "/" + options_.sequence + "/lidar_times/" + filename;
  frame.pointcloud = readPointCloud(dir_path_ + "/" + filename, precision_time_file, time_delta_sec,
                                    options_.min_dist_sensor_center, options_.max_dist_sensor_center,
                                    options_.lidar_timestamp_round, options_.lidar_timestamp_round_hz);

  // get IMU data for this pointcloud:
  double tmin = std::numeric_limits<double>::max();
  double tmax = std::numeric_limits<double>::min();
  for (auto &p : frame.pointcloud) {
    if (p.timestamp < tmin) tmin = p.timestamp;
    if (p.timestamp > tmax) tmax = p.timestamp;
  }
  frame.imu_data_vec.reserve(21);
  for (; curr_imu_idx_ < imu_data_vec_.size(); curr_imu_idx_++) {
    if (imu_data_vec_[curr_imu_idx_].timestamp < tmin) {
      continue;
    } else if (imu_data_vec_[curr_imu_idx_].timestamp >= tmin && imu_data_vec_[curr_imu_idx_].timestamp < tmax) {
      frame.imu_data_vec.emplace_back(imu_data_vec_[curr_imu_idx_]);
    } else {
      break;
    }
  }
  for (; curr_pose_meas_idx_ < pose_data_vec_.size(); curr_pose_meas_idx_++) {
    const auto pose_time = pose_data_vec_[curr_pose_meas_idx_].timestamp;
    if (pose_time < tmin) {
      continue;
    } else if (pose_time >= tmin && pose_time < tmax) {
      frame.pose_data_vec.emplace_back(pose_data_vec_[curr_pose_meas_idx_]);
    } else {
      break;
    }
  }

  frame.imu_data_vec.shrink_to_fit();
  LOG(INFO) << "IMU data : " << frame.imu_data_vec.size() << std::endl;
  LOG(INFO) << "Pose data : " << frame.pose_data_vec.size() << std::endl;
  return frame;
}

void BoreasVelodyneSequence::save(const std::string &path, const Trajectory &trajectory) const {
  //
  ArrayPoses poses;
  poses.reserve(trajectory.size());
  for (auto &frame : trajectory) {
    poses.emplace_back(frame.getMidPose());  // **TODO: do traj interpolation for the midposes
  }

  //
  {
    const auto filename = path + "/" + options_.sequence + "_poses.txt";
    std::ofstream posefile(filename);
    if (!posefile.is_open()) throw std::runtime_error{"failed to open file: " + filename};
    posefile << std::setprecision(std::numeric_limits<long double>::digits10 + 1);
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    for (auto &pose : poses) {
      R = pose.block<3, 3>(0, 0);
      t = pose.block<3, 1>(0, 3);
      posefile << R(0, 0) << " " << R(0, 1) << " " << R(0, 2) << " " << t(0) << " " << R(1, 0) << " " << R(1, 1) << " "
               << R(1, 2) << " " << t(1) << " " << R(2, 0) << " " << R(2, 1) << " " << R(2, 2) << " " << t(2)
               << std::endl;
    }
  }

  {
    const auto filename = path + "/" + options_.sequence + "_debug.txt";
    std::ofstream posefile(filename);
    if (!posefile.is_open()) throw std::runtime_error{"failed to open file: " + filename};
    posefile << std::setprecision(std::numeric_limits<long double>::digits10 + 1);
    for (auto &frame : trajectory) {
      const auto T = frame.getMidPose();
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          posefile << T(i, j) << " ";
        }
      }
      for (int i = 0; i < 6; ++i) {
        posefile << frame.mid_w(i, 0) << " ";
      }
      for (int i = 0; i < 6; ++i) {
        posefile << frame.mid_dw(i, 0) << " ";
      }
      for (int i = 0; i < 6; ++i) {
        posefile << frame.mid_b(i, 0) << " ";
      }
      // const auto P = frame.getMidPose();
      for (int i = 0; i < 18; ++i) {
        for (int j = 0; j < 18; ++j) {
          posefile << frame.mid_state_cov(i, j) << " ";
        }
      }
      const auto T_mi = frame.mid_T_mi;
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          posefile << T_mi(i, j) << " ";
        }
      }
      posefile << std::endl;
    }
  }
}

auto BoreasVelodyneSequence::evaluate(const std::string &path, const Trajectory &trajectory) const -> SeqError {
  //
  std::string ground_truth_file = options_.root_path + "/" + options_.sequence + "/applanix/lidar_poses.csv";
  const auto gt_poses_full = loadPoses(ground_truth_file);
  int last_frame = std::min(last_frame_, int(init_frame_ + trajectory.size()));
  const ArrayPoses gt_poses(gt_poses_full.begin() + init_frame_, gt_poses_full.begin() + last_frame);

  //
  ArrayPoses poses;
  poses.reserve(trajectory.size());
  for (auto &frame : trajectory) {
    poses.emplace_back(frame.getMidPose());
  }

  //
  if (gt_poses.size() == 0 || gt_poses.size() != poses.size())
    throw std::runtime_error{"estimated and ground truth poses are not the same size."};

  const auto filename = path + "/" + options_.sequence + "_eval.txt";
  return evaluateOdometry(filename, gt_poses, poses);
}

auto BoreasVelodyneSequence::evaluate(const std::string &path) const -> SeqError {
  //
  std::string ground_truth_file = options_.root_path + "/" + options_.sequence + "/applanix/lidar_poses.csv";
  const auto gt_poses = loadPoses(ground_truth_file);
  //
  const auto poses = loadPoses(path + "/" + options_.sequence + "_poses.txt");
  //
  if (gt_poses.size() == 0 || gt_poses.size() != poses.size())
    throw std::runtime_error{"estimated and ground truth poses are not the same size."};

  const auto filename = path + "/" + options_.sequence + "_eval.txt";
  return evaluateOdometry(filename, gt_poses, poses);
}

}  // namespace steam_icp