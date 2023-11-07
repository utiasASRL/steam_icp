#include "steam_icp/datasets/boreas_navtech.hpp"
#include "steam_icp/radar/utils.hpp"

#include <glog/logging.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include "steam_icp/datasets/utils.hpp"
#include "steam_icp/radar/detector.hpp"
#include "steam_icp/utils/stopwatch.hpp"

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

/// boreas navtech radar upgrade time
static constexpr int64_t upgrade_time = 1632182400000000000;

}  // namespace

BoreasNavtechSequence::BoreasNavtechSequence(const Options &options) : Sequence(options) {
  dir_path_ = options_.root_path + "/" + options_.sequence + "/radar/";
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

  const std::string imu_path = options_.root_path + "/" + options_.sequence + "/applanix/imu_raw.csv";
  std::ifstream imu_file(imu_path);
  Eigen::Matrix3d imu_body_raw_to_applanix, yfwd2xfwd;
  imu_body_raw_to_applanix << 0, -1, 0, -1, 0, 0, 0, 0, -1;
  yfwd2xfwd << 0, 1, 0, -1, 0, 0, 0, 0, 1;
  const std::string time_str = filenames_[0].substr(0, filenames_[0].find("."));
  if (time_str.size() < 10) throw std::runtime_error("filename does not have enough digits to encode epoch time");
  filename_to_time_convert_factor_ = 1.0 / pow(10, time_str.size() - 10);
  const double initial_timestamp_sec = initial_timestamp_ * filename_to_time_convert_factor_;

  if (imu_file.is_open()) {
    std::cout << "imu_file open" << std::endl;
    std::string line;
    std::getline(imu_file, line);  // header
    for (; std::getline(imu_file, line);) {
      if (line.empty()) continue;
      std::stringstream ss(line);
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
      imu_data.ang_vel = yfwd2xfwd * imu_body_raw_to_applanix * imu_data.ang_vel;
      std::getline(ss, value, ',');
      imu_data.lin_acc[2] = std::stod(value);
      std::getline(ss, value, ',');
      imu_data.lin_acc[1] = std::stod(value);
      std::getline(ss, value, ',');
      imu_data.lin_acc[0] = std::stod(value);
      imu_data.lin_acc = yfwd2xfwd * imu_body_raw_to_applanix * imu_data.lin_acc;
      imu_data_vec_.push_back(imu_data);
    }
  }
  LOG(INFO) << "Loaded IMU Data: " << imu_data_vec_.size() << std::endl;
}

DataFrame BoreasNavtechSequence::next() {
  if (!hasNext()) throw std::runtime_error("No more frames in sequence");
  int curr_frame = curr_frame_++;
  auto filename = filenames_.at(curr_frame);
  int64_t current_timestamp_micro = std::stoll(filename.substr(0, filename.find(".")));
  const double radar_resolution = current_timestamp_micro > upgrade_time ? 0.04381 : 0.0596;
  DataFrame frame;
  const double time_delta_sec = static_cast<double>(current_timestamp_micro - initial_timestamp_) * 1.0e-6;
  frame.timestamp = time_delta_sec;
  frame.pointcloud = readPointCloud(dir_path_ + "/" + filename, radar_resolution);
  // get IMU data for this pointcloud:
  double tmin = std::numeric_limits<double>::max();
  double tmax = std::numeric_limits<double>::min();
  for (auto &p : frame.pointcloud) {
    if (p.timestamp < tmin) tmin = p.timestamp;
    if (p.timestamp > tmax) tmax = p.timestamp;
  }

  frame.imu_data_vec.reserve(50);
  for (; curr_imu_idx_ < imu_data_vec_.size(); curr_imu_idx_++) {
    if (imu_data_vec_[curr_imu_idx_].timestamp < tmin) {
      continue;
    } else if (imu_data_vec_[curr_imu_idx_].timestamp >= tmin && imu_data_vec_[curr_imu_idx_].timestamp < tmax) {
      frame.imu_data_vec.emplace_back(imu_data_vec_[curr_imu_idx_]);
    } else {
      break;
    }
  }
  frame.imu_data_vec.shrink_to_fit();
  std::cout << "tmin " << tmin << " tmax " << tmax << " imu_t(0) " << frame.imu_data_vec.front().timestamp
            << " imu_t(-1) " << frame.imu_data_vec.back().timestamp << std::endl;
  LOG(INFO) << "IMU data : " << frame.imu_data_vec.size() << std::endl;

  return frame;
}

std::vector<Point3D> BoreasNavtechSequence::readPointCloud(const std::string &path, const double &radar_resolution) {
  std::vector<int64_t> azimuth_times;
  std::vector<double> azimuth_angles;
  cv::Mat fft_data;
  load_radar(path, azimuth_times, azimuth_angles, fft_data);

  ModifiedCACFAR<Point3D> detector(options_.modified_cacfar_width, options_.modified_cacfar_guard,
                                   options_.modified_cacfar_threshold, options_.modified_cacfar_threshold2,
                                   options_.modified_cacfar_threshold3, options_.modified_cacfar_num_threads,
                                   options_.min_dist_sensor_center, options_.max_dist_sensor_center,
                                   options_.radar_range_offset, initial_timestamp_);

  std::unique_ptr<Stopwatch<>> timer = std::make_unique<Stopwatch<>>(false);
  timer->start();
  const auto pc = detector.run(fft_data, radar_resolution, azimuth_times, azimuth_angles);
  timer->stop();
  LOG(INFO) << "Detector ..................... " << timer->count() << " ms" << std::endl;
  return pc;
}

void BoreasNavtechSequence::save(const std::string &path, const Trajectory &trajectory) const {
  //
  ArrayPoses poses;
  poses.reserve(trajectory.size());
  for (auto &frame : trajectory) {
    poses.emplace_back(frame.getMidPose());
  }

  //
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

auto BoreasNavtechSequence::evaluate(const std::string &path, const Trajectory &trajectory) const -> SeqError {
  //
  std::string ground_truth_file = options_.root_path + "/" + options_.sequence + "/applanix/radar_poses.csv";
  const auto gt_poses_full = loadPoses(ground_truth_file);
  const ArrayPoses gt_poses(gt_poses_full.begin() + init_frame_, gt_poses_full.begin() + last_frame_);

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
  const int step_size = 4;
  return evaluateOdometry(filename, gt_poses, poses, step_size);
}

}  // namespace steam_icp