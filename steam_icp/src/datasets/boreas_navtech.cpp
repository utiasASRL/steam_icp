#include "steam_icp/datasets/boreas_navtech.hpp"
#include "steam_icp/radar/detector.hpp"
#include "steam_icp/radar/utils.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>

namespace steam_icp {

/// boreas navtech radar upgrade time
static constexpr int64_t upgrade_time = 1632182400000000000;

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
  initial_timestamp_micro_ = std::stoll(filenames_[0].substr(0, filenames_[0].find(".")));
}

std::vector<Point3D> BoreasNavtechSequence::next() {
  if (!hasNext()) throw std::runtime_error("No more frames in sequence");
  int curr_frame = curr_frame_++;
  auto filename = filenames_.at(curr_frame);
  int64_t current_timestamp_micro = std::stoll(filename.substr(0, filename.find(".")));
  const double radar_resolution = current_timestamp_micro > upgrade_time ? 0.04381 : 0.0596;
  return readPointCloud(dir_path_ + "/" + filename, current_timestamp_micro, radar_resolution);
}

std::vector<Point3D> BoreasNavtechSequence::readPointCloud(const std::string &path,
                                                           const int64_t &current_timestamp_micro,
                                                           const double &radar_resolution) {
  std::vector<int64_t> azimuth_times;
  std::vector<double> azimuth_angles;
  cv::Mat fft_data;
  load_radar(path, azimuth_times, azimuth_angles, fft_data);
  ModifiedCACFAR detector = ModifiedCACFAR<Point3D>(
      options_.modified_cacfar_width, options_.modified_cacfar_guard, options_.modified_cacfar_threshold,
      options_.modified_cacfar_threshold2, options_.modified_cacfar_threshold3, options_.min_dist_sensor_center,
      options_.max_dist_sensor_center, options_.radar_range_offset, initial_timestamp_micro_, current_timestamp_micro);
  return detector.run(fft_data, radar_resolution, azimuth_times, azimuth_angles);
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

}  // namespace steam_icp