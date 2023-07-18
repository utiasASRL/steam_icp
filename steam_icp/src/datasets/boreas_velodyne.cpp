#include "steam_icp/datasets/boreas_velodyne.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>

namespace steam_icp {

namespace {

std::vector<Point3D> readPointCloud(const std::string &path, const double &time_delta_sec, const double &min_dist,
                                    const double &max_dist) {
  std::vector<Point3D> frame;
  // read bin file
  std::ifstream ifs(path, std::ios::binary);
  std::vector<char> buffer(std::istreambuf_iterator<char>(ifs), {});
  const unsigned float_offset = 4;
  const unsigned fields = 6;  // x, y, z, i, r, t
  const unsigned point_step = float_offset * fields;
  const unsigned numPointsIn = std::floor(buffer.size() / point_step);

  auto getFloatFromByteArray = [](char *byteArray, unsigned index) -> float { return *((float *)(byteArray + index)); };

  double frame_last_timestamp = std::numeric_limits<double>::min();
  double frame_first_timestamp = std::numeric_limits<double>::max();
  frame.reserve(numPointsIn);
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

    new_point.alpha_timestamp = getFloatFromByteArray(buffer.data(), bufpos + offset * float_offset);

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
}

std::vector<Point3D> BoreasVelodyneSequence::next() {
  if (!hasNext()) throw std::runtime_error("No more frames in sequence");
  int curr_frame = curr_frame_++;
  auto filename = filenames_.at(curr_frame);
  const std::string time_str = filename.substr(0, filename.find("."));
  // filenames are epoch times --> at least 9 digits to encode the seconds
  if (time_str.size() < 10) throw std::runtime_error("filename does not have enough digits to encode epoch time");
  filename_to_time_convert_factor_ = 1.0 / pow(10, time_str.size() - 10);
  int64_t time_delta = std::stoll(time_str) - initial_timestamp_;
  double time_delta_sec = static_cast<double>(time_delta) * filename_to_time_convert_factor_;
  std::cout << "time_delta_sec: " << std::setprecision(10) << time_delta_sec << std::endl;
  return readPointCloud(dir_path_ + "/" + filename, time_delta_sec, options_.min_dist_sensor_center,
                        options_.max_dist_sensor_center);
}

void BoreasVelodyneSequence::save(const std::string &path, const Trajectory &trajectory) const {
  //
  ArrayPoses poses;
  poses.reserve(trajectory.size());
  for (auto &frame : trajectory) {
    poses.emplace_back(frame.getMidPose());  // **TODO: do traj interpolation for the midposes
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