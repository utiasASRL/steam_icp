#include "steam_icp/datasets/aeva.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>

#include "steam_icp/datasets/utils.hpp"

namespace steam_icp {

namespace {
std::vector<Point3D> readPointCloud(const std::string &path, const double &time_sec, const double &min_dist,
                                    const double &max_dist) {
  std::vector<Point3D> frame;
  // read bin file
  std::ifstream ifs(path, std::ios::binary);
  std::vector<char> buffer(std::istreambuf_iterator<char>(ifs), {});
  unsigned float_offset = 4;
  unsigned fields = 5;  // x, y, z, v, t
  unsigned point_step = float_offset * fields;
  unsigned numPointsIn = std::floor(buffer.size() / point_step);

  auto getFloatFromByteArray = [](char *byteArray, unsigned index) -> float { return *((float *)(byteArray + index)); };

  double frame_last_timestamp = -1000000000.0;
  double frame_first_timestamp = std::numeric_limits<double>::max();
  frame.reserve(numPointsIn);
  for (unsigned i(0); i < numPointsIn; i++) {
    Point3D new_point;

    int bufpos = i * point_step;
    int offset = 0;
    new_point.raw_pt[0] = getFloatFromByteArray(buffer.data(), bufpos + offset * float_offset);
    ++offset;
    new_point.raw_pt[1] = getFloatFromByteArray(buffer.data(), bufpos + offset * float_offset);
    ++offset;
    new_point.raw_pt[2] = getFloatFromByteArray(buffer.data(), bufpos + offset * float_offset);
    new_point.pt = new_point.raw_pt;

    ++offset;
    new_point.radial_velocity = getFloatFromByteArray(buffer.data(), bufpos + offset * float_offset);
    ++offset;
    new_point.alpha_timestamp = getFloatFromByteArray(buffer.data(), bufpos + offset * float_offset);

    if (new_point.alpha_timestamp < frame_first_timestamp) {
      frame_first_timestamp = new_point.alpha_timestamp;
    }

    if (new_point.alpha_timestamp > frame_last_timestamp) {
      frame_last_timestamp = new_point.alpha_timestamp;
    }

    double r = new_point.raw_pt.norm();
    if ((r > min_dist) && (r < max_dist)) {
      frame.push_back(new_point);
    }
  }
  frame.shrink_to_fit();

  for (int i(0); i < (int)frame.size(); i++) {
    frame[i].timestamp = time_sec + frame[i].alpha_timestamp;
    frame[i].alpha_timestamp = std::min(1.0, std::max(0.0, 1 - (frame_last_timestamp - frame[i].alpha_timestamp) /
                                                                   (frame_last_timestamp - frame_first_timestamp)));
  }

  return frame;
}

/* -------------------------------------------------------------------------------------------------------------- */
ArrayPoses loadPoses(const std::string &file_path) {
  ArrayPoses poses;
  std::ifstream pFile(file_path);
  if (pFile.is_open()) {
    while (!pFile.eof()) {
      std::string line;
      std::getline(pFile, line);
      if (line.empty()) continue;
      std::stringstream ss(line);
      Eigen::Matrix4d P = Eigen::Matrix4d::Identity();
      ss >> P(0, 0) >> P(0, 1) >> P(0, 2) >> P(0, 3) >> P(1, 0) >> P(1, 1) >> P(1, 2) >> P(1, 3) >> P(2, 0) >>
          P(2, 1) >> P(2, 2) >> P(2, 3);
      poses.push_back(P);
    }
    pFile.close();
  } else {
    throw std::runtime_error{"unable to open file: " + file_path};
  }
  return poses;
}

}  // namespace

AevaSequence::AevaSequence(const Options &options) : Sequence(options) {
  dir_path_ = options_.root_path + "/" + options_.sequence + "/frames/";
  auto dir_iter = std::filesystem::directory_iterator(dir_path_);
  last_frame_ = std::count_if(begin(dir_iter), end(dir_iter), [this](auto &entry) {
    if (entry.is_regular_file()) filenames_.emplace_back(entry.path().filename().string());
    return entry.is_regular_file();
  });
  std::sort(filenames_.begin(), filenames_.end());
  //
  std::string timestamp_file = options_.root_path + "/" + options_.sequence + "/timestamps.txt";
  std::ifstream ifs(timestamp_file, std::ios::in);
  for (std::string line; std::getline(ifs, line);) {
    std::stringstream str(line);
    double timestamp;
    str >> timestamp;
    timestamps_.push_back(timestamp);
  }
  if ((int)timestamps_.size() != last_frame_)
    throw std::runtime_error{"timestamp file and point cloud file number mismatch"};

  //
  last_frame_ = std::min(last_frame_, options_.last_frame);
  curr_frame_ = std::max(0, options_.init_frame);
  init_frame_ = std::max(0, options_.init_frame);
}

std::vector<Point3D> AevaSequence::next() {
  if (!hasNext()) throw std::runtime_error("No more frames in sequence");
  int curr_frame = curr_frame_++;
  auto filename = filenames_.at(curr_frame);
  auto timestamp = timestamps_.at(curr_frame);

  return readPointCloud(dir_path_ + "/" + filename, timestamp, options_.min_dist_lidar_center,
                        options_.max_dist_lidar_center);
}

void AevaSequence::save(const std::string &path, const Trajectory &trajectory) const {
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

auto AevaSequence::evaluate(const std::string &path, const Trajectory &trajectory) const -> SeqError {
  //
  std::string ground_truth_file = options_.root_path + "/" + options_.sequence + "/aeva_poses.txt";
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
  return evaluateOdometry(filename, gt_poses, poses);
}

}  // namespace steam_icp