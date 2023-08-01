#include "steam_icp/datasets/boreas_aeva.hpp"

#include <glog/logging.h>
#include <algorithm>
#include <filesystem>
#include <fstream>

#include "steam_icp/datasets/utils.hpp"

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

Eigen::MatrixXd readCSVtoEigenXd(std::ifstream &csv) {
  std::string line;
  std::string cell;
  std::vector<std::vector<double>> mat_vec;
  while (std::getline(csv, line)) {
    std::stringstream lineStream(line);
    std::vector<double> row_vec;
    while (std::getline(lineStream, cell, ',')) {
      row_vec.push_back(std::stof(cell));
    }
    mat_vec.push_back(row_vec);
  }
  Eigen::MatrixXd output = Eigen::MatrixXd(mat_vec.size(), mat_vec[0].size());
  for (int i = 0; i < (int)mat_vec.size(); ++i) output.row(i) = Eigen::VectorXd::Map(&mat_vec[i][0], mat_vec[i].size());
  return output;
}

void getCalibData(const std::string &path_to_vcalib, Eigen::MatrixXd &rt_parts,
                  std::vector<Eigen::MatrixXd> &azi_ranges, std::vector<Eigen::MatrixXd> &vel_means) {
  // relative time partitions
  std::ifstream rt_csv(path_to_vcalib + "/rt_part.csv");
  if (!rt_csv) throw std::ios::failure("Error opening file rt_part.csv");
  rt_parts = readCSVtoEigenXd(rt_csv);

  // azimuth ranges (there are 4 for each beam)
  azi_ranges.clear();
  for (int b = 0; b < 4; ++b) {
    std::ifstream azi_csv(path_to_vcalib + "/azi_minmax_" + std::to_string(b) + ".csv");
    if (!azi_csv) throw std::ios::failure("Error opening file azi_minmax_" + std::to_string(b) + ".csv");
    azi_ranges.push_back(readCSVtoEigenXd(azi_csv));
  }

  // velocity mean values (there are 4 for each beam)
  vel_means.clear();
  for (int b = 0; b < 4; ++b) {
    std::ifstream vel_csv(path_to_vcalib + "/vel_mean_" + std::to_string(b) + ".csv");
    if (!vel_csv) throw std::ios::failure("Error opening file vel_mean_" + std::to_string(b) + ".csv");
    vel_means.push_back(readCSVtoEigenXd(vel_csv));
  }
}

void calibrate(const Eigen::MatrixXd &rt_parts, const std::vector<Eigen::MatrixXd> &azi_ranges,
               const std::vector<Eigen::MatrixXd> &vel_means, std::vector<Point3D> &point_cloud) {
  // iterate through each point
  for (auto &point : point_cloud) {
    const int b = point.beam_id;                                      // beam id
    const double rt = point.alpha_timestamp;                          // relative time
    const double azi = std::atan2(point.raw_pt(1), point.raw_pt(0));  // azimuth

    // determine beam partition
    int p = 0;
    if (rt >= rt_parts(b, rt_parts.cols() - 1))
      p = rt_parts.cols() - 1;
    else {
      while (rt > rt_parts(b, p + 1)) ++p;
      assert(rt >= rt_parts(b, p) && rt < rt_parts(b, p + 1));
    }

    // determine azimuth bin
    double azi_res = (azi_ranges[b](p, 1) - azi_ranges[b](p, 0)) / vel_means[b].cols();
    int bin_id = floor((azi - azi_ranges[b](p, 0)) / azi_res);

    // compensate
    point.radial_velocity -= vel_means[b](p, std::clamp(bin_id, 0, int(vel_means[b].cols() - 1)));
  }
}

std::vector<Point3D> readPointCloud(const std::string &path, const double &time_delta_sec, const double &min_dist,
                                    const double &max_dist, const bool has_beam_id) {
  std::vector<Point3D> frame;
  // read bin file
  std::ifstream ifs(path, std::ios::binary);
  std::vector<char> buffer(std::istreambuf_iterator<char>(ifs), {});
  unsigned float_offset = 4;
  unsigned fields = has_beam_id ? 7 : 6;  // x, y, z, i, r, t, b
  unsigned point_step = float_offset * fields;
  unsigned numPointsIn = std::floor(buffer.size() / point_step);

  auto getFloatFromByteArray = [](char *byteArray, unsigned index) -> float { return *((float *)(byteArray + index)); };

  double frame_last_timestamp = -1000000.0;
  double frame_first_timestamp = 1000000.0;
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
    // intensity skipped
    ++offset;
    new_point.radial_velocity = getFloatFromByteArray(buffer.data(), bufpos + offset * float_offset);
    ++offset;
    new_point.alpha_timestamp = getFloatFromByteArray(buffer.data(), bufpos + offset * float_offset);
    if (has_beam_id) {
      ++offset;
      new_point.beam_id = (int)getFloatFromByteArray(buffer.data(), bufpos + offset * float_offset);
    }

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
    frame[i].timestamp = frame[i].alpha_timestamp + time_delta_sec;
    frame[i].alpha_timestamp = std::min(1.0, std::max(0.0, 1 - (frame_last_timestamp - frame[i].alpha_timestamp) /
                                                                   (frame_last_timestamp - frame_first_timestamp)));
  }

  return frame;
}
}  // namespace

BoreasAevaSequence::BoreasAevaSequence(const Options &options) : Sequence(options) {
  dir_path_ = options_.root_path + "/" + options_.sequence + "/aeva/";
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
    std::string line;
    std::getline(imu_file, line);  // header
    for (; std::getline(imu_file, line);) {
      if (line.empty()) continue;
      std::stringstream ss(line);
      IMUData imu_data;
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

  std::string calib_path = options_.root_path + "/" + options_.sequence + "/aeva_calib/";
  if (std::filesystem::exists(calib_path)) {
    getCalibData(calib_path, rt_parts_, azi_ranges_, vel_means_);
    has_beam_id_ = true;
  } else {
    throw std::runtime_error("BoreasAevaSequence: calibration data not found");
  }
}

std::tuple<double, std::vector<Point3D>, std::vector<IMUData>> BoreasAevaSequence::next() {
  if (!hasNext()) throw std::runtime_error("No more frames in sequence");
  int curr_frame = curr_frame_++;
  auto filename = filenames_.at(curr_frame);
  const std::string time_str = filename.substr(0, filename.find("."));
  // filenames are epoch times --> at least 9 digits to encode the seconds
  if (time_str.size() < 10) throw std::runtime_error("filename does not have enough digits to encode epoch time");
  filename_to_time_convert_factor_ = 1.0 / pow(10, time_str.size() - 10);
  int64_t time_delta = std::stoll(time_str) - initial_timestamp_;
  double time_delta_sec = static_cast<double>(time_delta) * filename_to_time_convert_factor_;

  // load point cloud
  auto points = readPointCloud(dir_path_ + "/" + filename, time_delta_sec, options_.min_dist_sensor_center,
                               options_.max_dist_sensor_center, has_beam_id_);
  if (has_beam_id_) calibrate(rt_parts_, azi_ranges_, vel_means_, points);

  // get IMU data for this pointcloud:
  double tmin = std::numeric_limits<double>::max();
  double tmax = std::numeric_limits<double>::min();
  for (auto &p : points) {
    if (p.timestamp < tmin) tmin = p.timestamp;
    if (p.timestamp > tmax) tmax = p.timestamp;
  }
  std::vector<IMUData> curr_imu_data_vec;
  curr_imu_data_vec.reserve(20);
  for (; curr_imu_idx_ < imu_data_vec_.size(); curr_imu_idx_++) {
    if (imu_data_vec_[curr_imu_idx_].timestamp >= tmin && imu_data_vec_[curr_imu_idx_].timestamp < tmax)
      curr_imu_data_vec.emplace_back(imu_data_vec_[curr_imu_idx_]);
  }
  curr_imu_data_vec.shrink_to_fit();

  return std::make_tuple(time_delta_sec, points, curr_imu_data_vec);
}

void BoreasAevaSequence::save(const std::string &path, const Trajectory &trajectory) const {
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

auto BoreasAevaSequence::evaluate(const std::string &path, const Trajectory &trajectory) const -> SeqError {
  //
  std::string ground_truth_file = options_.root_path + "/" + options_.sequence + "/applanix/aeva_poses.csv";
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