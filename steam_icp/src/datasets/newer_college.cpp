#include "steam_icp/datasets/newer_college.hpp"

#include <glog/logging.h>

#define PCL_NO_PRECOMPILE
#include <pcl/io/pcd_io.h>
#include <pcl/pcl_macros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include "steam_icp/datasets/utils.hpp"
namespace fs = std::filesystem;

namespace steam_icp {

struct EIGEN_ALIGN16 NCPoint  // enforce SSE padding for correct memory alignment
{
  // PCL_ADD_POINT3D;  // preferred way of adding a XYZ+padding

  // float test;
  float x, y, z, intensity;
  std::uint32_t t;
  std::uint16_t reflectivity;
  uint8_t ring;
  std::uint16_t noise;
  std::uint32_t range;

  PCL_MAKE_ALIGNED_OPERATOR_NEW  // make sure our new allocators are aligned
};
}  // namespace steam_icp

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(steam_icp::NCPoint,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, intensity, intensity)
                                  (std::uint32_t, t, t)
                                  (std::uint16_t, reflectivity, reflectivity)
                                  (std::uint8_t, ring, ring)
                                  (std::uint16_t, noise, noise)
                                  (std::uint32_t, range, range)
)
// clang-format on

namespace steam_icp {

ArrayPoses loadPredPoses(const std::string &file_path) {
  ArrayPoses poses;
  std::ifstream pFile(file_path);
  std::string line;
  if (pFile.is_open()) {
    while (!pFile.eof()) {
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

ArrayPoses loadGTPoses(const std::string &file_path) {
  ArrayPoses poses;
  std::ifstream pose_file(file_path);
  if (pose_file.is_open()) {
    std::string line;
    std::getline(pose_file, line);  // header
    for (; std::getline(pose_file, line);) {
      if (line.empty()) continue;
      std::stringstream ss(line);

      // int sec = 0;
      // int nsec = 0;
      Eigen::Matrix4d T_ms = Eigen::Matrix4d::Identity();
      double qx = 0, qy = 0, qz = 0, qw = 0;

      for (int i = 0; i < 9; ++i) {
        std::string value;
        std::getline(ss, value, ',');

        // if (i == 0)
        //   sec = std::stol(value);
        // else if (i == 1)
        //   nsec = std::stol(value);
        if (i == 2)
          T_ms(0, 3) = std::stod(value);
        else if (i == 3)
          T_ms(1, 3) = std::stod(value);
        else if (i == 4)
          T_ms(2, 3) = std::stod(value);
        else if (i == 5)
          qx = std::stod(value);
        else if (i == 6)
          qy = std::stod(value);
        else if (i == 7)
          qz = std::stod(value);
        else if (i == 8)
          qw = std::stod(value);
      }
      // T_ms.block<3, 3>(0, 0) = rpy2rot(r, p, y);
      Eigen::Quaterniond q(qw, qx, qy, qz);
      T_ms.block<3, 3>(0, 0) = q.toRotationMatrix();

      // (void)timestamp;

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
  pcl::PointCloud<NCPoint>::Ptr cloud(new pcl::PointCloud<NCPoint>);
  if (pcl::io::loadPCDFile<NCPoint>(path, *cloud) == -1)  //* load the file
  {
    PCL_ERROR("Couldn't read file test_pcd.pcd \n");
  }

  std::vector<Point3D> frame;

  const unsigned numPointsIn = cloud->points.size();
  const double timestamp_round_dt = 1.0 / timestamp_round_hz;

  double frame_last_timestamp = std::numeric_limits<double>::min();
  double frame_first_timestamp = std::numeric_limits<double>::max();
  frame.reserve(numPointsIn);

  const double min_dist2 = min_dist * min_dist;
  const double max_dist2 = max_dist * max_dist;
  for (unsigned i(0); i < numPointsIn; i++) {
    Point3D new_point;

    new_point.raw_pt[0] = cloud->points[i].x;
    new_point.raw_pt[1] = cloud->points[i].y;
    new_point.raw_pt[2] = cloud->points[i].z;

    const double r2 = new_point.raw_pt[0] * new_point.raw_pt[0] + new_point.raw_pt[1] * new_point.raw_pt[1] +
                      new_point.raw_pt[2] * new_point.raw_pt[2];
    if ((r2 <= min_dist2) || (r2 >= max_dist2)) continue;

    new_point.pt = new_point.raw_pt;
    // intensity and ring number skipped

    new_point.alpha_timestamp = cloud->points[i].t * 1.0e-9;

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

  // std::set<double> unique_timeset;
  // for (const auto &point : frame) {
  //   unique_timeset.insert(point.alpha_timestamp);
  // }
  // std::vector<double> unique_timevec(unique_timeset.begin(), unique_timeset.end());
  // double ts_hz = unique_timeset.size() / (unique_timevec.back() - unique_timevec[0]);
  // std::cout << "ts_hz: " << ts_hz << std::endl;
  // int downsample = std::floor(ts_hz / timestamp_round_hz);
  // std::cout << "downsample: " << downsample << std::endl;
  // std::vector<double> unique_timevec2;
  // for (int i = 0; i < unique_timevec.size(); i += downsample) unique_timevec2.push_back(unique_timevec[i]);

  // for (int i(0); i < (int)frame.size(); i++) {
  //   const auto p = std::equal_range(unique_timevec2.begin(), unique_timevec2.end(), frame[i].alpha_timestamp);
  //   if (p.first == unique_timevec2.end()) {
  //     frame[i].alpha_timestamp = *p.first;
  //   } else {
  //     frame[i].alpha_timestamp =
  //         fabs(frame[i].alpha_timestamp - *p.first) < fabs(frame[i].alpha_timestamp - *p.second) ? *p.first :
  //         *p.second;
  //   }
  //   if (frame[i].alpha_timestamp < frame_first_timestamp) {
  //     frame_first_timestamp = frame[i].alpha_timestamp;
  //   }
  //   if (frame[i].alpha_timestamp > frame_last_timestamp) {
  //     frame_last_timestamp = frame[i].alpha_timestamp;
  //   }
  // }
  for (int i(0); i < (int)frame.size(); i++) {
    frame[i].timestamp = frame[i].alpha_timestamp + time_delta_sec;
    frame[i].alpha_timestamp = std::min(1.0, std::max(0.0, 1 - (frame_last_timestamp - frame[i].alpha_timestamp) /
                                                                   (frame_last_timestamp - frame_first_timestamp)));
  }

  return frame;
}

NewerCollegeSequence::NewerCollegeSequence(const Options &options) : Sequence(options) {
  Eigen::Matrix4d T_base1_imu = Eigen::Matrix4d::Identity();
  // const double x = std::cos(M_PI_4);
  // yaw by pi / 4
  T_base1_imu.block<3, 3>(0, 0) << std::cos(M_PI_4), std::sin(M_PI_4), 0, -std::sin(M_PI_4), std::cos(M_PI_4), 0.0, 0.0, 0.0, 1.0;
  T_base1_imu.block<3, 1>(0, 3) << -0.08815464364571213, -0.03774772105123108, 0.02165299999999999;
  
  std::cout << "T_base1_imu: " << T_base1_imu << std::endl;
  // Eigen::Matrix4d T2 = Eigen::Matrix4d::Identity();
  // T2.block<3, 3>(0, 0) = T_base1_imu.block<3, 3>(0, 0).transpose();
  // T2.block<3, 1>(0, 3) = -T_base1_imu.block<3, 3>(0, 0).transpose() * T_base1_imu.block<3, 1>(0, 3);
  // T_base1_imu = T2;
  std::cout << "T_base1_imu (inverse): " << T_base1_imu << std::endl;
  Eigen::Matrix4d T_base2_base1 = Eigen::Matrix4d::Identity();
  // xforwards to zforwards (optimal frame convention)
  T_base2_base1.block<3, 3>(0, 0) << 0, -1, 0, 0, 0, -1, 1, 0, 0;
  // T_base2_imu_ = T_base2_base1 * T_base1_imu;
  T_base2_imu_ = T_base1_imu;
  T_imu_lidar_ = Eigen::Matrix4d::Identity();
  T_imu_lidar_.block<3, 1>(0, 3) << -0.006252999999999995, 0.011774999999999994, 0.02853499999999999;
  T_imu_lidar_.block<3, 3>(0, 0) << -1, 0, 0, 0, -1, 0, 0, 0, 1;  // yaw rotation by pi
  T_lidar_base2_ = (T_base2_imu_ * T_imu_lidar_).inverse();
  std::cout << "T_lidar_base2_: " << T_lidar_base2_ << std::endl;

  dir_path_ = options_.root_path + "/" + options_.sequence + "/raw_format/ouster_zip_files/ouster_scan/";
  auto dir_iter = std::filesystem::directory_iterator(dir_path_);
  last_frame_ = std::count_if(begin(dir_iter), end(dir_iter), [this](auto &entry) {
    if (entry.is_regular_file()) filenames_.emplace_back(entry.path().filename().string());
    return entry.is_regular_file();
  });
  last_frame_ = std::min(last_frame_, options_.last_frame);
  curr_frame_ = std::max((int)0, options_.init_frame);
  init_frame_ = std::max((int)0, options_.init_frame);
  std::sort(filenames_.begin(), filenames_.end());
  std::vector<std::string> elems;
  std::stringstream ss(filenames_[0]);
  std::string item;
  while (std::getline(ss, item, '_')) {
    elems.push_back(item);
  }
  std::string sec = elems[1];
  std::string nsec = elems[2].substr(0, elems[2].find("."));
  filename_to_time_convert_factor_ = 1.0e-9;

  for (const std::string filename : filenames_) {
    std::vector<std::string> elems;
    std::stringstream ss(filename);
    std::string item;
    while (std::getline(ss, item, '_')) {
      elems.push_back(item);
    }
    std::string sec = elems[1];
    std::string nsec = elems[2].substr(0, elems[2].find("."));
    uint64_t ts = std::stoll(sec) * uint64_t(1000000000) + std::stoll(nsec);
    timestamps_.push_back(ts);
  }

  initial_timestamp_ = timestamps_[0];
  std::cout << "initial timestamp: " << initial_timestamp_ << std::endl;

  fs::path root_path{options_.root_path};
  std::string ground_truth_file = options_.root_path + "/" + options_.sequence + "/ground_truth/registered_poses.csv";
  const auto gt_poses_full = loadGTPoses(ground_truth_file);
  const ArrayPoses gt_poses(gt_poses_full.begin() + init_frame_, gt_poses_full.begin() + last_frame_);
  T_i_r_gt_poses = gt_poses;

  // std::string imu_path = options_.root_path + "/" + options_.sequence + "/raw_format/realsense_imu/data.csv";
  std::string imu_path = options_.root_path + "/" + options_.sequence + "/raw_format/ouster_imu/data.csv";
  std::ifstream imu_file(imu_path);

  // Eigen::Matrix3d C_imu = Eigen::Matrix3d::Zero();
  // const double x = std::cos(M_PI / 4.0);
  // C_imu << x, 0, x, -x, 0, x, 0, -1, 0;
  // std::cout << "C_imu: " << C_imu << std::endl;

  // const double initial_timestamp_sec = initial_timestamp_ * filename_to_time_convert_factor_;
  if (imu_file.is_open()) {
    std::string line;
    std::getline(imu_file, line);  // header
    for (; std::getline(imu_file, line);) {
      if (line.empty()) continue;

      std::stringstream ss(line);

      steam::IMUData imu_data;
      std::string value;
      std::getline(ss, value, ',');  // counter
      std::string sec, nsec;
      std::getline(ss, sec, ',');
      std::getline(ss, nsec, ',');
      const uint64_t sec_u  = std::stoull(sec);
      const uint64_t nsec_u = std::stoull(nsec);
      if (nsec_u >= 1'000'000'000ULL) {
          throw std::runtime_error("nsec out of range");
      }
      const uint64_t tns = sec_u * 1'000'000'000ULL + nsec_u;
      const int64_t dt_ns = static_cast<int64_t>(tns) - static_cast<int64_t>(initial_timestamp_);
      imu_data.timestamp = static_cast<double>(dt_ns) * 1e-9;
      std::getline(ss, value, ',');
      imu_data.ang_vel[0] = std::stod(value);
      std::getline(ss, value, ',');
      imu_data.ang_vel[1] = std::stod(value);
      std::getline(ss, value, ',');
      imu_data.ang_vel[2] = std::stod(value);

      std::getline(ss, value, ',');
      imu_data.lin_acc[0] = std::stod(value);
      std::getline(ss, value, ',');
      imu_data.lin_acc[1] = std::stod(value);
      std::getline(ss, value, ',');
      imu_data.lin_acc[2] = std::stod(value);

      // imu_data.ang_vel = C_imu * imu_data.ang_vel;
      // imu_data.lin_acc = C_imu * imu_data.lin_acc;

      imu_data_vec_.emplace_back(imu_data);
    }
  }
  LOG(INFO) << "Loaded IMU Data: " << imu_data_vec_.size() << std::endl;
}

DataFrame NewerCollegeSequence::next() {
  if (!hasNext()) throw std::runtime_error("No more frames in sequence");
  int curr_frame = curr_frame_++;
  auto filename = filenames_.at(curr_frame);

  std::vector<std::string> elems;
  std::stringstream ss(filename);
  std::string item;
  while (std::getline(ss, item, '_')) {
    elems.push_back(item);
  }
  // std::string sec = elems[1];
  // std::string nsec = elems[2].substr(0, elems[2].find("."));
  // filename_to_time_convert_factor_ = 1.0e-9;

  DataFrame frame;
  // int64_t time_delta = std::stoll(sec) * uint64_t(1e9) + std::stoll(nsec) - initial_timestamp_;
  uint64_t time_delta = timestamps_[curr_frame] - initial_timestamp_;
  double time_delta_sec = double(time_delta) * filename_to_time_convert_factor_;
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
  LOG(INFO) << "ts: " << frame.timestamp << " tmin: " << tmin << " tmax: " << tmax << std::endl;
  // frame.timestamp = (tmax + tmin) / 2.0;
  
  frame.imu_data_vec.reserve(10);
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
  LOG(INFO) << "IMU data : " << frame.imu_data_vec.size() << std::endl;
  return frame;
}

void NewerCollegeSequence::save(const std::string &path, const Trajectory &trajectory) const {
  //
  ArrayPoses poses;
  poses.reserve(trajectory.size());

  for (auto &frame : trajectory) {
    poses.emplace_back(T_base2_imu_ * frame.getMidPose() * T_lidar_base2_);
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

  //
  {
    const auto filename = path + "/" + options_.sequence + "_tum.txt";
    std::ofstream posefile(filename);
    if (!posefile.is_open()) throw std::runtime_error{"failed to open file: " + filename};
    posefile << std::setprecision(std::numeric_limits<long double>::digits10 + 1);
    // timestamp x y z q_x q_y q_z q_w
    // for (auto &frame : trajectory) {
    for (size_t i = 0; i < trajectory.size(); ++i) {
      const auto &frame = trajectory[i];
      const uint64_t ts = timestamps_[i];
      const uint64_t sec = ts / uint64_t(1000000000);
      const std::string sec_str = std::to_string(sec);
      const uint64_t nsec = ts % uint64_t(1000000000);
      const std::string nsec_str = std::to_string(nsec);
      int n_zero = 9;
      const auto nsec_str2 = std::string(n_zero - std::min(n_zero, int(nsec_str.length())), '0') + nsec_str;
      const Eigen::Matrix4d T = T_base2_imu_ * frame.getMidPose() * T_lidar_base2_;
      const Eigen::Quaterniond q(Eigen::Matrix3d(T.block<3, 3>(0, 0)));
      posefile << sec_str << "." << nsec_str2 << " " << T(0, 3) << " " << T(1, 3) << " " << T(2, 3) << " " << q.x()
               << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    }
  }

  {
    const auto filename = path + "/" + options_.sequence + "_debug.txt";
    std::ofstream posefile(filename);
    if (!posefile.is_open()) throw std::runtime_error{"failed to open file: " + filename};
    posefile << std::setprecision(std::numeric_limits<long double>::digits10 + 1);
    for (auto &frame : trajectory) {
      const auto T = T_base2_imu_ * frame.getMidPose() * T_lidar_base2_;
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

auto NewerCollegeSequence::evaluate(const std::string &path, const Trajectory &trajectory) const -> SeqError {
  //
  std::string ground_truth_file = options_.root_path + "/" + options_.sequence + "/ground_truth/registered_poses.csv";
  const auto gt_poses_full = loadGTPoses(ground_truth_file);
  int last_frame = std::min(last_frame_, int(init_frame_ + trajectory.size()));
  const ArrayPoses gt_poses(gt_poses_full.begin() + init_frame_, gt_poses_full.begin() + last_frame);

  //
  ArrayPoses poses;
  poses.reserve(trajectory.size());
  for (auto &frame : trajectory) {
    poses.emplace_back(T_base2_imu_ * frame.getMidPose() * T_lidar_base2_);
  }

  //
  if (gt_poses.size() == 0 || gt_poses.size() != poses.size())
    throw std::runtime_error{"estimated and ground truth poses are not the same size."};

  const auto filename = path + "/" + options_.sequence + "_eval.txt";
  return evaluateOdometry(filename, gt_poses, poses);
}

auto NewerCollegeSequence::evaluate(const std::string &path) const -> SeqError {
  //
  std::string ground_truth_file = options_.root_path + "/" + options_.sequence + "/ground_truth/registered_poses.csv";
  const auto gt_poses = loadGTPoses(ground_truth_file);
  //
  const auto poses = loadPredPoses(path + "/" + options_.sequence + "_poses.txt");
  //
  if (gt_poses.size() == 0 || gt_poses.size() != poses.size())
    throw std::runtime_error{"estimated and ground truth poses are not the same size."};

  const auto filename = path + "/" + options_.sequence + "_eval.txt";
  return evaluateOdometry(filename, gt_poses, poses);
}

}  // namespace steam_icp