#pragma once

#include "steam_icp/dataset.hpp"

namespace steam_icp {

class NewerCollegeSequence : public Sequence {
 public:
  NewerCollegeSequence(const Options& options);

  int currFrame() const override { return curr_frame_; }
  int numFrames() const override { return last_frame_ - init_frame_; }
  bool hasNext() const override { return curr_frame_ < last_frame_; }
  DataFrame next() override;

  void save(const std::string& path, const Trajectory& trajectory) const override;

  bool hasGroundTruth() const override { return true; }
  SeqError evaluate(const std::string& path, const Trajectory& trajectory) const override;
  SeqError evaluate(const std::string& path) const override;

 private:
  std::string dir_path_;
  std::vector<std::string> filenames_;
  int64_t initial_timestamp_;
  std::vector<steam::IMUData> imu_data_vec_;
  std::vector<PoseData> pose_data_vec_;
  unsigned curr_imu_idx_ = 0;
  unsigned curr_pose_meas_idx_ = 0;
  int init_frame_ = 0;
  int curr_frame_ = 0;
  int last_frame_ = std::numeric_limits<int>::max();  // exclusive bound
  double filename_to_time_convert_factor_ = 1.0e-6;   // may change depending on length of timestamp (ns vs. us)

  Eigen::Matrix4d T_base2_imu_ = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d T_imu_lidar_ = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d T_lidar_base2_ = Eigen::Matrix4d::Identity();
  std::vector<uint64_t> timestamps_;
};

class NewerCollegeDataset : public Dataset {
 public:
  NewerCollegeDataset(const Options& options) : Dataset(options) {
    if (options_.all_sequences)
      sequences_ = SEQUENCES;
    else
      sequences_.emplace_back(options_.sequence);
  }

  bool hasNext() const override { return next_sequence_ < sequences_.size(); }
  Sequence::Ptr next() override {
    if (!hasNext()) return nullptr;
    Sequence::Options options(options_);
    options.sequence = sequences_[next_sequence_++];
    return std::make_shared<NewerCollegeSequence>(options);
  }

 private:
  std::vector<std::string> sequences_;
  size_t next_sequence_ = 0;

 private:
  static inline std::vector<std::string> SEQUENCES{
      "01_short_experiment", "02_long_experiment", "05_quad_with_dynamics", "06_dynamic_spinning", "07_parkland_mound",
  };

  STEAM_ICP_REGISTER_DATASET("NewerCollege", NewerCollegeDataset);
};

}  // namespace steam_icp
