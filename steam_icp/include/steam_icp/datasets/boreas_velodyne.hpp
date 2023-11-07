#pragma once

#include "steam_icp/dataset.hpp"

namespace steam_icp {

class BoreasVelodyneSequence : public Sequence {
 public:
  BoreasVelodyneSequence(const Options& options);

  int currFrame() const override { return curr_frame_; }
  int numFrames() const override { return last_frame_ - init_frame_; }
  bool hasNext() const override { return curr_frame_ < last_frame_; }
  DataFrame next() override;

  void save(const std::string& path, const Trajectory& trajectory) const override;

  bool hasGroundTruth() const override { return true; }
  SeqError evaluate(const std::string& path, const Trajectory& trajectory) const override;

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
};

class BoreasVelodyneDataset : public Dataset {
 public:
  BoreasVelodyneDataset(const Options& options) : Dataset(options) {
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
    return std::make_shared<BoreasVelodyneSequence>(options);
  }

 private:
  std::vector<std::string> sequences_;
  size_t next_sequence_ = 0;

 private:
  static inline std::vector<std::string> SEQUENCES{
      "boreas-2022-05-13-09-23",  // highway 7
      "boreas-2022-05-13-10-30",  // marc santi
      "boreas-2022-05-13-11-47",  // glen shields
      "boreas-2022-05-18-17-23",  // cocksfield
  };

  STEAM_ICP_REGISTER_DATASET("BoreasVelodyne", BoreasVelodyneDataset);
};

}  // namespace steam_icp
