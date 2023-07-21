#pragma once

#include "steam_icp/dataset.hpp"

namespace steam_icp {

class KittiRawSequence : public Sequence {
 public:
  KittiRawSequence(const Options& options);

  int currFrame() const override { return curr_frame_; }
  int numFrames() const override { return last_frame_ - init_frame_; }
  bool hasNext() const override { return curr_frame_ < last_frame_; }
  std::pair<double, std::vector<Point3D>> next() override;

  void save(const std::string& path, const Trajectory& trajectory) const override;

  bool hasGroundTruth() const override { return has_ground_truth_; }
  SeqError evaluate(const std::string& path, const Trajectory& trajectory) const override;

 private:
  std::string dir_path_;
  int sequence_id_ = -1;
  int init_frame_ = 0;
  int curr_frame_ = 0;
  int last_frame_ = std::numeric_limits<int>::max();  // exclusive bound
  bool has_ground_truth_ = false;
};

class KittiRawDataset : public Dataset {
 public:
  KittiRawDataset(const Options& options) : Dataset(options) {
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
    return std::make_shared<KittiRawSequence>(options);
  }

 private:
  std::vector<std::string> sequences_;
  size_t next_sequence_ = 0;

 private:
  static inline std::vector<std::string> SEQUENCES{"00", "01", "02", "04", "05", "06", "07", "08", "09", "10"};

  STEAM_ICP_REGISTER_DATASET("KITTI_raw", KittiRawDataset);
};

}  // namespace steam_icp
