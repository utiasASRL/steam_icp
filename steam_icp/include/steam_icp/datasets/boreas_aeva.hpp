#pragma once

#include "steam_icp/dataset.hpp"

namespace steam_icp {

class BoreasAevaSequence : public Sequence {
 public:
  BoreasAevaSequence(const Options& options);

  int currFrame() const override { return curr_frame_; }
  int numFrames() const override { return last_frame_ - init_frame_; }
  bool hasNext() const override { return curr_frame_ < last_frame_; }
  std::pair<double, std::vector<Point3D>> next() override;

  void save(const std::string& path, const Trajectory& trajectory) const override;

  bool hasGroundTruth() const override { return true; }
  SeqError evaluate(const std::string& path, const Trajectory& trajectory) const override;

 private:
  std::string dir_path_;
  std::vector<std::string> filenames_;
  int64_t initial_timestamp_;
  int init_frame_ = 0;
  int curr_frame_ = 0;
  int last_frame_ = std::numeric_limits<int>::max();  // exclusive bound
  double filename_to_time_convert_factor_ = 1.0e-6;   // may change depending on length of timestamp (ns vs. us)

  // velocity calibration parameters
  Eigen::MatrixXd rt_parts_;
  std::vector<Eigen::MatrixXd> azi_ranges_;
  std::vector<Eigen::MatrixXd> vel_means_;
  bool has_beam_id_ = false;
};

class BoreasAevaDataset : public Dataset {
 public:
  BoreasAevaDataset(const Options& options) : Dataset(options) {
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
    return std::make_shared<BoreasAevaSequence>(options);
  }

 private:
  std::vector<std::string> sequences_;
  size_t next_sequence_ = 0;

 private:
  static inline std::vector<std::string> SEQUENCES{
      "04",  // Highway 7
      "05",  // Highway 404
      "06",  // Don Valley Parkway
      "07",  // Highway 427
  };

  STEAM_ICP_REGISTER_DATASET("BoreasAeva", BoreasAevaDataset);
};

}  // namespace steam_icp
