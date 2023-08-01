#pragma once

#include "steam_icp/dataset.hpp"

namespace steam_icp {

class AevaSequence : public Sequence {
 public:
  AevaSequence(const Options& options);

  int currFrame() const override { return curr_frame_; }
  int numFrames() const override { return last_frame_ - init_frame_; }
  bool hasNext() const override { return curr_frame_ < last_frame_; }
  std::tuple<double, std::vector<Point3D>, std::vector<IMUData>> next() override;

  void save(const std::string& path, const Trajectory& trajectory) const override;

  bool hasGroundTruth() const override { return true; }
  SeqError evaluate(const std::string& path, const Trajectory& trajectory) const override;

 private:
  std::string dir_path_;
  std::vector<std::string> filenames_;
  std::vector<double> timestamps_;
  int init_frame_ = 0;
  int curr_frame_ = 0;
  int last_frame_ = std::numeric_limits<int>::max();  // exclusive bound
};

class AevaDataset : public Dataset {
 public:
  AevaDataset(const Options& options) : Dataset(options) {
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
    return std::make_shared<AevaSequence>(options);
  }

 private:
  std::vector<std::string> sequences_;
  size_t next_sequence_ = 0;

 private:
  static inline std::vector<std::string> SEQUENCES{
      "00",  // Baker-Barry Tunnel (Empty)
      "01",  // Baker-Barry Tunnel (Vehicles)
      "02",  // Robin Williams Tunnel
      "03",  // Brisbane Lagoon Freeway
      "04",  // Highway 7
      "05",  // Highway 404
      "06",  // Don Valley Parkway
      "07",  // Highway 427
  };

  STEAM_ICP_REGISTER_DATASET("Aeva", AevaDataset);
};

}  // namespace steam_icp
