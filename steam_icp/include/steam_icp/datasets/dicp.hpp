#pragma once

#include "steam_icp/dataset.hpp"

namespace steam_icp {

class DICPSequence : public Sequence {
 public:
  DICPSequence(const Options& options);

  int currFrame() const override { return curr_frame_; }
  int numFrames() const override { return last_frame_ - init_frame_; }
  bool hasNext() const override { return curr_frame_ < last_frame_; }
  std::vector<Point3D> next() override;

  void save(const std::string& path, const Trajectory& trajectory) const override;

  bool hasGroundTruth() const override { return true; }
  SeqError evaluate(const std::string &path, const Trajectory& trajectory) const override;

 private:
  std::string dir_path_;
  std::vector<std::string> filenames_;
  std::vector<double> timestamps_;
  int init_frame_ = 0;
  int curr_frame_ = 0;
  int last_frame_ = std::numeric_limits<int>::max();  // exclusive bound
};

class DICPDataset : public Dataset {
 public:
  DICPDataset(const Options& options) : Dataset(options) {
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
    return std::make_shared<DICPSequence>(options);
  }

 private:
  std::vector<std::string> sequences_;
  size_t next_sequence_ = 0;

 private:
  static inline std::vector<std::string> SEQUENCES{
      "bunker-road",              //
      "bunker-road-vehicles",     //
      "robin-williams-tunnel",    //
      "brisbane-lagoon-freeway",  //
      // "san-francisco-city",       //
  };

  STEAM_ICP_REGISTER_DATASET("DICP", DICPDataset);
};

}  // namespace steam_icp
