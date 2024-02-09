#pragma once

#include "steam_icp/dataset.hpp"

namespace steam_icp {

class BoreasNavtechSequence : public Sequence {
 public:
  BoreasNavtechSequence(const Options &options);

  int currFrame() const override { return curr_frame_; }
  int numFrames() const override { return last_frame_ - init_frame_; }
  bool hasNext() const override { return curr_frame_ < last_frame_; }
  DataFrame next() override;

  void save(const std::string &path, const Trajectory &trajectory) const override;

  bool hasGroundTruth() const override { return true; }
  SeqError evaluate(const std::string &path, const Trajectory &trajectory) const override;
  SeqError evaluate(const std::string &path) const override;

 private:
  std::string dir_path_;
  std::vector<std::string> filenames_;
  int64_t initial_timestamp_;
  std::vector<steam::IMUData> imu_data_vec_;
  unsigned curr_imu_idx_ = 0;
  int init_frame_ = 0;
  int curr_frame_ = 0;
  int last_frame_ = std::numeric_limits<int>::max();  // exclusive bound
  double filename_to_time_convert_factor_ = 1.0e-6;   // may change depending on length of timestamp (ns vs. us)
  double beta = 0.049;

  std::vector<Point3D> readPointCloud(const std::string &path, const double &radar_resolution);
};

class BoreasNavtechDataset : public Dataset {
 public:
  BoreasNavtechDataset(const Options &options) : Dataset(options) {
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
    return std::make_shared<BoreasNavtechSequence>(options);
  }

 private:
  std::vector<std::string> sequences_;
  size_t next_sequence_ = 0;

 private:
  static inline std::vector<std::string> SEQUENCES{
      // "boreas-2022-05-13-09-23",  // highway 7
      // "boreas-2022-05-13-10-30",  // marc santi
      // "boreas-2022-05-13-11-47",  // glen shields
      // "boreas-2022-05-18-17-23",  // cocksfield
      "boreas-2020-12-04-14-00", "boreas-2021-01-26-10-59", "boreas-2021-02-09-12-55", "boreas-2021-03-09-14-23",
      "boreas-2021-04-22-15-00", "boreas-2021-06-29-18-53", "boreas-2021-06-29-20-43", "boreas-2021-09-08-21-00",
      "boreas-2021-09-09-15-28", "boreas-2021-10-05-15-35", "boreas-2021-10-26-12-35", "boreas-2021-11-06-18-55",
      "boreas-2021-11-28-09-18",
  };

  STEAM_ICP_REGISTER_DATASET("BoreasNavtech", BoreasNavtechDataset);
};

}  // namespace steam_icp
