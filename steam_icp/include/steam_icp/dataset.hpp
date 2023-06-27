#pragma once

#include <memory>
#include <string>
#include <vector>

#include "steam_icp/point.hpp"
#include "steam_icp/trajectory.hpp"

namespace steam_icp {

class Sequence {
 public:
  using Ptr = std::shared_ptr<Sequence>;
  using ConstPtr = std::shared_ptr<const Sequence>;

  struct Options {
    std::string root_path;
    std::string sequence;
    int init_frame = 0;
    int last_frame = std::numeric_limits<int>::max();  // exclusive bound
    double min_dist_sensor_center = 3.0;               // Threshold to filter points too close to the sensor center
    double max_dist_sensor_center = 100.0;             // Threshold to filter points too far to the sensor center
    // Navtech extraction parameters
    double radar_resolution = 0.0596;
    double radar_range_offset = -0.31;
    int modified_cacfar_width = 101;
    int modified_cacfar_guard = 5;
    double modified_cacfar_threshold = 1.0;
    double modified_cacfar_threshold2 = 0.0;
    double modified_cacfar_threshold3 = 0.09;
  };

  Sequence(const Options &options) : options_(options) {}
  virtual ~Sequence() = default;

  std::string name() const { return options_.sequence; }
  virtual int currFrame() const = 0;
  virtual int numFrames() const = 0;
  virtual void setInitFrame(int /* frame_index */) {
    throw std::runtime_error("set random initial frame not supported");
  };
  virtual bool hasNext() const = 0;
  virtual std::vector<Point3D> next() = 0;
  virtual bool withRandomAccess() const { return false; }
  virtual std::vector<Point3D> frame(size_t /* index */) const {
    throw std::runtime_error("random access not supported");
  }

  virtual void save(const std::string &path, const Trajectory &trajectory) const = 0;

  struct SeqError {
    struct Error {
      double t_err;
      double r_err;
      Error(double t_err, double r_err) : t_err(t_err), r_err(r_err) {}
    };
    std::vector<Error> tab_errors;
    double mean_t_rpe_2d;
    double mean_r_rpe_2d;
    double mean_t_rpe;
    double mean_r_rpe;
    double mean_ape;
    double max_ape;
    double mean_local_err;
    double max_local_err;
    double average_elapsed_ms = -1.0;
    int index_max_local_err;
    double mean_num_attempts;
  };
  virtual bool hasGroundTruth() const { return false; }
  virtual SeqError evaluate(const std::string & /* path */, const Trajectory & /* trajectory */) const {
    throw std::runtime_error("no ground truth available");
  }

 protected:
  const Options options_;
};

class Dataset {
 public:
  using Ptr = std::shared_ptr<Dataset>;
  using ConstPtr = std::shared_ptr<const Dataset>;

  struct Options : public Sequence::Options {
    bool all_sequences = false;
  };

  static Dataset::Ptr Get(const std::string &dataset, const Options &options) {
    return name2Ctor().at(dataset)(options);
  }

  Dataset(const Options &options) : options_(options) {}
  virtual ~Dataset() = default;

  virtual bool hasNext() const = 0;
  virtual Sequence::Ptr next() = 0;

 protected:
  const Options options_;

 private:
  using CtorFunc = std::function<Ptr(const Options &)>;
  using Name2Ctor = std::unordered_map<std::string, CtorFunc>;
  static Name2Ctor &name2Ctor() {
    static Name2Ctor name2ctor;
    return name2ctor;
  }

  template <typename T>
  friend class DatasetRegister;
};

template <typename T>
struct DatasetRegister {
  DatasetRegister() {
    bool success = Dataset::name2Ctor()
                       .try_emplace(T::dataset_name_, Dataset::CtorFunc([](const Dataset::Options &options) {
                                      return std::make_shared<T>(options);
                                    }))
                       .second;
    if (!success) throw std::runtime_error{"DatasetRegister failed - duplicated name"};
  }
};

#define STEAM_ICP_REGISTER_DATASET(NAME, TYPE)       \
 public:                                             \
  inline static constexpr auto dataset_name_ = NAME; \
                                                     \
 private:                                            \
  inline static steam_icp::DatasetRegister<TYPE> dataset_reg_;

}  // namespace steam_icp

///
#include "steam_icp/datasets/aeva.hpp"
#include "steam_icp/datasets/boreas_aeva.hpp"
#include "steam_icp/datasets/boreas_navtech.hpp"
#include "steam_icp/datasets/boreas_velodyne.hpp"
#include "steam_icp/datasets/kitti_360.hpp"
#include "steam_icp/datasets/kitti_raw.hpp"
