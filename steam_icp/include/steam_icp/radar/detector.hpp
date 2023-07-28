// Copyright 2021, Autonomous Space Robotics Lab (ASRL)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * \file detector.hpp
 * \author Keenan Burnett, Autonomous Space Robotics Lab (ASRL)
 * \brief Keypoint extraction methods for Navtech radar
 */

#pragma once

#include "opencv2/opencv.hpp"

namespace steam_icp {

template <class PointT>
class Detector {
 public:
  virtual ~Detector() = default;

  virtual std::vector<PointT> run(const cv::Mat &raw_scan, const float &res, const std::vector<int64_t> &azimuth_times,
                                  const std::vector<double> &azimuth_angles) = 0;
};

template <class PointT>
class ModifiedCACFAR : public Detector<PointT> {
 public:
  ModifiedCACFAR() = default;
  ModifiedCACFAR(int width, int guard, double threshold, double threshold2, double threshold3, int num_threads,
                 double minr, double maxr, double range_offset, int64_t initial_timestamp_micro)
      : width_(width),
        guard_(guard),
        threshold_(threshold),
        threshold2_(threshold2),
        threshold3_(threshold3),
        num_threads_(num_threads),
        minr_(minr),
        maxr_(maxr),
        range_offset_(range_offset),
        initial_timestamp_(initial_timestamp_micro) {}

  std::vector<PointT> run(const cv::Mat &raw_scan, const float &res, const std::vector<int64_t> &azimuth_times,
                          const std::vector<double> &azimuth_angles) override;

 private:
  int width_ = 41;  // window = width + 2 * guard
  int guard_ = 2;
  double threshold_ = 3.0;
  double threshold2_ = 1.1;
  double threshold3_ = 0.22;
  int num_threads_ = 1;
  double minr_ = 2.0;
  double maxr_ = 100.0;
  double range_offset_ = -0.31;
  int64_t initial_timestamp_ = 0;
};

}  // namespace steam_icp

#include "steam_icp/radar/detector.inl"