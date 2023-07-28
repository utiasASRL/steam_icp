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
 * \file detector.inl
 * \author Keenan Burnett, Autonomous Space Robotics Lab (ASRL)
 * \brief Keypoint extraction methods for Navtech radar
 */
#pragma once

#include "steam_icp/radar/detector.hpp"

namespace steam_icp {

template <class PointT>
std::vector<PointT> ModifiedCACFAR<PointT>::run(const cv::Mat &raw_scan, const float &res,
                                                const std::vector<int64_t> &azimuth_times,
                                                const std::vector<double> &azimuth_angles) {
  const int rows = raw_scan.rows;
  const int cols = raw_scan.cols;
  if (width_ % 2 == 0) width_ += 1;
  const int w2 = std::floor(width_ / 2);
  auto mincol = minr_ / res + w2 + guard_ + 1;
  if (mincol > cols || mincol < 0) mincol = 0;
  auto maxcol = maxr_ / res - w2 - guard_;
  if (maxcol > cols || maxcol < 0) maxcol = cols;
  const int N = maxcol - mincol;

  std::vector<PointT> raw_points;
  raw_points.reserve(2000);

  const double time_delta = azimuth_times.back() - azimuth_times.front();

#pragma omp declare reduction( \
        merge_points : std::vector<PointT> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for num_threads(num_threads_) reduction(merge_points : raw_points)
  for (int i = 0; i < rows; ++i) {
    const double azimuth = azimuth_angles[i];
    const double time = (azimuth_times[i] - initial_timestamp_) * 1.0e-6;
    const double alpha_time = std::min(1.0, std::max(0.0, 1 - (azimuth_times.back() - azimuth_times[i]) / time_delta));
    double mean = 0;
    for (int j = mincol; j < maxcol; ++j) {
      mean += raw_scan.at<float>(i, j);
    }
    mean /= N;

    float peak_points = 0;
    int num_peak_points = 0;

    for (int j = mincol; j < maxcol; ++j) {
      double left = 0;
      double right = 0;
      for (int k = -w2 - guard_; k < -guard_; ++k) left += raw_scan.at<float>(i, j + k);
      for (int k = guard_ + 1; k <= w2 + guard_; ++k) right += raw_scan.at<float>(i, j + k);
      // (statistic) estimate of clutter power
      // const double stat = (left + right) / (2 * w2);
      const double stat = std::max(left, right) / w2;  // GO-CFAR
      const float thres = threshold_ * stat + threshold2_ * mean + threshold3_;
      if (raw_scan.at<float>(i, j) > thres) {
        peak_points += j;
        num_peak_points += 1;
      } else if (num_peak_points > 0) {
        PointT p;
        const double rho = res * peak_points / num_peak_points + range_offset_;
        p.raw_pt[0] = rho * std::cos(-azimuth);
        p.raw_pt[1] = rho * std::sin(-azimuth);
        p.raw_pt[2] = 0.0;
        p.pt = p.raw_pt;
        p.timestamp = time;
        p.alpha_timestamp = alpha_time;
        raw_points.emplace_back(p);
        peak_points = 0;
        num_peak_points = 0;
      }
    }
  }
  raw_points.shrink_to_fit();
  return raw_points;
}

}  // namespace steam_icp