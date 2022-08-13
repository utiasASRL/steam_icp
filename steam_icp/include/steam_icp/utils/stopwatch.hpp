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
 * \file stopwatch.hpp
 * \author Yuchen Wu, Autonomous Space Robotics Lab (ASRL)
 */
#pragma once

#include <chrono>

namespace steam_icp {

template <class clock = std::chrono::steady_clock>
class Stopwatch {
 public:
  Stopwatch(const bool start = true) {
    if (start) this->start();
  }

  void start() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!started_) {
      started_ = true;
      paused_ = false;
      reference_ = clock::now();
      accumulated_ = typename clock::duration(0);
    } else if (paused_) {
      paused_ = false;
      reference_ = clock::now();
    }
  }

  void stop() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (started_ && !paused_) {
      accumulated_ += std::chrono::duration_cast<typename clock::duration>(clock::now() - reference_);
      paused_ = true;
    }
  }

  void reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    started_ = false;
    paused_ = false;
    reference_ = clock::now();
    accumulated_ = typename clock::duration(0);
  }

  template <class duration_t = std::chrono::milliseconds>
  typename duration_t::rep count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (started_) {
      if (paused_) {
        return std::chrono::duration_cast<duration_t>(accumulated_).count();
      } else {
        return std::chrono::duration_cast<duration_t>(accumulated_ + (clock::now() - reference_)).count();
      }
    } else {
      return duration_t(0).count();
    }
  }

  friend std::ostream &operator<<(std::ostream &os, const Stopwatch &sw) { return os << sw.count() << "ms"; }

 private:
  mutable std::mutex mutex_;
  bool started_ = false;
  bool paused_ = false;
  typename clock::time_point reference_ = clock::now();
  typename clock::duration accumulated_ = typename clock::duration(0);
};

}  // namespace steam_icp
