
#ifndef DECISIONTREE_MATHSUTILITY_H
#define DECISIONTREE_MATHSUTILITY_H

#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <random>
#include <cstring>
#include "../Global/GlobalConsts.h"
#include "../Dataset/Dataset.h"

using std::vector;
using std::accumulate;
using std::fill;
using std::uniform_int_distribution;
using std::mt19937;

class Maths {

 public:
  Maths(uint32_t cost_function,
        uint32_t random_state):
    cost_function(cost_function), multiplier(0), random_generator(random_state) {};

  void BindToDataset(const Dataset &dataset) {
    multiplier = dataset.multiplier;
    if (cost_function == EntropyCost)
      ConstructNLogNTable(dataset);
  }

  void UpdateDataset(const Dataset &dataset) {
    if (multiplier != dataset.multiplier) {
      BindToDataset(dataset);
    } else if (cost_function == EntropyCost) {
      ExtendNLogNTable(dataset);
    }
  }

  void UnbindFromDataset() {
    multiplier = 0;
    nlogn_table.clear();
    nlogn_table.shrink_to_fit();
  }

  double NLogN(double x) const {
    auto idx = static_cast<uint32_t>(0.5 + x * multiplier);
    return nlogn_table[idx];
  }

  double NLogN(uint32_t x) const {
    return nlogn_table[x];
  }

  double DeltaNLogN(double x2,
                    double x1) const {
    return NLogN(x2) - NLogN(x1);
  }

  double DeltaNLogN(uint32_t x2,
                    uint32_t x1) const {
    return NLogN(x2) - NLogN(x1);
  }

  void SequentialFillNonzero(uint32_t num,
                             const vector<uint32_t> &weights,
                             vector<uint32_t> &target) const {
    target.reserve(num);
    for (uint32_t idx = 0; idx != num; ++idx)
      if (weights[idx]) target.push_back(idx);
    target.shrink_to_fit();
  }

  uint32_t GetNumSamples(const vector<uint32_t> &sample_ids,
                         const vector<uint32_t> &sample_weights) const {
    uint32_t ret = 0;
    for (const auto &sample_id: sample_ids)
      ret += sample_weights[sample_id];
    return ret;
  }

  double GetCost(const vector<double> &weighted_histogram,
                 double weighted_num_samples) const {
    return GetCostGini(weighted_histogram, weighted_num_samples);
  }

  double GetCost(const vector<uint32_t> &integral_histogram,
                 uint32_t integral_num_samples) const {
    return GetCostEntropy(integral_histogram, integral_num_samples);
  }

  void GetHistogramAndSum(const vector<uint32_t> &sample_ids,
                          const vector<uint32_t> &sample_weights,
                          const vector<uint32_t> &labels,
                          const vector<double> &class_weights,
                          vector<double> &histogram,
                          double &weighted_num_samples) const {
    fill(histogram.begin(), histogram.end(), 0.0);
    for (const auto &sample_id: sample_ids) {
      uint32_t label = labels[sample_id];
      double weighted_height = sample_weights[sample_id] * class_weights[label];
      histogram[label] += weighted_height;
    }
    weighted_num_samples = accumulate(histogram.begin(), histogram.end(), 0.0);
  }

  void GetIntegralHistogramAndSum(const vector<uint32_t> &sample_ids,
                                  const vector<uint32_t> &sample_weights,
                                  const vector<uint32_t> &labels,
                                  const vector<uint32_t> &integral_class_weights,
                                  vector<uint32_t> &integral_histogram,
                                  uint32_t &integral_num_samples) const {
    fill(integral_histogram.begin(), integral_histogram.end(), 0);
    for (const auto &sample_id: sample_ids) {
      uint32_t label = labels[sample_id];
      uint32_t integral_weighted_height = sample_weights[sample_id] * integral_class_weights[label];
      integral_histogram[label] += integral_weighted_height;
    }
    integral_num_samples = accumulate(integral_histogram.begin(), integral_histogram.end(), 0u);
  }

  void GetHistogramAndSum(const vector<uint32_t> &integral_histogram,
                          const uint32_t integral_num_samples,
                          vector<double> &histogram,
                          double &weighted_num_samples) const {
    fill(histogram.begin(), histogram.end(), 0);
    for (uint32_t idx = 0; idx != integral_histogram.size(); ++idx)
      histogram[idx] = static_cast<double>(integral_histogram[idx]) / multiplier;
    weighted_num_samples = static_cast<double>(integral_num_samples) / multiplier;
  }

  void GetIntegralHistogramAndSum(const vector<double> &histogram,
                                  const double weighted_num_samples,
                                  vector<uint32_t> &integral_histogram,
                                  uint32_t &integral_num_samples) const {
    fill(integral_histogram.begin(), integral_histogram.end(), 0);
    for (uint32_t idx = 0; idx != histogram.size(); ++idx)
      integral_histogram[idx] = static_cast<uint32_t>(0.5 + histogram[idx] * multiplier);
    integral_num_samples = static_cast<uint32_t>(0.5 + weighted_num_samples * multiplier);
  }

  void Normalize(double sum,
                 vector<float> &histogram) const {
    for (auto &height: histogram)
      height /= sum;
  }

  float Max(const vector<float> &histogram,
            uint32_t num_classes) const {
    float max = 0.0;
    for (const auto &x: histogram)
      if (x > max) max = x;
    return max;
  }

  uint32_t Argmax(const vector<float> &histogram,
                  uint32_t num_classes) const {
    uint32_t ret = 0;
    float max = 0.0;
    for (uint32_t idx = 0; idx != num_classes; ++idx)
      if (histogram[idx] > max) {
        max = histogram[idx];
        ret = idx;
      }
    return ret;
  }

  template<typename data_t>
  void VectorAddInPlace(uint32_t num_classes,
                        const vector<data_t> &source,
                        uint32_t offset,
                        vector<data_t> &target) const {
    for (uint32_t idx = 0; idx != num_classes; ++idx)
      target[idx] += source[offset + idx];
  }

  template<typename data_t>
  void VectorMinusInPlace(uint32_t num_classes,
                          const vector<data_t> &source,
                          uint32_t offset,
                          vector<data_t> &target) const {
    for (uint32_t idx = 0; idx != num_classes; ++idx)
      target[idx] -= source[offset + idx];
  }

  template<typename data_t>
  void VectorAddOutOfPlace(uint32_t num_classes,
                           const vector<data_t> &lhs,
                           const vector<data_t> &rhs,
                           uint32_t offset,
                           vector<data_t> &target) const {
    for (uint32_t idx = 0; idx != num_classes; ++idx)
      target[idx] = lhs[idx] + rhs[offset + idx];
  }

  template<typename data_t>
  void VectorMinusOutOfPlace(uint32_t num_classes,
                             const vector<data_t> &lhs,
                             const vector<data_t> &rhs,
                             uint32_t offset,
                             vector<data_t> &target) const {
    for (uint32_t idx = 0; idx != num_classes; ++idx)
      target[idx] = lhs[idx] - rhs[offset + idx];
  }

  template<typename data_t>
  void VectorMove(uint32_t num_classes,
                  const vector<data_t> &amount,
                  const uint32_t offset,
                  vector<data_t> &move_from,
                  vector<data_t> &move_to) const {
    for (uint32_t idx = 0; idx != num_classes; ++idx) {
      move_from[idx] -= amount[offset + idx];
      move_to[idx] += amount[offset + idx];
    }
  }

  void SampleWithoutReplacement(uint32_t n,
                                uint32_t k,
                                vector<uint32_t> &target) {
    if (n == k) return;
    uniform_int_distribution<uint32_t> distribution(0, UINT32_MAX);
    for (uint32_t idx = 0; idx != k; ++idx) {
      uint32_t next_random = (distribution(random_generator) % n) + idx;
      uint32_t temp = target[idx];
      target[idx] = target[next_random];
      target[next_random] = temp;
      --n;
    }
  }

  void SampleWithReplacement(uint32_t n,
                             uint32_t k,
                             vector<uint32_t> &histogram) {
    uniform_int_distribution<uint32_t> distribution(0, n - 1);
    for (uint32_t i = 0; i != k; ++i) {
      uint32_t next_random = distribution(random_generator);
      ++histogram[next_random];
    }
  }

  static uint32_t NextToFlip(uint32_t n) {
    return static_cast<uint32_t>(ffs(n) - 1);
  }

 private:
  mt19937 random_generator;
  const uint32_t cost_function;
  double multiplier;
  vector<double> nlogn_table;

  void ConstructNLogNTable(const Dataset &dataset) {
    uint32_t upper_bound = 0;

    for (uint32_t sample_id = 0; sample_id < dataset.num_samples; ++sample_id) {
      const uint32_t label = (*dataset.labels)[sample_id];
      double height = (*dataset.sample_weights)[sample_id] * (*dataset.class_weights)[label];
      auto integral_height = static_cast<uint32_t>(0.5 + height * multiplier);
      upper_bound += integral_height;
    }
    ++upper_bound;
    nlogn_table.reserve(upper_bound);
    nlogn_table.push_back(0.0);
    for (uint32_t idx = 1; idx != upper_bound; ++idx) {
      const double x = static_cast<double>(idx) / multiplier;
      nlogn_table.push_back(x * log2(x));
    }
  }

  void ExtendNLogNTable(const Dataset &dataset) {
    uint32_t upper_bound = 0;

    for (uint32_t sample_id = 0; sample_id < dataset.num_samples; ++sample_id) {
      const uint32_t label = (*dataset.labels)[sample_id];
      double height = (*dataset.sample_weights)[sample_id] * (*dataset.class_weights)[label];
      auto integral_height = static_cast<uint32_t>(0.5 + height * multiplier);
      upper_bound += integral_height;
    }
    ++upper_bound;
    auto old_size = static_cast<uint32_t>(nlogn_table.size());
    if (upper_bound <= old_size) return;
    nlogn_table.reserve(upper_bound);
    for (uint32_t idx = old_size; idx != upper_bound; ++idx) {
      const double x = static_cast<double>(idx) / multiplier;
      nlogn_table.push_back(x * log2(x));
    }
  }

  double GetCostGini(const vector<double> &weighted_histogram,
                     double weighted_num_samples) const {
    double ret = 0.0;
    for (const auto &height: weighted_histogram)
      ret += height * (weighted_num_samples - height);
    ret /= weighted_num_samples;
    return ret;
  }

  double GetCostEntropy(const vector<uint32_t> &integral_histogram,
                        uint32_t integral_num_samples) const {
    double ret = NLogN(integral_num_samples);
    for (const auto &height: integral_histogram)
      ret -= NLogN(height);
    return ret;
  }
};

#endif
