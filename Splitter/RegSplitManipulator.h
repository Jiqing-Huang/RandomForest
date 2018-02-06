
#ifndef DECISIONTREE_REGRESSIONSPLITMANIPULATOR_H
#define DECISIONTREE_REGRESSIONSPLITMANIPULATOR_H

#include <cstdint>
#include <vector>
#include "../Dataset/Dataset.h"
#include "../Tree/TreeParams.h"
#include "../Util/Maths.h"
#include "../Tree/TreeNode.h"

#define IS_INTEGRAL_FEATURE (std::is_integral<feature_t>::value)
#define IS_INTEGRAL_LABEL (std::is_integral<label_t>::value)

using std::vector;

class RegStats {
 public:
  double sum;
  double sum_left;
  double sum_right;
  double square_sum;

  vector<double> binwise_sum;
  vector<double> binwise_num_samples;
  vector<uint32_t> bin_ids;
  vector<double> means;
  double num_samples_left;
  double num_samples_right;
  double num_samples;
  uint32_t num_bins;

  const MetaData &meta;
  const TreeParams &params;

  RegStats(const MetaData &meta,
           const TreeParams &params):
    meta(meta), params(params), sum(0.0), sum_left(0.0), sum_right(0.0), square_sum(0.0),
    binwise_sum(meta.max_num_bins, 0.0), binwise_num_samples(meta.max_num_bins, 0.0), bin_ids(meta.max_num_bins, 0),
    means(meta.max_num_bins, 0.0), num_samples_left(0.0), num_samples_right(0.0), num_samples(0.0), num_bins(0) {}
};

template <typename CostComputer>
class RegSplitManipulator {
 public:
  RegSplitManipulator(const Dataset *dataset,
                      const TreeParams &params):
    stats(dataset->Meta(), params), cost_computer() {};

  void NumericalInit(const TreeNode *node) {
    stats.square_sum = node->Stats()->SquareSum();
    stats.sum = node->Stats()->Sum();
    stats.sum_left = stats.sum;
    stats.sum_right = 0.0;
    stats.num_samples = node->Stats()->NumSamples();
    stats.num_samples_left = stats.num_samples;
    stats.num_samples_right = 0;
  }

  template <typename label_t>
  typename std::enable_if<!IS_INTEGRAL_LABEL, void>::type
  MoveOneSample(const vector<label_t> &labels,
                const vec_uint32_t &sample_weights,
                uint32_t idx,
                double &cost) {
    label_t label = labels[idx];
    uint32_t sample_weight = sample_weights[idx];
    stats.num_samples_left -= sample_weight;
    stats.num_samples_right += sample_weight;
    double weighted_label = label * sample_weight;
    stats.sum_left -= weighted_label;
    stats.sum_right += weighted_label;
    cost = cost_computer.ComputeCost(stats.square_sum, stats.sum_left, stats.sum_right,
                                     stats.num_samples_left, stats.num_samples_right);
  }

  template <typename feature_t, typename label_t>
  typename std::enable_if<!IS_INTEGRAL_LABEL && IS_INTEGRAL_FEATURE, void>::type
  DiscreteInit(const vector<feature_t> &features,
               const vector<label_t> &labels,
               const vec_uint32_t &sample_weights,
               uint32_t feature_idx,
               TreeNode *node) {
    for (uint32_t idx = 0; idx != node->Size(); ++idx) {
      feature_t bin = features[idx];
      label_t label = labels[idx];
      uint32_t sample_weight = sample_weights[idx];
      double weighted_label = label * sample_weight;
      stats.binwise_sum[bin] += weighted_label;
      stats.binwise_num_samples[bin] += sample_weight;
    }

    for (uint32_t idx = 0; idx != stats.meta.num_bins[feature_idx]; ++idx)
      if (stats.binwise_num_samples[idx])
        stats.bin_ids[stats.num_bins++] = idx;

    stats.square_sum = node->Stats()->SquareSum();
    stats.sum = node->Stats()->Sum();
    stats.sum_left = stats.sum;
    stats.sum_right = 0.0;

    stats.num_samples = node->Stats()->NumSamples();
    stats.num_samples_left = stats.num_samples;
    stats.num_samples_right = 0;
  }

  void Clear() {
    for (uint32_t idx = 0; idx != stats.num_bins; ++idx) {
      uint32_t bin = stats.bin_ids[idx];
      stats.binwise_sum[bin] = 0.0;
      stats.binwise_num_samples[bin] = 0.0;
    }
    stats.num_bins = 0;
  }

  void MoveOneBinLToR(uint32_t bin,
                      double &cost) {
    stats.num_samples_left -= stats.binwise_num_samples[bin];
    stats.num_samples_right += stats.binwise_num_samples[bin];

    stats.sum_left -= stats.binwise_sum[bin];
    stats.sum_right += stats.binwise_sum[bin];

    cost = cost_computer.ComputeCost(stats.square_sum, stats.sum_left, stats.sum_right,
                                     stats.num_samples_left, stats.num_samples_right);
  }

  void MoveOneBinRToL(uint32_t bin,
                      double &cost) {
    // shouldn't be called
    assert(false);
  }

  void SetOneVsAll(uint32_t bin,
                   double &cost) {
    stats.num_samples_left = stats.binwise_num_samples[bin];
    stats.num_samples_right = stats.num_samples - stats.num_samples_left;
    stats.sum_left = stats.binwise_sum[bin];
    stats.sum_right = stats.sum - stats.sum_left;
    cost = cost_computer.ComputeCost(stats.square_sum, stats.sum_left, stats.sum_right,
                                     stats.num_samples_left, stats.num_samples_right);
  }

  void ReorderBinIds() {
    const uint32_t num_classes = 2;
    vector<double> &means = stats.means;
    for (uint32_t idx = 0; idx != stats.num_bins; ++idx) {
      uint32_t bin = stats.bin_ids[idx];
      means[bin] = stats.binwise_sum[bin] / stats.binwise_num_samples[bin];
    }
    sort(stats.bin_ids.begin(),
         stats.bin_ids.begin() + stats.num_bins,
         [&means](uint32_t x, uint32_t y) {
           return means[x] < means[y];
         });
  }

  void MoveOneBinOutOfPlace(uint32_t bin,
                            double &cost) {
    // shouldn't be called
    assert(false);
  }

  void MoveOneBinInPlace(uint32_t bin) {
    // shouldn't be called
    assert(false);
  }

  bool LessThanMinLeafNode() {
    return stats.num_samples_left < stats.params.min_leaf_node ||
           stats.num_samples_right < stats.params.min_leaf_node;
  }

  template <typename feature_t>
  typename std::enable_if<!IS_INTEGRAL_FEATURE, bool>::type
  Splittable(const vector<feature_t> &features,
             const vec_uint32_t &sample_ids,
             const vec_uint32_t &sorted_idx,
             uint32_t idx) {
    uint32_t first = sample_ids[sorted_idx[idx]];
    uint32_t second = sample_ids[sorted_idx[idx + 1]];
    return features[first] != features[second];
  }

  template <typename feature_t>
  typename std::enable_if<!IS_INTEGRAL_FEATURE, float>::type
  NumericalThreshold(const vector<feature_t> &features,
                     const vec_uint32_t &sample_ids,
                     const vec_uint32_t &sorted_idx,
                     uint32_t idx) {
    uint32_t first = sample_ids[sorted_idx[idx]];
    uint32_t second = sample_ids[sorted_idx[idx + 1]];
    return (features[first] + features[second]) / 2.0f;
  }

  uint32_t NumBins() {
    return stats.num_bins;
  }

  uint32_t MaxNumBins(uint32_t feature_idx) {
    return stats.meta.num_bins[feature_idx];
  }

  uint32_t BinId(uint32_t idx) {
    return stats.bin_ids[idx];
  }

  void ShuffleBinId(uint32_t n,
                    uint32_t k) {
    // shouldn't be called
    assert(false);
  }

  void SwitchWithLast(uint32_t idx,
                      uint32_t num_bins) {
    // shouldn't be called
    assert(false);
  }

 private:
  RegStats stats;
  CostComputer cost_computer;
};

class VarianceCostComputer {
 public:
  double ComputeCost(double square_sum,
                     double sum_left,
                     double sum_right,
                     double num_samples_left,
                     double num_samples_right) {
    return square_sum - sum_left * sum_left / num_samples_left - sum_right * sum_right / num_samples_right;
  }
};

using VarianceSplitManipulator = RegSplitManipulator<VarianceCostComputer>;

#endif
