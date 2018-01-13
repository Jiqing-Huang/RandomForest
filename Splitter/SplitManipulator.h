
#ifndef DECISIONTREE_SPLITMANIPULATOR_H
#define DECISIONTREE_SPLITMANIPULATOR_H

#include <vector>
#include <cstdint>
#include "../Dataset/Dataset.h"
#include "../Tree/TreeParams.h"
#include "../Util/Maths.h"
#include "../Tree/TreeNode.h"

using std::vector;
using std::copy;
using std::fill;
using std::iota;
using std::sort;

template <typename data_t>
class SplitData {
 public:
  vector<data_t> init_left;
  vector<data_t> init_right;
  vector<data_t> cur_left;
  vector<data_t> cur_right;
  vector<data_t> bin_class_matrix;
  vector<uint32_t> binwise_num_samples;
  vector<data_t> binwise_wnum_samples;
  vector<uint32_t> bin_ids;
  vector<double> fractions;
  uint32_t num_samples_left;
  uint32_t num_samples_right;
  uint32_t num_samples;
  data_t wnum_samples_left;
  data_t wnum_samples_right;
  data_t wnum_samples;
  uint32_t num_bins;

  double updater_left;
  double updater_right;

  const Dataset &dataset;
  const TreeParams &params;
  Maths &util;
  const vector<data_t> *class_weights;

  SplitData(const Dataset &dataset,
            const TreeParams &params,
            Maths &util):
          dataset(dataset), params(params), util(util), class_weights(nullptr),
          init_left(dataset.num_classes, 0), init_right(dataset.num_classes, 0),
          cur_left(dataset.num_classes, 0), cur_right(dataset.num_classes, 0),
          bin_class_matrix(dataset.max_num_bins * dataset.num_classes, 0),
          binwise_num_samples(dataset.max_num_bins, 0), binwise_wnum_samples(dataset.max_num_bins, 0),
          bin_ids(dataset.max_num_bins, 0), fractions(dataset.max_num_bins, 0.0),
          num_samples_left(0), num_samples_right(0), num_samples(0),
          wnum_samples_left(0), wnum_samples_right(0), wnum_samples(0), num_bins(0),
          updater_left(0.0), updater_right(0.0) {}
};

template <typename data_t, typename CostComputer>
class SplitManipulator {
 public:
  explicit SplitManipulator(const Dataset &dataset,
                            const TreeParams &params,
                            Maths &util):
          split_data(dataset, params, util), cost_computer() {
    cost_computer.LinkClassWeights(dataset, split_data.class_weights);
  }

  void NumericalInit(TreeNode *node) {
    cost_computer.Init(node, split_data.cur_left, split_data.wnum_samples, split_data.updater_left);
    fill(split_data.cur_right.begin(), split_data.cur_right.end(), static_cast<data_t>(0));
    split_data.num_samples = node->stats->num_samples;
    split_data.num_samples_left = split_data.num_samples;
    split_data.num_samples_right = 0;
    split_data.wnum_samples_left = split_data.wnum_samples;
    split_data.wnum_samples_right = 0;
    split_data.updater_right = 0;
  }

  void GetLoopBound(uint32_t feature_idx,
                    const vector<uint32_t> &sorted_idx,
                    const vector<uint32_t> &sample_weights,
                    uint32_t &loop_begin,
                    uint32_t &loop_end) {
    uint32_t idx = 0;
    uint32_t accumulated_weight = 0;
    uint32_t min_leaf_node = split_data.params.min_leaf_node;
    while (accumulated_weight < min_leaf_node) {
      uint32_t sample_id = sorted_idx[idx];
      uint32_t sample_weight = sample_weights[sample_id];
      accumulated_weight += sample_weight;
      ++idx;
    }
    loop_begin = idx - 1;
    idx = static_cast<uint32_t>(sorted_idx.size()) - 1;
    accumulated_weight = 0;
    while (accumulated_weight < min_leaf_node) {
      uint32_t sample_id = sorted_idx[idx];
      uint32_t sample_weight = sample_weights[sample_id];
      accumulated_weight += sample_weight;
      --idx;
    }
    loop_end = idx + 1;
  }

  double MoveOneSample(const vector<uint32_t> &labels,
                       const vector<uint32_t> &sample_weights,
                       uint32_t sample_id,
                       double &cost) {
    uint32_t label = labels[sample_id];
    data_t weight = sample_weights[sample_id] * (*split_data.class_weights)[label];
    split_data.cur_left[label] -= weight;
    split_data.cur_right[label] += weight;
    split_data.wnum_samples_left -= weight;
    split_data.wnum_samples_right += weight;

    cost_computer.UpdateCost(split_data, weight, split_data.wnum_samples_left, split_data.cur_left[label],
                             split_data.wnum_samples_right, split_data.cur_right[label], cost);
  }

  void DiscreteInit(uint32_t feature_idx,
                    TreeNode *node) {
    const vector<uint32_t> &sample_ids = node->subset->sample_ids;
    const vector<uint32_t> &feature = node->subset->discrete_features[feature_idx];
    const vector<uint32_t> &labels = node->subset->labels;
    const vector<uint32_t> &sample_weights = node->subset->sample_weights;

    const uint32_t num_classes = split_data.dataset.num_classes;

    for (uint32_t idx = 0; idx != node->size; ++idx) {
      uint32_t sample_id = sample_ids[idx];
      uint32_t bin = feature[idx];
      uint32_t label = labels[idx];
      uint32_t sample_weight = sample_weights[idx];
      data_t weight = sample_weight * (*split_data.class_weights)[label];
      split_data.binwise_num_samples[bin] += sample_weight;
      split_data.bin_class_matrix[bin * num_classes + label] += weight;
    }

    for (uint32_t idx = 0; idx != split_data.dataset.num_bins[feature_idx]; ++idx)
      if (split_data.binwise_num_samples[idx])
        split_data.bin_ids[split_data.num_bins++] = idx;

    for (uint32_t idx = 0; idx != split_data.num_bins; ++idx) {
      uint32_t bin = split_data.bin_ids[idx];
      uint32_t offset = bin * num_classes;
      split_data.binwise_wnum_samples[bin] = accumulate(split_data.bin_class_matrix.begin() + offset,
                                                        split_data.bin_class_matrix.begin() + offset + num_classes,
                                                        static_cast<data_t>(0));
    }

    cost_computer.Init(node, split_data.init_left, split_data.wnum_samples, split_data.updater_left);
    copy(split_data.init_left.begin(), split_data.init_left.end(), split_data.cur_left.begin());
    fill(split_data.init_right.begin(), split_data.init_right.end(), static_cast<data_t>(0));
    fill(split_data.cur_right.begin(), split_data.cur_right.end(), static_cast<data_t>(0));
    split_data.num_samples = node->stats->num_samples;
    split_data.num_samples_left = node->stats->num_samples;
    split_data.num_samples_right = 0;
    split_data.wnum_samples_left = split_data.wnum_samples;
    split_data.wnum_samples_right = 0;
  }

  void Clear() {
    const uint32_t num_classes = split_data.dataset.num_classes;
    for (uint32_t idx = 0; idx != split_data.num_bins; ++idx) {
      uint32_t bin = split_data.bin_ids[idx];
      split_data.binwise_wnum_samples[bin] = 0;
      split_data.binwise_num_samples[bin] = 0;
      uint32_t offset = bin * num_classes;
      fill(split_data.bin_class_matrix.begin() + offset,
           split_data.bin_class_matrix.begin() + offset + num_classes,
           static_cast<data_t>(0));
    }
    split_data.num_bins = 0;
  }

  void MoveOneBinLToR(uint32_t bin,
                      double &cost) {
    uint32_t num_classes = split_data.dataset.num_classes;
    split_data.util.VectorMove(num_classes, split_data.bin_class_matrix, num_classes * bin,
                               split_data.cur_left, split_data.cur_right);
    split_data.num_samples_left -= split_data.binwise_num_samples[bin];
    split_data.num_samples_right += split_data.binwise_num_samples[bin];
    split_data.wnum_samples_left -= split_data.binwise_wnum_samples[bin];
    split_data.wnum_samples_right += split_data.binwise_wnum_samples[bin];

    cost = cost_computer.ComputeCost(split_data, split_data.cur_left, split_data.wnum_samples_left, 0, num_classes) +
           cost_computer.ComputeCost(split_data, split_data.cur_right, split_data.wnum_samples_right, 0, num_classes);
  }

  void MoveOneBinRToL(uint32_t bin,
                      double &cost) {
    uint32_t num_classes = split_data.dataset.num_classes;
    split_data.util.VectorMove(num_classes, split_data.bin_class_matrix, num_classes * bin,
                               split_data.cur_right, split_data.cur_left);
    split_data.num_samples_left += split_data.binwise_num_samples[bin];
    split_data.num_samples_right -= split_data.binwise_num_samples[bin];
    split_data.wnum_samples_left += split_data.binwise_wnum_samples[bin];
    split_data.wnum_samples_right -= split_data.binwise_wnum_samples[bin];

    cost = cost_computer.ComputeCost(split_data, split_data.cur_left, split_data.wnum_samples_left, 0, num_classes) +
           cost_computer.ComputeCost(split_data, split_data.cur_right, split_data.wnum_samples_right, 0, num_classes);
  }

  void SetOneVsAll(uint32_t bin,
                   double &cost) {
    uint32_t num_classes = split_data.dataset.num_classes;
    uint32_t offset = bin * num_classes;
    split_data.util.VectorMinusOutOfPlace(num_classes, split_data.init_left, split_data.bin_class_matrix,
                                          offset, split_data.cur_left);
    split_data.num_samples_left = split_data.num_samples - split_data.binwise_num_samples[bin];
    split_data.num_samples_right = split_data.binwise_num_samples[bin];
    split_data.wnum_samples_left = split_data.wnum_samples - split_data.binwise_wnum_samples[bin];
    split_data.wnum_samples_right = split_data.binwise_wnum_samples[bin];
    cost = cost_computer.ComputeCost(split_data, split_data.cur_left, split_data.wnum_samples_left, 0, num_classes) +
           cost_computer.ComputeCost(split_data, split_data.bin_class_matrix, split_data.wnum_samples_right, offset, num_classes);
  }

  void ReorderBinIds() {
    const uint32_t num_classes = 2;
    vector<double> &fractions = split_data.fractions;
    for (uint32_t idx = 0; idx != split_data.num_bins; ++idx) {
      uint32_t bin = split_data.bin_ids[idx];
      fractions[bin] = static_cast<double>(split_data.bin_class_matrix[bin * num_classes]) /
                       static_cast<double>(split_data.binwise_wnum_samples[bin]);
    }
    sort(split_data.bin_ids.begin(),
         split_data.bin_ids.begin() + split_data.num_bins,
         [&fractions](uint32_t x, uint32_t y) {
           return fractions[x] < fractions[y];
         });
  }

  void MoveOneBinOutOfPlace(uint32_t bin,
                            double &cost) {
    uint32_t num_classes = split_data.dataset.num_classes;
    uint32_t offset = num_classes * bin;
    split_data.util.VectorMinusOutOfPlace(num_classes, split_data.init_left, split_data.bin_class_matrix, offset,
                                          split_data.cur_left);
    split_data.util.VectorAddOutOfPlace(num_classes, split_data.init_right, split_data.bin_class_matrix, offset,
                                        split_data.cur_right);
    uint32_t num_samples_left_cur = split_data.num_samples_left - split_data.binwise_num_samples[bin];
    uint32_t num_samples_right_cur = split_data.num_samples_right + split_data.binwise_num_samples[bin];
    data_t wnum_samples_left_cur = split_data.wnum_samples_left - split_data.binwise_wnum_samples[bin];
    data_t wnum_samples_right_cur = split_data.wnum_samples_right + split_data.binwise_wnum_samples[bin];
    cost = cost_computer.ComputeCost(split_data, split_data.cur_left, num_samples_left_cur, 0, num_classes) +
           cost_computer.ComputeCost(split_data, split_data.cur_right, num_samples_right_cur, 0, num_classes);
  }

  void MoveOneBinInPlace(uint32_t bin) {
    uint32_t num_classes = split_data.dataset.num_classes;
    uint32_t offset = num_classes * bin;
    split_data.util.VectorMove(num_classes, split_data.bin_class_matrix, offset,
                               split_data.init_left, split_data.init_right);
    split_data.num_samples_left -= split_data.binwise_num_samples[bin];
    split_data.num_samples_right += split_data.binwise_num_samples[bin];
    split_data.wnum_samples_left -= split_data.binwise_wnum_samples[bin];
    split_data.wnum_samples_right += split_data.binwise_wnum_samples[bin];
  }

  bool LessThanMinLeafNode() {
    return split_data.num_samples_left < split_data.params.min_leaf_node ||
           split_data.num_samples_right < split_data.params.min_leaf_node;
  }

  uint32_t GetNumBins() {
    return split_data.num_bins;
  }

  uint32_t GetMaxNumBins(uint32_t feature_idx) {
    return split_data.dataset.num_bins[feature_idx];
  }

  uint32_t GetBinId(uint32_t idx) {
    return split_data.bin_ids[idx];
  }

  void ShuffleBinId(uint32_t n,
                    uint32_t k) {
    split_data.util.SampleWithoutReplacement(n, k, split_data.bin_ids);
  }

  void SwitchWithLast(uint32_t idx,
                      uint32_t num_bins) {
    uint32_t temp = split_data.bin_ids[idx];
    split_data.bin_ids[idx] = split_data.bin_ids[num_bins - 1];
    split_data.bin_ids[num_bins - 1] = temp;
  }

 private:
  SplitData<data_t> split_data;
  CostComputer cost_computer;
};

class GiniCostComputer {
 public:

  void LinkClassWeights(const Dataset &dataset,
                        const vector<double> *&class_weights) {
    class_weights = dataset.class_weights;
  }

  void Init(const TreeNode *node,
            vector<double> &init_histo,
            double &init_wnum_samples,
            double &init_updater) {
    copy(node->stats->histogram.begin(), node->stats->histogram.end(), init_histo.begin());
    init_wnum_samples = node->stats->weighted_num_samples;
    init_updater = node->stats->weighted_num_samples * node->stats->cost;
  }

  void UpdateCost(SplitData<double> &split_data,
                  double weight,
                  double wnum_all_left,
                  double wnum_one_left,
                  double wnum_all_right,
                  double wnum_one_right,
                  double &cost) {
    split_data.updater_left -= 2 * weight * (wnum_all_left - wnum_one_left);
    split_data.updater_right += 2 * weight * (wnum_all_right - wnum_one_right);
    cost = split_data.updater_left / wnum_all_left + split_data.updater_right / wnum_all_right;
  }

  double ComputeCost(SplitData<double> &split_data,
                     const vector<double> &histo,
                     double wnum_samples,
                     uint32_t begin,
                     uint32_t num_classes) {
    double cost = 0.0;
    for (uint32_t idx = begin; idx != begin + num_classes; ++idx)
      cost += histo[idx] * (wnum_samples - histo[idx]);
    cost /= wnum_samples;
    return cost;
  }
};

class EntropyCostComputer {
 public:

  void LinkClassWeights(const Dataset &dataset,
                        const vector<uint32_t> *&class_weights) {
    class_weights = &dataset.integral_class_weights;
  }

  void Init(const TreeNode *node,
            vector<uint32_t> &init_histo,
            uint32_t &init_wnum_samples,
            double &init_updater) {
    copy(node->stats->integral_histogram.begin(), node->stats->integral_histogram.end(), init_histo.begin());
    init_wnum_samples = node->stats->integral_num_samples;
    init_updater = node->stats->cost;
  }

  void UpdateCost(SplitData<uint32_t> &split_data,
                  uint32_t weight,
                  uint32_t wnum_all_left,
                  uint32_t wnum_one_left,
                  uint32_t wnum_all_right,
                  uint32_t wnum_one_right,
                  double &cost) {
    split_data.updater_left -= split_data.util.DeltaNLogN(wnum_all_left + weight, wnum_all_left) -
                               split_data.util.DeltaNLogN(wnum_one_left + weight, wnum_one_left);
    split_data.updater_right += split_data.util.DeltaNLogN(wnum_all_right, wnum_all_right - weight) -
                                split_data.util.DeltaNLogN(wnum_one_right, wnum_one_right - weight);
    cost = split_data.updater_left + split_data.updater_right;
  }

  double ComputeCost(SplitData<uint32_t> &split_data,
                     const vector<uint32_t> &histo,
                     uint32_t wnum_samples,
                     uint32_t begin,
                     uint32_t num_classes) {
    double cost = split_data.util.NLogN(wnum_samples);
    for (uint32_t idx = begin; idx != begin + num_classes; ++idx)
      cost -= split_data.util.NLogN(histo[idx]);
    return cost;
  }
};

#endif
