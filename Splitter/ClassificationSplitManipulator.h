
#ifndef DECISIONTREE_CLASSIFICATIONSPLITMANIPULATOR_H
#define DECISIONTREE_CLASSIFICATIONSPLITMANIPULATOR_H

#include <vector>
#include <cstdint>
#include "../Dataset/Dataset.h"
#include "../Tree/TreeParams.h"
#include "../Tree/TreeNode.h"
#include "../Util/Maths.h"
#include "../Util/Random.h"

using std::vector;
using std::copy;
using std::fill;
using std::iota;
using std::sort;

template <typename class_weight_t>
class ClassificationStats {
 public:
  vector<class_weight_t> init_left;
  vector<class_weight_t> init_right;
  vector<class_weight_t> cur_left;
  vector<class_weight_t> cur_right;
  vector<class_weight_t> bin_class_matrix;
  vector<class_weight_t> binwise_wnum_samples;
  vector<uint32_t> bin_ids;
  vector<double> fractions;
  class_weight_t wnum_samples_left;
  class_weight_t wnum_samples_right;
  class_weight_t wnum_samples;
  uint32_t num_bins;

  double updater_left;
  double updater_right;

  vector<class_weight_t> class_weights;
  const MetaData &meta;
  const TreeParams &params;

  ClassificationStats(const MetaData &meta,
                      const generic_vec_t &class_weights,
                      const TreeParams &params):
    meta(meta), params(params), init_left(meta.num_classes, 0), init_right(meta.num_classes, 0),
    cur_left(meta.num_classes, 0), cur_right(meta.num_classes, 0), bin_class_matrix(meta.max_num_bins * meta.num_classes, 0),
    binwise_wnum_samples(meta.max_num_bins, 0), bin_ids(meta.max_num_bins, 0), fractions(meta.max_num_bins, 0.0),
    wnum_samples_left(0), wnum_samples_right(0), wnum_samples(0), num_bins(0), updater_left(0.0), updater_right(0.0) {
    Maths::CastAndCopyVisitor<class_weight_t> visitor;
    this->class_weights = boost::apply_visitor(visitor, class_weights);
  }
};

template <typename class_weight_t, typename CostComputer>
class ClassificationSplitManipulator {
 public:
  explicit ClassificationSplitManipulator(const Dataset *dataset,
                                          const TreeParams &params):
          stats(dataset->Meta(), dataset->ClassWeights(), params), cost_computer() {}

  void NumericalInit(const TreeNode *node) {
    cost_computer.Init(node->Stats(), stats.cur_left, stats.wnum_samples, stats.updater_left);
    fill(stats.cur_right.begin(), stats.cur_right.end(), static_cast<class_weight_t>(0));
    stats.wnum_samples_left = stats.wnum_samples;
    stats.wnum_samples_right = 0;
    stats.updater_right = 0;
  }

  template <typename label_t>
  void MoveOneSample(const vector<label_t> &labels,
                     const vec_uint32_t &sample_weights,
                     uint32_t idx,
                     double &cost) {
    label_t label = labels[idx];
    class_weight_t weight = sample_weights[idx] * stats.class_weights[label];
    stats.cur_left[label] -= weight;
    stats.cur_right[label] += weight;
    stats.wnum_samples_left -= weight;
    stats.wnum_samples_right += weight;
    cost_computer.UpdateCost(stats, weight, stats.wnum_samples_left, stats.cur_left[label],
                             stats.wnum_samples_right, stats.cur_right[label], cost);
  }

  template <typename feature_t, typename label_t>
  void DiscreteInit(const vector<feature_t> &features,
                    const vector<label_t> &labels,
                    const vec_uint32_t &sample_weights,
                    uint32_t feature_idx,
                    TreeNode *node) {
    const uint32_t num_classes = stats.meta.num_classes;

    for (uint32_t idx = 0; idx != node->Size(); ++idx) {
      feature_t bin = features[idx];
      label_t label = labels[idx];
      uint32_t sample_weight = sample_weights[idx];
      class_weight_t weight = sample_weight * stats.class_weights[label];
      stats.bin_class_matrix[bin * num_classes + label] += weight;
    }

    for (uint32_t idx = 0; idx != stats.meta.num_bins[feature_idx]; ++idx) {
      uint32_t offset = idx * num_classes;
      stats.binwise_wnum_samples[idx] = accumulate(stats.bin_class_matrix.begin() + offset,
                                                   stats.bin_class_matrix.begin() + offset + num_classes,
                                                   static_cast<class_weight_t>(0));
      if (stats.binwise_wnum_samples[idx] > 0)
        stats.bin_ids[stats.num_bins++] = idx;
    }

    cost_computer.Init(node->Stats(), stats.init_left, stats.wnum_samples, stats.updater_left);
    copy(stats.init_left.begin(), stats.init_left.end(), stats.cur_left.begin());
    fill(stats.init_right.begin(), stats.init_right.end(), static_cast<class_weight_t>(0));
    fill(stats.cur_right.begin(), stats.cur_right.end(), static_cast<class_weight_t>(0));
    stats.wnum_samples_left = stats.wnum_samples;
    stats.wnum_samples_right = 0;
  }

  void Clear() {
    const uint32_t num_classes = stats.meta.num_classes;
    for (uint32_t idx = 0; idx != stats.num_bins; ++idx) {
      uint32_t bin = stats.bin_ids[idx];
      stats.binwise_wnum_samples[bin] = 0;
      uint32_t offset = bin * num_classes;
      fill(stats.bin_class_matrix.begin() + offset,
           stats.bin_class_matrix.begin() + offset + num_classes,
           static_cast<class_weight_t>(0));
    }
    stats.num_bins = 0;
  }

  void MoveOneBinLToR(uint32_t bin,
                      double &cost) {
    const uint32_t num_classes = stats.meta.num_classes;
    Maths::VectorMove(num_classes, stats.bin_class_matrix, num_classes * bin, stats.cur_left, stats.cur_right);
    stats.wnum_samples_left -= stats.binwise_wnum_samples[bin];
    stats.wnum_samples_right += stats.binwise_wnum_samples[bin];

    cost = cost_computer.ComputeCost(stats, stats.cur_left, stats.wnum_samples_left, 0, num_classes) +
           cost_computer.ComputeCost(stats, stats.cur_right, stats.wnum_samples_right, 0, num_classes);
  }

  void MoveOneBinRToL(uint32_t bin,
                      double &cost) {
    const uint32_t num_classes = stats.meta.num_classes;
    Maths::VectorMove(num_classes, stats.bin_class_matrix, num_classes * bin, stats.cur_right, stats.cur_left);
    stats.wnum_samples_left += stats.binwise_wnum_samples[bin];
    stats.wnum_samples_right -= stats.binwise_wnum_samples[bin];

    cost = cost_computer.ComputeCost(stats, stats.cur_left, stats.wnum_samples_left, 0, num_classes) +
           cost_computer.ComputeCost(stats, stats.cur_right, stats.wnum_samples_right, 0, num_classes);
  }

  void SetOneVsAll(uint32_t bin,
                   double &cost) {
    const uint32_t num_classes = stats.meta.num_classes;
    uint32_t offset = bin * num_classes;
    Maths::VectorMinusOutOfPlace(num_classes, stats.init_left, stats.bin_class_matrix, offset, stats.cur_left);
    stats.wnum_samples_left = stats.wnum_samples - stats.binwise_wnum_samples[bin];
    stats.wnum_samples_right = stats.binwise_wnum_samples[bin];
    cost = cost_computer.ComputeCost(stats, stats.cur_left, stats.wnum_samples_left, 0, num_classes) +
           cost_computer.ComputeCost(stats, stats.bin_class_matrix, stats.wnum_samples_right, offset, num_classes);
  }

  void ReorderBinIds() {
    const uint32_t num_classes = 2;
    vector<double> &fractions = stats.fractions;
    for (uint32_t idx = 0; idx != stats.num_bins; ++idx) {
      uint32_t bin = stats.bin_ids[idx];
      fractions[bin] = static_cast<double>(stats.bin_class_matrix[bin * num_classes]) /
                       static_cast<double>(stats.binwise_wnum_samples[bin]);
    }
    sort(stats.bin_ids.begin(),
         stats.bin_ids.begin() + stats.num_bins,
         [&fractions](uint32_t x, uint32_t y) {
           return fractions[x] < fractions[y];
         });
  }

  void MoveOneBinOutOfPlace(uint32_t bin,
                            double &cost) {
    const uint32_t num_classes = stats.meta.num_classes;
    uint32_t offset = num_classes * bin;
    Maths::VectorMinusOutOfPlace(num_classes, stats.init_left, stats.bin_class_matrix, offset, stats.cur_left);
    Maths::VectorAddOutOfPlace(num_classes, stats.init_right, stats.bin_class_matrix, offset, stats.cur_right);
    class_weight_t wnum_samples_left_cur = stats.wnum_samples_left - stats.binwise_wnum_samples[bin];
    class_weight_t wnum_samples_right_cur = stats.wnum_samples_right + stats.binwise_wnum_samples[bin];
    cost = cost_computer.ComputeCost(stats, stats.cur_left, wnum_samples_left_cur, 0, num_classes) +
           cost_computer.ComputeCost(stats, stats.cur_right, wnum_samples_right_cur, 0, num_classes);
  }

  void MoveOneBinInPlace(uint32_t bin) {
    const uint32_t num_classes = stats.meta.num_classes;
    uint32_t offset = num_classes * bin;
    Maths::VectorMove(num_classes, stats.bin_class_matrix, offset, stats.init_left, stats.init_right);
    stats.wnum_samples_left -= stats.binwise_wnum_samples[bin];
    stats.wnum_samples_right += stats.binwise_wnum_samples[bin];
  }

  bool LessThanMinLeafNode() {
    return stats.wnum_samples_left < stats.params.min_leaf_node ||
           stats.wnum_samples_right < stats.params.min_leaf_node;
  }

  template <typename feature_t>
  bool Splittable(const vector<feature_t> &features,
                  const vec_uint32_t &sample_ids,
                  const vec_uint32_t &sorted_idx,
                  uint32_t idx) {
    uint32_t first = sample_ids[sorted_idx[idx]];
    uint32_t second = sample_ids[sorted_idx[idx + 1]];
    return features[first] != features[second];
  }

  template <typename feature_t>
  float NumericalThreshold(const vector<feature_t> &features,
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
    Random::PartialShuffle(n, k, stats.bin_ids);
  }

  void SwitchWithLast(uint32_t idx,
                      uint32_t num_bins) {
    uint32_t temp = stats.bin_ids[idx];
    stats.bin_ids[idx] = stats.bin_ids[num_bins - 1];
    stats.bin_ids[num_bins - 1] = temp;
  }

 private:
  ClassificationStats<class_weight_t> stats;
  CostComputer cost_computer;
};

class GiniCostComputer {
 public:

  void Init(const NodeStats *node_stats,
            vector<double> &init_histo,
            double &init_wnum_samples,
            double &init_updater) {
    Maths::CastAndCopyVisitor<double> visitor;
    vector<double> temp_init_histo = boost::apply_visitor(visitor, node_stats->Histogram());
    copy(temp_init_histo.begin(), temp_init_histo.end(), init_histo.begin());
    init_wnum_samples = Generics::Round<double>(node_stats->WNumSamples());
    init_updater = init_wnum_samples * node_stats->Cost();
  }

  void UpdateCost(ClassificationStats<double> &stats,
                  double weight,
                  double wnum_all_left,
                  double wnum_one_left,
                  double wnum_all_right,
                  double wnum_one_right,
                  double &cost) {
    stats.updater_left -= 2 * weight * (wnum_all_left - wnum_one_left);
    stats.updater_right += 2 * weight * (wnum_all_right - wnum_one_right);
    cost = stats.updater_left / wnum_all_left + stats.updater_right / wnum_all_right;
  }

  double ComputeCost(ClassificationStats<double> &stats,
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
  void Init(const NodeStats *node_stats,
            vector<uint32_t> &init_histo,
            uint32_t &init_wnum_samples,
            double &init_updater) {
    Maths::CastAndCopyVisitor<uint32_t> visitor;
    vector<uint32_t> temp_init_histo = boost::apply_visitor(visitor, node_stats->Histogram());
    copy(temp_init_histo.begin(), temp_init_histo.end(), init_histo.begin());
    init_wnum_samples = Generics::Round<uint32_t>(node_stats->WNumSamples());
    init_updater = node_stats->Cost();
  }

  void UpdateCost(ClassificationStats<uint32_t> &stats,
                  uint32_t weight,
                  uint32_t wnum_all_left,
                  uint32_t wnum_one_left,
                  uint32_t wnum_all_right,
                  uint32_t wnum_one_right,
                  double &cost) {
    stats.updater_left -= Cost::DeltaNLogN(wnum_all_left + weight, wnum_all_left) -
                          Cost::DeltaNLogN(wnum_one_left + weight, wnum_one_left);
    stats.updater_right += Cost::DeltaNLogN(wnum_all_right, wnum_all_right - weight) -
                           Cost::DeltaNLogN(wnum_one_right, wnum_one_right - weight);
    cost = stats.updater_left + stats.updater_right;
  }

  double ComputeCost(ClassificationStats<uint32_t> &stats,
                     const vector<uint32_t> &histo,
                     uint32_t wnum_samples,
                     uint32_t begin,
                     uint32_t num_classes) {
    double cost = Cost::NLogN(wnum_samples);
    for (uint32_t idx = begin; idx != begin + num_classes; ++idx)
      cost -= Cost::NLogN(histo[idx]);
    return cost;
  }
};

using GiniSplitManipulator = ClassificationSplitManipulator<double, GiniCostComputer>;
using EntropySplitManipulator = ClassificationSplitManipulator<uint32_t, EntropyCostComputer>;

#endif
