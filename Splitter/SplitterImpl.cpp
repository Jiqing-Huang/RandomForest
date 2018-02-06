
#include <cstdint>
#include <cfloat>
#include <boost/variant.hpp>
#include "SplitterImpl.h"
#include "../Tree/ParallelTreeNode.h"

template <typename SplitManipulatorType>
SplitterImpl<SplitManipulatorType>::SplitterImpl():
  BaseSplitterImpl::BaseSplitterImpl(), cost_function(UndefinedCost), num_classes(0) {}

template <typename SplitManipulatorType>
void SplitterImpl<SplitManipulatorType>::Init(const uint32_t num_threads,
                                                     const Dataset *dataset,
                                                     const TreeParams &params) {
  split_manipulators.reserve(num_threads);
  for (uint32_t idx = 0; idx != num_threads; ++idx)
    split_manipulators.emplace_back(dataset, params);
  cost_function = params.cost_function;
  num_classes = dataset->Meta().num_classes;
}

template <typename SplitManipulatorType>
void SplitterImpl<SplitManipulatorType>::CleanUp() {
  split_manipulators.clear();
  split_manipulators.shrink_to_fit();
}

template <typename SplitManipulatorType>
void SplitterImpl<SplitManipulatorType>::Split(const uint32_t feature_idx,
                                                      const uint32_t feature_type,
                                                      const Dataset *dataset,
                                                      TreeNode *node) {
  if (feature_type == IsContinuous) {
    boost::apply_visitor([this, &feature_idx, &dataset, &node] (const auto &features, const auto &labels) {
      this->ContinuousSplit(features, labels, node->Subset()->SortedSampleWeights(feature_idx),
                            feature_idx, node);
    }, dataset->Features(feature_idx), node->Subset()->SortedLabels(feature_idx));
  } else {
    boost::apply_visitor([this, &feature_idx, &feature_type, &node] (const auto &features, const auto &labels) {
      this->DiscreteSplit(features, labels, node->Subset()->SampleWeights(),
                          feature_idx, feature_type, node);
    }, node->Subset()->Features(feature_idx), node->Subset()->Labels());
  }
}

template <typename SplitManipulatorType>
template <typename feature_t, typename label_t>
std::enable_if_t<IS_VALID_LABEL && !IS_INTEGRAL_FEATURE, void>
SplitterImpl<SplitManipulatorType>::ContinuousSplit(const vector<feature_t> &features,
                                                           const vector<label_t> &labels,
                                                           const vec_uint32_t &sample_weights,
                                                           const uint32_t feature_idx,
                                                           TreeNode *node) {
  uint32_t thread_id = ParallelTreeNodeNonMember::GetThreadId(feature_idx, node);
  split_manipulators[thread_id].NumericalInit(node);
  NumericalSplitter(features, labels, sample_weights, feature_idx, thread_id, node);
}

template <typename SplitManipulatorType>
template <typename feature_t, typename label_t>
std::enable_if_t<!IS_VALID_LABEL || IS_INTEGRAL_FEATURE, void>
SplitterImpl<SplitManipulatorType>::ContinuousSplit(const vector<feature_t> &features,
                                                           const vector<label_t> &labels,
                                                           const vec_uint32_t &sample_weights,
                                                           const uint32_t feature_idx,
                                                           TreeNode *node) {}

template <typename SplitManipulatorType>
template <typename feature_t, typename label_t>
std::enable_if_t<IS_VALID_LABEL && IS_INTEGRAL_FEATURE, void>
SplitterImpl<SplitManipulatorType>::DiscreteSplit(const vector<feature_t> &features,
                                                         const vector<label_t> &labels,
                                                         const vec_uint32_t &sample_weights,
                                                         const uint32_t feature_idx,
                                                         const uint32_t feature_type,
                                                         TreeNode *node) {
  uint32_t thread_id = ParallelTreeNodeNonMember::GetThreadId(feature_idx, node);
  split_manipulators[thread_id].DiscreteInit(features, labels, sample_weights, feature_idx, node);
  if (split_manipulators[thread_id].NumBins() > 1) {
    if (feature_type == IsOrdinal) {
      OrdinalSplitter(feature_idx, thread_id, node);
    } else if (feature_type == IsOneVsAll) {
      OneVsAllSplitter(feature_idx, thread_id, node);
    } else if (feature_type == IsManyVsMany) {
      if (cost_function == Variance || num_classes == 2) {
        LinearSplitter(feature_idx, thread_id, node);
      } else if (split_manipulators[thread_id].NumBins() <= MaxNumBinsForBruteSplitter) {
        BruteSplitter(feature_idx, thread_id, node);
      } else {
        GreedySplitter(feature_idx, thread_id, node);
      }
    }
  }
  split_manipulators[thread_id].Clear();
}

template <typename SplitManipulatorType>
template <typename feature_t, typename label_t>
std::enable_if_t<!IS_VALID_LABEL || !IS_INTEGRAL_FEATURE, void>
SplitterImpl<SplitManipulatorType>::DiscreteSplit(const vector<feature_t> &features,
                                                         const vector<label_t> &labels,
                                                         const vec_uint32_t &sample_weights,
                                                         const uint32_t feature_idx,
                                                         const uint32_t feature_type,
                                                         TreeNode *node) {}

template <typename SplitManipulatorType>
template <typename feature_t, typename label_t>
std::enable_if_t<IS_VALID_LABEL && !IS_INTEGRAL_FEATURE, void>
SplitterImpl<SplitManipulatorType>::NumericalSplitter(const vector<feature_t> &features,
                                                             const vector<label_t> &labels,
                                                             const vec_uint32_t &sample_weights,
                                                             const uint32_t feature_idx,
                                                             const uint32_t thread_id,
                                                             TreeNode *node) {
  double lowest_cost = node->Stats()->Cost();
  double cost = 0.0;
  uint32_t best_idx = 0;

  SplitManipulatorType &split_manipulator = split_manipulators[thread_id];
  const vector<uint32_t> &sample_ids = node->Subset()->SampleIds();
  const vector<uint32_t> &sorted_idx = node->Subset()->SortedIdx(feature_idx);

  for (uint32_t idx = 0; idx != node->Size() - 1; ++idx) {
    split_manipulator.MoveOneSample(labels, sample_weights, idx, cost);
    if (split_manipulator.LessThanMinLeafNode()) continue;
    if (cost < lowest_cost && split_manipulator.Splittable(features, sample_ids, sorted_idx, idx)) {
      lowest_cost = cost;
      best_idx = idx;
    }
  }

  float threshold = split_manipulator.NumericalThreshold(features, sample_ids, sorted_idx, best_idx);
  node->Split()->UpdateFloat(ParallelTreeNodeNonMember::IsParallelNode(node), node->Stats()->Cost() - lowest_cost,
                             IsContinuous, feature_idx, threshold);
}

template <typename SplitManipulatorType>
void SplitterImpl<SplitManipulatorType>::OrdinalSplitter(const uint32_t feature_idx,
                                                                const uint32_t thread_id,
                                                                TreeNode *node) {
  double lowest_cost = node->Stats()->Cost();
  double cost = 0.0;
  uint32_t best_ordinal_ceiling = 0;

  SplitManipulatorType &split_manipulator = split_manipulators[thread_id];
  uint32_t num_bins = split_manipulator.NumBins();

  for (uint32_t idx = 0; idx != num_bins; ++idx) {
    uint32_t bin = split_manipulator.BinId(idx);
    split_manipulator.MoveOneBinLToR(bin, cost);
    if (split_manipulator.LessThanMinLeafNode()) continue;
    if (cost < lowest_cost) {
      lowest_cost = cost;
      best_ordinal_ceiling = bin;
    }
  }

  node->Split()->UpdateUInt(ParallelTreeNodeNonMember::IsParallelNode(node), node->Stats()->Cost() - lowest_cost,
                            IsOrdinal, feature_idx, best_ordinal_ceiling);
}

template <typename SplitManipulatorType>
void SplitterImpl<SplitManipulatorType>::OneVsAllSplitter(const uint32_t feature_idx,
                                                                 const uint32_t thread_id,
                                                                 TreeNode *node) {
  double lowest_cost = node->Stats()->Cost();
  double cost = 0.0;
  uint32_t best_on_vs_all = 0;

  SplitManipulatorType &split_manipulator = split_manipulators[thread_id];
  uint32_t num_bins = split_manipulator.NumBins();

  for (uint32_t idx = 0; idx != num_bins; ++idx) {
    uint32_t bin = split_manipulator.BinId(idx);
    split_manipulator.SetOneVsAll(bin, cost);
    if (split_manipulator.LessThanMinLeafNode()) continue;
    if (cost < lowest_cost) {
      lowest_cost = cost;
      best_on_vs_all = bin;
    }
  }

  node->Split()->UpdateUInt(ParallelTreeNodeNonMember::IsParallelNode(node), node->Stats()->Cost() - lowest_cost,
                            IsOneVsAll, feature_idx, best_on_vs_all);
}

template <typename SplitManipulatorType>
void SplitterImpl<SplitManipulatorType>::LinearSplitter(const uint32_t feature_idx,
                                                               const uint32_t thread_id,
                                                               TreeNode *node) {
  double lowest_cost = node->Stats()->Cost();
  double cost = 0.0;
  uint32_t best_linear_ceiling = 0;

  SplitManipulatorType &split_manipulator = split_manipulators[thread_id];
  split_manipulator.ReorderBinIds();
  uint32_t num_bins = split_manipulator.NumBins();

  for (uint32_t idx = 0; idx != num_bins; ++idx) {
    uint32_t bin = split_manipulator.BinId(idx);
    split_manipulator.MoveOneBinLToR(bin, cost);
    if (split_manipulator.LessThanMinLeafNode()) continue;
    if (cost < lowest_cost) {
      lowest_cost = cost;
      best_linear_ceiling = idx;
    }
  }

  vec_uint32_t indicators(best_linear_ceiling + 1, 0);
  iota(indicators.begin(), indicators.end(), 0);
  UpdateManyVsManySplit(indicators, feature_idx, node->Stats()->Cost() - lowest_cost, thread_id, node);
}

template <typename SplitManipulatorType>
void SplitterImpl<SplitManipulatorType>::BruteSplitter(const uint32_t feature_idx,
                                                              const uint32_t thread_id,
                                                              TreeNode *node){
  double lowest_cost = node->Stats()->Cost();
  double cost = 0.0;
  uint32_t best_bitmask = 0;

  SplitManipulatorType &split_manipulator = split_manipulators[thread_id];
  uint32_t num_bins = split_manipulator.NumBins();
  uint32_t num_flips = 1u << (num_bins - 1);
  uint32_t bitmask = 0x0;
  for (uint32_t ite = 1; ite != num_flips; ++ite) {
    uint32_t idx = static_cast<uint32_t>(ffs(ite) - 1);
    uint32_t mask = 1u << idx;
    bool left_to_right = !(bitmask & mask);
    bitmask ^= mask;
    uint32_t bin = split_manipulator.BinId(idx);
    if (left_to_right) {
      split_manipulator.MoveOneBinLToR(bin, cost);
    } else {
      split_manipulator.MoveOneBinRToL(bin, cost);
    }
    if (split_manipulator.LessThanMinLeafNode()) continue;
    if (cost < lowest_cost) {
      lowest_cost = cost;
      best_bitmask = bitmask;
    }
  }

  vec_uint32_t indicators;
  for (uint32_t idx = 0; idx != num_bins; ++idx)
    if (best_bitmask & (1 << idx))
      indicators.push_back(idx);
  UpdateManyVsManySplit(indicators, feature_idx, node->Stats()->Cost() - lowest_cost, thread_id, node);
}

template <typename SplitManipulatorType>
void SplitterImpl<SplitManipulatorType>::GreedySplitter(const uint32_t feature_idx,
                                                               const uint32_t thread_id,
                                                               TreeNode *node) {
  double global_lowest_cost = node->Stats()->Cost();
  double lowest_cost = DBL_MAX;
  double cost = 0.0;
  uint32_t best_idx = 0;
  uint32_t best_num_bins_left = 0;

  SplitManipulatorType &split_manipulator = split_manipulators[thread_id];
  uint32_t num_bins = split_manipulator.NumBins();

  for (uint32_t num_bins_left = num_bins; num_bins_left != 1; --num_bins_left) {
    uint32_t num_bins_to_sample = (num_bins_left < MaxNumBinsForSampling)? num_bins_left : MaxNumBinsForSampling;
    split_manipulator.ShuffleBinId(num_bins_left, num_bins_to_sample);
    for (uint32_t idx = 0; idx != num_bins_to_sample; ++idx) {
      uint32_t bin = split_manipulator.BinId(idx);
      split_manipulator.MoveOneBinOutOfPlace(bin, cost);
      if (cost < lowest_cost) {
        lowest_cost = cost;
        best_idx = idx;
      }
    }
    split_manipulator.MoveOneBinInPlace(split_manipulator.BinId(best_idx));
    split_manipulator.SwitchWithLast(best_idx, num_bins_left);
    if (lowest_cost < global_lowest_cost && !split_manipulator.LessThanMinLeafNode()) {
      global_lowest_cost = lowest_cost;
      best_num_bins_left = num_bins_left - 1;
    }
    lowest_cost = DBL_MAX;
  }

  vec_uint32_t indicators(best_num_bins_left, 0);
  iota(indicators.begin(), indicators.end(), 0);
  UpdateManyVsManySplit(indicators, feature_idx, node->Stats()->Cost() - global_lowest_cost, thread_id, node);
}

template <typename SplitManipulatorType>
void SplitterImpl<SplitManipulatorType>::UpdateManyVsManySplit(const vec_uint32_t &indicators,
                                                                      const uint32_t feature_idx,
                                                                      const double gain,
                                                                      const uint32_t thread_id,
                                                                      TreeNode *node) {
  bool is_parallel = ParallelTreeNodeNonMember::IsParallelNode(node);
  SplitManipulatorType &split_manipulator = split_manipulators[thread_id];
  uint32_t max_num_bins = split_manipulator.MaxNumBins(feature_idx);
  if (max_num_bins <= NumBitsPerWord) {
    uint32_t bitmask = 0;
    for (const auto &idx: indicators) {
      uint32_t bin = split_manipulator.BinId(idx);
      bitmask |= (1 << bin);
    }
    node->Split()->UpdateUInt(is_parallel, gain, IsLowCardinality, feature_idx, bitmask);
  } else {
    uint32_t bitmask_size = (max_num_bins + NumBitsPerWord - 1) / NumBitsPerWord;
    vec_uint32_t bitmask(bitmask_size, 0);
    for (const auto &idx: indicators) {
      uint32_t bin = split_manipulator.BinId(idx);
      uint32_t mask_idx = bin >> GetMaskIdx;
      uint32_t mask_shift = bin & GetMaskShift;
      bitmask[mask_idx] |= (1 << mask_shift);
    }
    node->Split()->UpdatePtr(is_parallel, gain, IsHighCardinality, feature_idx, bitmask);
  }
}

template class SplitterImpl<GiniSplitManipulator>;
template class SplitterImpl<EntropySplitManipulator>;
template class SplitterImpl<VarianceSplitManipulator>;