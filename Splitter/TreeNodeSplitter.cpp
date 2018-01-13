
#include "TreeNodeSplitter.h"

#include <cfloat>

template<typename data_t, typename CostComputer>
void TreeNodeSplitter<data_t, CostComputer>::NumericalSplitter
        (uint32_t feature_idx,
         SplitManipulator<data_t, CostComputer> &split_manipulator,
         TreeNode *node) {
  double lowest_cost = node->stats->cost;
  double cost = 0.0;
  uint32_t best_idx = 0;

  const vector<float> &feature = node->subset->numerical_features[feature_idx];
  const vector<uint32_t> &sorted_idx = node->subset->sorted_indices[feature_idx];
  const vector<uint32_t> &labels = node->subset->labels;
  const vector<uint32_t> &sample_weights = node->subset->sample_weights;

  uint32_t loop_begin, loop_end;
  split_manipulator.GetLoopBound(feature_idx, sorted_idx, sample_weights, loop_begin, loop_end);

  for (uint32_t idx = 0; idx != loop_begin; ++idx) {
    uint32_t sample_id = sorted_idx[idx];
    split_manipulator.MoveOneSample(labels, sample_weights, sample_id, cost);
  }

  for (uint32_t idx = loop_begin; idx != loop_end; ++idx) {
    uint32_t sample_id = sorted_idx[idx];
    split_manipulator.MoveOneSample(labels, sample_weights, sample_id, cost);
    if (cost < lowest_cost && feature[sorted_idx[idx]] != feature[sorted_idx[idx + 1]]) {
      lowest_cost = cost;
      best_idx = idx;
    }
  }

  float threshold = (feature[sorted_idx[best_idx]] + feature[sorted_idx[best_idx + 1]]) / 2;
  bool is_sync = ((node->type & IsParallelSplitting) != 0);
  node->split_info->UpdateFloat(is_sync, node->stats->cost - lowest_cost, IsContinuous, feature_idx, threshold);
}

template <typename data_t, typename CostComputer>
void TreeNodeSplitter<data_t, CostComputer>::OrdinalSplitter
        (uint32_t feature_idx,
         SplitManipulator<data_t, CostComputer> &split_manipulator,
         TreeNode *node) {
  double cost = 0.0;
  double lowest_cost = node->stats->cost;
  uint32_t best_ordinal_ceiling = 0;

  uint32_t num_bins = split_manipulator.GetNumBins();
  if (num_bins == 1) {
    split_manipulator.Clear();
    return;
  }

  for (uint32_t idx = 0; idx != num_bins; ++idx) {
    uint32_t bin = split_manipulator.GetBinId(idx);
    split_manipulator.MoveOneBinLToR(bin, cost);
    if (split_manipulator.LessThanMinLeafNode()) continue;
    if (cost < lowest_cost) {
      lowest_cost = cost;
      best_ordinal_ceiling = bin;
    }
  }

  bool is_sync = ((node->type & IsParallelSplitting) != 0);
  node->split_info->UpdateUInt(is_sync, node->stats->cost - lowest_cost, IsOrdinal, feature_idx, best_ordinal_ceiling);
  split_manipulator.Clear();
}

template <typename data_t, typename CostComputer>
void TreeNodeSplitter<data_t, CostComputer>::OneVsAllSplitter
        (uint32_t feature_idx,
         SplitManipulator<data_t, CostComputer> &split_manipulator,
         TreeNode *node) {
  double cost = 0.0;
  double lowest_cost = node->stats->cost;
  uint32_t best_on_vs_all = 0;

  uint32_t num_bins = split_manipulator.GetNumBins();
  if (num_bins == 1) {
    split_manipulator.Clear();
    return;
  }

  for (uint32_t idx = 0; idx != num_bins; ++idx) {
    uint32_t bin = split_manipulator.GetBinId(idx);
    split_manipulator.SetOneVsAll(bin, cost);
    if (split_manipulator.LessThanMinLeafNode()) continue;
    if (cost < lowest_cost) {
      lowest_cost = cost;
      best_on_vs_all = bin;
    }
  }

  bool is_sync = ((node->type & IsParallelSplitting) != 0);
  node->split_info->UpdateUInt(is_sync, node->stats->cost - lowest_cost, IsOneVsAll, feature_idx, best_on_vs_all);
  split_manipulator.Clear();
}

template <typename data_t, typename  CostComputer>
void TreeNodeSplitter<data_t, CostComputer>::LinearSplitter
        (uint32_t feature_idx,
         SplitManipulator<data_t, CostComputer> &split_manipulator,
         TreeNode *node) {
  split_manipulator.ReorderBinIds();

  double cost = 0.0;
  double lowest_cost = node->stats->cost;
  uint32_t best_linear_ceiling = 0;

  uint32_t num_bins = split_manipulator.GetNumBins();
  if (num_bins == 1) {
    split_manipulator.Clear();
    return;
  }

  for (uint32_t idx = 0; idx != num_bins; ++idx) {
    uint32_t bin = split_manipulator.GetBinId(idx);
    split_manipulator.MoveOneBinLToR(bin, cost);
    if (split_manipulator.LessThanMinLeafNode()) continue;
    if (cost < lowest_cost) {
      lowest_cost = cost;
      best_linear_ceiling = idx;
    }
  }

  uint32_t max_num_bins = split_manipulator.GetMaxNumBins(feature_idx);
  bool is_sync = ((node->type & IsParallelSplitting) != 0);
  if (max_num_bins <= NumBitsPerWord) {
    uint32_t bitmask = 0;
    for (uint32_t idx = 0; idx <= best_linear_ceiling; ++idx) {
      uint32_t bin = split_manipulator.GetBinId(idx);
      bitmask |= (1 << bin);
    }
    node->split_info->UpdateUInt(is_sync, node->stats->cost - lowest_cost, IsLowCardinality, feature_idx, bitmask);
  } else {
    uint32_t bitmask_size = (max_num_bins + NumBitsPerWord - 1) / NumBitsPerWord;
    vector<uint32_t> bitmask(bitmask_size, 0);
    for (uint32_t idx = 0; idx <= best_linear_ceiling; ++idx) {
      uint32_t bin = split_manipulator.GetBinId(idx);
      uint32_t mask_idx = bin >> GetMaskIdx;
      uint32_t mask_shift = bin & GetMaskShift;
      bitmask[mask_idx] |= (1 << mask_shift);
    }
    node->split_info->UpdatePtr(is_sync, node->stats->cost - lowest_cost, IsHighCardinality, feature_idx, bitmask);
  }
  split_manipulator.Clear();
}

template <typename data_t, typename CostComputer>
void TreeNodeSplitter<data_t, CostComputer>::ManyVsManySplitter
        (uint32_t feature_idx,
         SplitManipulator<data_t, CostComputer> &split_manipulator,
         TreeNode *node) {
  uint32_t num_bins = split_manipulator.GetNumBins();
  if (num_bins == 1) {
    split_manipulator.Clear();
    return;
  }
  if (num_bins <= MaxNumBinsForBruteSplitter) {
    BruteSplitter(feature_idx, split_manipulator, node);
  } else {
    GreedySplitter(feature_idx, split_manipulator, node);
  }
}

template <typename data_t, typename CostComputer>
void TreeNodeSplitter<data_t, CostComputer>::BruteSplitter
        (uint32_t feature_idx,
         SplitManipulator<data_t, CostComputer> &split_manipulator,
         TreeNode *node) {
  double cost = 0.0;
  double lowest_cost = node->stats->cost;
  uint32_t best_bitmask = 0;

  uint32_t num_bins = split_manipulator.GetNumBins();
  uint32_t num_flips = 1u << (num_bins - 1);
  uint32_t bitmask = 0x0;
  for (uint32_t ite = 1; ite != num_flips; ++ite) {
    uint32_t idx = Maths::NextToFlip(ite);
    uint32_t mask = 1u << idx;
    bool left_to_right = !(bitmask & mask);
    bitmask ^= mask;
    uint32_t bin = split_manipulator.GetBinId(idx);
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

  uint32_t max_num_bins = split_manipulator.GetMaxNumBins(feature_idx);
  bool is_sync = ((node->type & IsParallelSplitting) != 0);
  if (max_num_bins <= NumBitsPerWord) {
    uint32_t bin_bitmask = 0;
    for (uint32_t idx = 0; idx != num_bins; ++idx)
      if (best_bitmask & (1 << idx)){
        uint32_t bin = split_manipulator.GetBinId(idx);
        bin_bitmask |= (1 << bin);
      }
    node->split_info->UpdateUInt(is_sync, node->stats->cost - lowest_cost, IsLowCardinality, feature_idx, bin_bitmask);
  } else {
    uint32_t bitmask_size = (max_num_bins + NumBitsPerWord - 1) / NumBitsPerWord;
    vector<uint32_t> bin_bitmask(bitmask_size, 0);
    for (uint32_t idx = 0; idx != num_bins; ++idx)
      if (best_bitmask & (1 << idx)) {
        uint32_t bin = split_manipulator.GetBinId(idx);
        uint32_t mask_idx = bin >> GetMaskIdx;
        uint32_t mask_shift = bin & GetMaskShift;
        bin_bitmask[mask_idx] |= (1 << mask_shift);
      }
    node->split_info->UpdatePtr(is_sync, node->stats->cost - lowest_cost, IsHighCardinality, feature_idx, bin_bitmask);
  }
  split_manipulator.Clear();
}

template <typename data_t, typename CostComputer>
void TreeNodeSplitter<data_t, CostComputer>::GreedySplitter
        (uint32_t feature_idx,
         SplitManipulator<data_t, CostComputer> &split_manipulator,
         TreeNode *node) {
  double cost = 0.0;
  double lowest_cost = DBL_MAX;
  uint32_t best_idx = 0;
  double global_lowest_cost = node->stats->cost;
  uint32_t best_num_bins_left = 0;

  uint32_t num_bins = split_manipulator.GetNumBins();

  if (node->node_id == 113) {
    cost = 0.0;
  }

  for (uint32_t num_bins_left = num_bins; num_bins_left != 1; --num_bins_left) {
    uint32_t num_bins_to_sample = (num_bins_left < MaxNumBinsForSampling)? num_bins_left : MaxNumBinsForSampling;
    split_manipulator.ShuffleBinId(num_bins_left, num_bins_to_sample);
    for (uint32_t idx = 0; idx != num_bins_to_sample; ++idx) {
      uint32_t bin = split_manipulator.GetBinId(idx);
      split_manipulator.MoveOneBinOutOfPlace(bin, cost);
      if (cost < lowest_cost) {
        lowest_cost = cost;
        best_idx = idx;
      }
    }
    split_manipulator.MoveOneBinInPlace(split_manipulator.GetBinId(best_idx));
    split_manipulator.SwitchWithLast(best_idx, num_bins_left);
    if (lowest_cost < global_lowest_cost && !split_manipulator.LessThanMinLeafNode()) {
      global_lowest_cost = lowest_cost;
      best_num_bins_left = num_bins_left - 1;
    }
    lowest_cost = DBL_MAX;
  }

  uint32_t max_num_bins = split_manipulator.GetMaxNumBins(feature_idx);
  bool is_sync = ((node->type & IsParallelSplitting) != 0);
  if (max_num_bins <= NumBitsPerWord) {
    uint32_t bitmask = 0;
    for (uint32_t idx = 0; idx != best_num_bins_left; ++idx) {
      uint32_t bin = split_manipulator.GetBinId(idx);
      bitmask |= (1 << bin);
    }
    node->split_info->UpdateUInt(is_sync, node->stats->cost - global_lowest_cost, IsLowCardinality, feature_idx, bitmask);
  } else {
    uint32_t bitmask_size = (max_num_bins + NumBitsPerWord - 1) / NumBitsPerWord;
    vector<uint32_t> bitmask(bitmask_size, 0);
    for (uint32_t idx = 0; idx != best_num_bins_left; ++idx) {
      uint32_t bin = split_manipulator.GetBinId(idx);
      uint32_t mask_idx = bin >> GetMaskIdx;
      uint32_t mask_shift = bin & GetMaskShift;
      bitmask[mask_idx] |= (1 << mask_shift);
    }
    node->split_info->UpdatePtr(is_sync, node->stats->cost - global_lowest_cost, IsHighCardinality, feature_idx, bitmask);
  }
  split_manipulator.Clear();
}

template class TreeNodeSplitter<double, GiniCostComputer>;
template class TreeNodeSplitter<uint32_t, EntropyCostComputer>;