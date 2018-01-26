
#ifndef DECISIONTREE_TREENODESPLITTER_H
#define DECISIONTREE_TREENODESPLITTER_H

#include <cstdint>
#include <vector>
#include <algorithm>
#include <cfloat>

#include "SplitInfo.h"
#include "../Util/Maths.h"
#include "../Dataset/Dataset.h"
#include "../Tree/TreeNode.h"
#include "../Tree/ParallelTreeNode.h"
#include "ClassificationSplitManipulator.h"
#include "RegressionSplitManipulator.h"

using std::vector;
using std::conditional;
using std::is_same;

template <typename SplitManipulatorType>
class TreeNodeSplitter {
 public:
  TreeNodeSplitter() = default;

  void Init(uint32_t num_threads,
            const Dataset *dataset,
            const TreeParams &params) {
    split_manipulators.reserve(num_threads);
    for (uint32_t idx = 0; idx != num_threads; ++idx)
      split_manipulators.emplace_back(dataset, params);
    cost_function = params.cost_function;
    num_classes = dataset->Meta().num_classes;
  }

  void CleanUp() {
    split_manipulators.clear();
    split_manipulators.shrink_to_fit();
  }

  void Split(const uint32_t feature_idx,
             const uint32_t feature_type,
             const Dataset *dataset,
             TreeNode *node) {
    if (feature_type == IsContinuous) {
      SplitVisitor visitor(this, feature_idx, feature_type, node->Subset()->SortedSampleWeights(feature_idx), node);
      boost::apply_visitor(visitor, dataset->Features(feature_idx), node->Subset()->SortedLabels(feature_idx));
    } else {
      SplitVisitor visitor(this, feature_idx, feature_type, node->Subset()->SampleWeights(), node);
      boost::apply_visitor(visitor, node->Subset()->Features(feature_idx), node->Subset()->Labels());
    }
  }

 private:
  vector<SplitManipulatorType> split_manipulators;
  uint32_t cost_function;
  uint32_t num_classes;

  struct SplitVisitor: public boost::static_visitor<> {

    TreeNodeSplitter *splitter;
    const uint32_t feature_idx;
    const uint32_t feature_type;
    const vec_uint32_t &sample_weights;
    TreeNode *node;

    SplitVisitor(TreeNodeSplitter *splitter,
                 const uint32_t feature_idx,
                 const uint32_t feature_type,
                 const vec_uint32_t &sample_weights,
                 TreeNode *node):
      splitter(splitter), feature_idx(feature_idx), feature_type(feature_type), sample_weights(sample_weights),
      node(node) {}

    template <typename feature_t, typename label_t>
    void operator()(const vector<feature_t> &features,
                    const vector<label_t> &labels) {
      splitter->Split(features, labels, sample_weights, feature_idx, feature_type, node);
    }
  };

  template <typename feature_t, typename label_t>
  void Split(const vector<feature_t> &features,
             const vector<label_t> &labels,
             const vec_uint32_t &sample_weights,
             const uint32_t feature_idx,
             const uint32_t feature_type,
             TreeNode *node) {
    uint32_t thread_id = ParallelTreeNodeNonMember::GetThreadId(feature_idx, node);
    if (feature_type == IsContinuous) {
      split_manipulators[thread_id].NumericalInit(node);
      NumericalSplitter(features, labels, sample_weights, feature_idx, thread_id, node);
    } else {
      split_manipulators[thread_id].DiscreteInit(features, labels, sample_weights, feature_idx, node);
      if (split_manipulators[thread_id].NumBins() > 1) {
        if (feature_type == IsOrdinal) {
          OrdinalSplitter(features, labels, sample_weights, feature_idx, thread_id, node);
        } else if (feature_type == IsOneVsAll) {
          OneVsAllSplitter(features, labels, sample_weights, feature_idx, thread_id, node);
        } else if (feature_type == IsManyVsMany) {
          if (cost_function == Variance || num_classes == 2) {
            LinearSplitter(features, labels, sample_weights, feature_idx, thread_id, node);
          } else if (split_manipulators[thread_id].NumBins() <= MaxNumBinsForBruteSplitter) {
            BruteSplitter(features, labels, sample_weights, feature_idx, thread_id, node);
          } else {
            GreedySplitter(features, labels, sample_weights, feature_idx, thread_id, node);
          }
        }
      }
      split_manipulators[thread_id].Clear();
    }
  }

  template <typename feature_t, typename label_t>
  void NumericalSplitter(const vector<feature_t> &features,
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

  template <typename feature_t, typename label_t>
  void OrdinalSplitter(const vector<feature_t> &features,
                       const vector<label_t> &labels,
                       const vec_uint32_t &sample_weights,
                       const uint32_t feature_idx,
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

  template <typename feature_t, typename label_t>
  void OneVsAllSplitter(const vector<feature_t> &features,
                        const vector<label_t> &labels,
                        const vec_uint32_t &sample_weights,
                        const uint32_t feature_idx,
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

  template <typename feature_t, typename label_t>
  void LinearSplitter(const vector<feature_t> &features,
                      const vector<label_t> &labels,
                      const vec_uint32_t &sample_weights,
                      const uint32_t feature_idx,
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

    uint32_t max_num_bins = split_manipulator.MaxNumBins(feature_idx);
    if (max_num_bins <= NumBitsPerWord) {
      uint32_t bitmask = 0;
      for (uint32_t idx = 0; idx <= best_linear_ceiling; ++idx) {
        uint32_t bin = split_manipulator.BinId(idx);
        bitmask |= (1 << bin);
      }
      node->Split()->UpdateUInt(ParallelTreeNodeNonMember::IsParallelNode(node), node->Stats()->Cost() - lowest_cost,
                                IsLowCardinality, feature_idx, bitmask);
    } else {
      uint32_t bitmask_size = (max_num_bins + NumBitsPerWord - 1) / NumBitsPerWord;
      vector<uint32_t> bitmask(bitmask_size, 0);
      for (uint32_t idx = 0; idx <= best_linear_ceiling; ++idx) {
        uint32_t bin = split_manipulator.BinId(idx);
        uint32_t mask_idx = bin >> GetMaskIdx;
        uint32_t mask_shift = bin & GetMaskShift;
        bitmask[mask_idx] |= (1 << mask_shift);
      }
      node->Split()->UpdatePtr(ParallelTreeNodeNonMember::IsParallelNode(node), node->Stats()->Cost() - lowest_cost,
                               IsHighCardinality, feature_idx, bitmask);
    }
  }

  template <typename feature_t, typename label_t>
  void BruteSplitter(const vector<feature_t> &features,
                     const vector<label_t> &labels,
                     const vec_uint32_t &sample_weights,
                     const uint32_t feature_idx,
                     const uint32_t thread_id,
                     TreeNode *node) {
    double lowest_cost = node->Stats()->Cost();
    double cost = 0.0;
    uint32_t best_bitmask = 0;

    SplitManipulatorType &split_manipulator = split_manipulators[thread_id];
    uint32_t num_bins = split_manipulator.NumBins();
    uint32_t num_flips = 1u << (num_bins - 1);
    uint32_t bitmask = 0x0;
    for (uint32_t ite = 1; ite != num_flips; ++ite) {
      uint32_t idx = Maths::NextToFlip(ite);
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

    uint32_t max_num_bins = split_manipulator.MaxNumBins(feature_idx);
    if (max_num_bins <= NumBitsPerWord) {
      uint32_t bin_bitmask = 0;
      for (uint32_t idx = 0; idx != num_bins; ++idx)
        if (best_bitmask & (1 << idx)){
          uint32_t bin = split_manipulator.BinId(idx);
          bin_bitmask |= (1 << bin);
        }
      node->Split()->UpdateUInt(ParallelTreeNodeNonMember::IsParallelNode(node), node->Stats()->Cost() - lowest_cost,
                                IsLowCardinality, feature_idx, bin_bitmask);
    } else {
      uint32_t bitmask_size = (max_num_bins + NumBitsPerWord - 1) / NumBitsPerWord;
      vector<uint32_t> bin_bitmask(bitmask_size, 0);
      for (uint32_t idx = 0; idx != num_bins; ++idx)
        if (best_bitmask & (1 << idx)) {
          uint32_t bin = split_manipulator.BinId(idx);
          uint32_t mask_idx = bin >> GetMaskIdx;
          uint32_t mask_shift = bin & GetMaskShift;
          bin_bitmask[mask_idx] |= (1 << mask_shift);
        }
      node->Split()->UpdatePtr(ParallelTreeNodeNonMember::IsParallelNode(node), node->Stats()->Cost() - lowest_cost,
                               IsHighCardinality, feature_idx, bin_bitmask);
    }
  }

  template <typename feature_t, typename label_t>
  void GreedySplitter(const vector<feature_t> &features,
                      const vector<label_t> &labels,
                      const vec_uint32_t &sample_weights,
                      const uint32_t feature_idx,
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

    uint32_t max_num_bins = split_manipulator.MaxNumBins(feature_idx);
    if (max_num_bins <= NumBitsPerWord) {
      uint32_t bitmask = 0;
      for (uint32_t idx = 0; idx != best_num_bins_left; ++idx) {
        uint32_t bin = split_manipulator.BinId(idx);
        bitmask |= (1 << bin);
      }
      node->Split()->UpdateUInt(ParallelTreeNodeNonMember::IsParallelNode(node), node->Stats()->Cost() - global_lowest_cost,
                                IsLowCardinality, feature_idx, bitmask);
    } else {
      uint32_t bitmask_size = (max_num_bins + NumBitsPerWord - 1) / NumBitsPerWord;
      vector<uint32_t> bitmask(bitmask_size, 0);
      for (uint32_t idx = 0; idx != best_num_bins_left; ++idx) {
        uint32_t bin = split_manipulator.BinId(idx);
        uint32_t mask_idx = bin >> GetMaskIdx;
        uint32_t mask_shift = bin & GetMaskShift;
        bitmask[mask_idx] |= (1 << mask_shift);
      }
      node->Split()->UpdatePtr(ParallelTreeNodeNonMember::IsParallelNode(node), node->Stats()->Cost() - global_lowest_cost,
                               IsHighCardinality, feature_idx, bitmask);
    }
  }
};

using GiniSplitter = TreeNodeSplitter<GiniSplitManipulator>;
using EntropySplitter = TreeNodeSplitter<EntropySplitManipulator>;
using VarianceSplitter = TreeNodeSplitter<VarianceSplitManipulator>;

struct SplitterPtr {
  unique_ptr<GiniSplitter> gini_splitter;
  unique_ptr<EntropySplitter> entropy_splitter;
  unique_ptr<VarianceSplitter> variance_splitter;

  explicit SplitterPtr(uint32_t cost_function) {
    if (cost_function == GiniImpurity)
      gini_splitter = make_unique<GiniSplitter>();
    if (cost_function == Entropy)
      entropy_splitter = make_unique<EntropySplitter>();
    if (cost_function == Variance)
      variance_splitter = make_unique<VarianceSplitter>();
  }

  ~SplitterPtr() {
    gini_splitter.reset();
    entropy_splitter.reset();
    variance_splitter.reset();
  }
};
#endif
