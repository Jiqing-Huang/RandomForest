
#ifndef DECISIONTREE_TREENODESPLITTER_H
#define DECISIONTREE_TREENODESPLITTER_H

#include <cstdint>
#include <vector>
#include <algorithm>

#include "SplitInfo.h"
#include "SplitManipulator.h"
#include "../Util/Maths.h"
#include "../Dataset/Dataset.h"
#include "../Tree/TreeNode.h"
#include "../Tree/ParallelTreeNode.h"
using std::vector;

template <typename data_t, typename CostComputer>
class TreeNodeSplitter {

 public:
  TreeNodeSplitter() = default;

  void Init(uint32_t num_threads,
            const Dataset &dataset,
            const TreeParams &params,
            Maths &util) {
    split_manipulator.reserve(num_threads);
    for (uint32_t idx = 0; idx != num_threads; ++idx)
      split_manipulator.emplace_back(dataset, params, util);
  }

  void CleanUp() {
    split_manipulator.clear();
    split_manipulator.shrink_to_fit();
  }

  void SplitNumerical(uint32_t feature_idx,
                      const Dataset &dataset,
                      const TreeParams &params,
                      const Maths &util,
                      TreeNode *node) {
    uint32_t thread_id = GetThreadId(feature_idx, node);
    split_manipulator[thread_id].NumericalInit(node);
    NumericalSplitter(feature_idx, split_manipulator[thread_id], node);
  }

  void SplitOrdinal(uint32_t feature_idx,
                    const Dataset &dataset,
                    const TreeParams &params,
                    const Maths &util,
                    TreeNode *node) {
    uint32_t thread_id = GetThreadId(feature_idx + dataset.num_numerical_features, node);
    split_manipulator[thread_id].DiscreteInit(feature_idx, node);
    OrdinalSplitter(feature_idx, split_manipulator[thread_id], node);
  }

  void SplitOneVsAll(uint32_t feature_idx,
                     const Dataset &dataset,
                     const TreeParams &params,
                     const Maths &util,
                     TreeNode *node) {
    uint32_t thread_id = GetThreadId(feature_idx + dataset.num_numerical_features, node);
    split_manipulator[thread_id].DiscreteInit(feature_idx, node);
    OneVsAllSplitter(feature_idx, split_manipulator[thread_id], node);
  }

  void SplitManyVsMany(uint32_t feature_idx,
                       const Dataset &dataset,
                       const TreeParams &params,
                       Maths &util,
                       TreeNode *node) {
    uint32_t thread_id = GetThreadId(feature_idx + dataset.num_numerical_features, node);
    split_manipulator[thread_id].DiscreteInit(feature_idx, node);
    if (dataset.num_classes == 2) {
      LinearSplitter(feature_idx, split_manipulator[thread_id], node);
    } else {
      ManyVsManySplitter(feature_idx, split_manipulator[thread_id], node);
    }
  }

 private:

  vector<SplitManipulator<data_t, CostComputer>> split_manipulator;

  uint32_t GetThreadId(uint32_t feature,
                       TreeNode *node) {
    if (!(node->type & (IsParallelBuilding | IsParallelSplitting)))
      return 0;
    auto *parallel_node = dynamic_cast<ParallelTreeNode *>(node);
    if (node->type & IsParallelBuilding)
      return parallel_node->thread_id_builder;
    if (node->type & IsParallelSplitting)
      return parallel_node->thread_id_splitter[feature];
  }

  void NumericalSplitter(uint32_t feature_idx,
                         SplitManipulator<data_t, CostComputer> &split_manipulator,
                         TreeNode *node);
  void OrdinalSplitter(uint32_t feature_idx,
                       SplitManipulator<data_t, CostComputer> &split_manipulator,
                       TreeNode *node);
  void OneVsAllSplitter(uint32_t feature_idx,
                        SplitManipulator<data_t, CostComputer> &split_manipulator,
                        TreeNode *node);
  void LinearSplitter(uint32_t feature_idx,
                      SplitManipulator<data_t, CostComputer> &split_manipulator,
                      TreeNode *node);
  void ManyVsManySplitter(uint32_t feature_idx,
                          SplitManipulator<data_t, CostComputer> &split_manipulator,
                          TreeNode *node);
  void BruteSplitter(uint32_t feature_idx,
                     SplitManipulator<data_t, CostComputer> &split_manipulator,
                     TreeNode *node);
  void GreedySplitter(uint32_t feature_idx,
                      SplitManipulator<data_t, CostComputer> &split_manipulator,
                      TreeNode *node);
};

#endif
