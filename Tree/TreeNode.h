
#ifndef DECISIONTREE_TREENODE_H
#define DECISIONTREE_TREENODE_H

#include <cstdint>
#include <vector>
#include <memory>
#include <cassert>

#include "../Splitter/SplitInfo.h"
#include "TreeParams.h"
#include "NodeStats.h"
#include "../Dataset/Dataset.h"
#include "../Dataset/Subdataset.h"

class TreeNode {
 public:
  explicit TreeNode(const Dataset *dataset):
    type(IsRootType), depth(1), parent(nullptr), left(nullptr), right(nullptr), left_child_processed(false),
    right_child_processed(false), subset(std::make_unique<Subdataset>(dataset)), split_info(nullptr), stats(nullptr) {}

  void SetStats(const Dataset *dataset,
                const uint32_t cost_function) {
    stats = std::make_unique<NodeStats>();
    stats->SetStats(subset.get(), dataset, cost_function);
  }

  void InitSplitInfo() {
    split_info = std::make_unique<SplitInfo>();
  }

  void DiscardTemporaryElements() {
    subset->DiscardTemporaryElements();
  }

  void DiscardSortedIdx(uint32_t feature_idx) {
    subset->DiscardSortedIdx(feature_idx);
  }

  void DiscardSubset() {
    subset.reset();
  }

  bool IsRoot() const {
    return type == IsRootType;
  }

  bool IsLeftChild() const {
    return type == IsLeftChildType;
  }

  bool IsRightChild() const {
    return type == IsRightChildType;
  }

  uint32_t Size() const {
    return subset->Size();
  }

  uint32_t Depth() const {
    return depth;
  }

  TreeNode *Parent() const {
    return parent;
  }

  TreeNode *Left() const {
    return left.get();
  }

  TreeNode *Right() const {
    return right.get();
  }

  bool LeftChildProcessed() const {
    return left_child_processed;
  }

  bool RightChildProcessed() const {
    return right_child_processed;
  }

  void ProcessedLeft() {
    left_child_processed = true;
  }

  void ProcessedRight() {
    right_child_processed = true;
  }

  Subdataset *Subset() {
    return subset.get();
  }

  void SpawnChildren(const Dataset *dataset) {
    left.reset(new TreeNode(IsLeftChildType, this));
    right.reset(new TreeNode(IsRightChildType, this));
    subset->Partition(dataset->Features(split_info->feature_idx), split_info.get(), left->subset, right->subset);
  }

  SplitInfo *Split() const {
    return split_info.get();
  }

  NodeStats *Stats() const {
    return stats.get();
  }

 private:
  uint32_t type;
  uint32_t depth;
  TreeNode *parent;
  std::unique_ptr<TreeNode> left;
  std::unique_ptr<TreeNode> right;
  bool left_child_processed;
  bool right_child_processed;
  std::unique_ptr<Subdataset> subset;
  std::unique_ptr<SplitInfo> split_info;
  std::unique_ptr<NodeStats> stats;

  TreeNode(uint32_t type,
           TreeNode *parent):
    type(type), depth(parent->depth + 1), parent(parent), left(nullptr), right(nullptr), left_child_processed(false),
    right_child_processed(false), subset(nullptr), split_info(nullptr), stats(nullptr) {}
};
#endif