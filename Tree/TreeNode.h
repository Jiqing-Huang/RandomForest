
#ifndef DECISIONTREE_TREENODE_H
#define DECISIONTREE_TREENODE_H

#include <cstdint>
#include <vector>
#include <memory>
#include <cassert>

#include "../Splitter/SplitInfo.h"
#include "TreeParams.h"
#include "NodeStats.h"
#include "../Dataset/SubDataset.h"

using std::vector;
using std::move;
using std::unique_ptr;
using std::make_unique;

class TreeNode {
 public:
  TreeNode():
          node_id(0), type(0), depth(0), cell_id(0),
          parent(nullptr), left(nullptr), right(nullptr), left_child_processed(false), right_child_processed(false),
          subset(nullptr), split_info(nullptr), stats(nullptr) {}

  explicit TreeNode(const Dataset *dataset):
          node_id(0), type(IsRootType), depth(1), cell_id(0),
          parent(nullptr), left(nullptr), right(nullptr), left_child_processed(false), right_child_processed(false),
          subset(make_unique<SubDataset>(dataset)), split_info(nullptr), stats(nullptr) {}

  TreeNode(uint32_t node_id,
           uint32_t type,
           TreeNode *parent):
          node_id(node_id), type(type), depth(parent->depth + 1), cell_id(0),
          parent(parent), left(nullptr), right(nullptr), left_child_processed(false), right_child_processed(false),
          subset(nullptr), split_info(nullptr), stats(nullptr) {}

  virtual ~TreeNode() = default;

  void SetStats(const Dataset *dataset,
                const uint32_t cost_function) {
    stats = make_unique<NodeStats>();
    stats->SetStats(subset.get(), dataset, cost_function);
  }

  void InitSplitInfo() {
    split_info = make_unique<SplitInfo>();
  }

  void DiscardTemporaryElements() {
    subset->DiscardTemporaryElements();
    stats.reset();
    split_info.reset();
  }

  void DiscardSortedIdx(uint32_t feature_idx) {
    subset->DiscardSortedIdx(feature_idx);
  }

  void DiscardSubset() {
    subset.reset();
  }

  uint32_t NodeId() const {
    return node_id;
  }

  bool IsRoot() const {
    return (type & IsRootType) > 0;
  }

  bool IsLeftChild() const {
    return (type & IsLeftChildType) > 0;
  }

  bool IsRightChild() const {
    return (type & IsRightChildType) > 0;
  }

  uint32_t Size() const {
    return subset->Size();
  }

  uint32_t Depth() const {
    return depth;
  }

  uint32_t CellId() const {
    return cell_id;
  }

  void SetCellId(uint32_t cell_id) {
    this->cell_id = cell_id;
  }

  TreeNode *Parent() const {
    return parent;
  }

  TreeNode *Left() const {
    return left;
  }

  TreeNode *Right() const {
    return right;
  }

  void LinkChildren(TreeNode *left_child,
                    TreeNode *right_child) {
    left = left_child;
    right = right_child;
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

  SubDataset *Subset() {
    return subset.get();
  }

  void PartitionSubset(const Dataset *dataset,
                       TreeNode *left,
                       TreeNode *right) {
    subset->Partition(dataset->Features(split_info->feature_idx), split_info.get(), left->subset, right->subset);
  }

  SplitInfo *Split() {
    return split_info.get();
  }

  NodeStats *Stats() const {
    return stats.get();
  }

 private:
  uint32_t node_id;
  uint32_t type;
  uint32_t depth;
  uint32_t cell_id;
  TreeNode *parent;
  TreeNode *left;
  TreeNode *right;
  bool left_child_processed;
  bool right_child_processed;
  unique_ptr<SubDataset> subset;
  unique_ptr<SplitInfo> split_info;
  unique_ptr<NodeStats> stats;
};
#endif