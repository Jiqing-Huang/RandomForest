
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
  uint32_t node_id;
  uint32_t type;
  uint32_t depth;
  uint32_t cell_id;
  uint32_t size;
  TreeNode *parent;
  TreeNode *left;
  TreeNode *right;
  bool left_child_processed;
  bool right_child_processed;

  unique_ptr<SubDataset> subset;
  unique_ptr<SplitInfo> split_info;
  unique_ptr<NodeStats> stats;

  TreeNode():
          node_id(0), type(0), depth(0), cell_id(0), size(0),
          parent(nullptr), left(nullptr), right(nullptr), left_child_processed(false), right_child_processed(false),
          subset(nullptr), split_info(nullptr), stats(nullptr) {}

  TreeNode(const Dataset &dataset,
           const Maths &util):
          node_id(0), type(IsRoot), depth(1), cell_id(0),
          parent(nullptr), left(nullptr), right(nullptr), left_child_processed(false), right_child_processed(false),
          subset(new SubDataset(dataset, util)), split_info(nullptr), stats(nullptr) {
    size = static_cast<uint32_t>(subset->sample_ids.size());
  }

  TreeNode(uint32_t node_id,
           uint32_t type,
           uint32_t depth,
           TreeNode *parent,
           const Dataset &dataset,
           vector<uint32_t> &&sample_ids,
           vector<uint32_t> &&labels,
           vector<uint32_t> &&sample_weights):
          node_id(node_id), type(type), depth(depth), cell_id(0),
          parent(parent), left(nullptr), right(nullptr), left_child_processed(false), right_child_processed(false),
          subset(new SubDataset(dataset.num_numerical_features, dataset.num_discrete_features,
                                move(sample_ids), move(labels), move(sample_weights))),
          split_info(nullptr), stats(nullptr) {
    size = static_cast<uint32_t>(subset->sample_ids.size());
  }

  TreeNode(TreeNode &&node) noexcept:
          node_id(node.node_id), type(node.type), depth(node.depth), cell_id(node.cell_id),
          size(node.size), parent(parent), left(left), right(right),
          left_child_processed(node.left_child_processed), right_child_processed(node.right_child_processed) {
    subset = move(node.subset);
    split_info = move(node.split_info);
    stats = move(node.stats);
  }

  virtual ~TreeNode() = default;

  TreeNode &operator=(TreeNode &&node) noexcept {
    assert(this != &node);
    node_id = node.node_id;
    type = node.type;
    depth = node.depth;
    cell_id = node.cell_id;
    size = node.size;
    parent = node.parent;
    left = node.left;
    right = node.right;
    left_child_processed = node.left_child_processed;
    right_child_processed = node.right_child_processed;
    subset = move(node.subset);
    split_info = move(node.split_info);
    stats = move(node.stats);
  }

  void GetStats(const Dataset &dataset,
                const TreeParams &params,
                const Maths &util) {
    stats = make_unique<NodeStats>(dataset.num_classes, params.cost_function);
    stats->GetStats(params, dataset, subset->sample_ids, util);
  }

  void GetEmptySplitInfo() {
    split_info = make_unique<SplitInfo>();
  }

  void DiscardTemporaryElements() {
    subset->DiscardSubsetFeature();
    stats.reset();
    split_info.reset();
  }

  void DiscardSortedIdx(uint32_t feature_idx) {
    subset->DiscardSortedIdx(feature_idx);
  }

  void DiscardSubset() {
    subset.reset();
  }
};

#endif
