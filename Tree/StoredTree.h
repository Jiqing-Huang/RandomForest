
#ifndef DECISIONTREE_STOREDTREE_H
#define DECISIONTREE_STOREDTREE_H

#include <vector>
#include <cstdint>
#include "../Util/Maths.h"
#include "../Tree/TreeNode.h"

class StoredTree {
 public:
  union Info {
    float float_point;
    uint32_t integer;
    Info():
            integer(0) {};
  };

  uint32_t num_cell;
  uint32_t num_leaf;
  uint32_t max_depth;
  uint32_t num_bitmask;

  vec_uint32_t cell_type;
  std::vector<Info> cell_info;
  vec_int32_t left;
  vec_int32_t right;

  vec_vec_uint32_t bitmasks;

  vec_dbl_t feature_importance;

  double total_gain;
  double init_loss;
  double final_loss;
  double relative_loss_reduction;

  StoredTree():
          num_cell(0), num_leaf(0), num_bitmask(0), max_depth(0), cell_type(), cell_info(), left(), right(),
          bitmasks(), feature_importance(), total_gain(0.0), final_loss(0.0), relative_loss_reduction(0.0) {};

  virtual ~StoredTree() = default;

  virtual void Init(const Dataset &dataset,
                    const uint32_t num_cell,
                    const uint32_t num_leaf) {
    this->num_cell = num_cell;
    this->num_leaf = num_leaf;
    feature_importance.resize(dataset.Meta().num_features, 0.0);
    cell_type.resize(num_cell);
    cell_info.resize(num_cell);
    left.resize(num_cell);
    right.resize(num_cell);
    bitmasks.resize(num_cell);
  }

  void CleanUp() {
    bitmasks.resize(num_bitmask);
    bitmasks.shrink_to_fit();

    total_gain = accumulate(feature_importance.begin(), feature_importance.end(), 0.0);
    if (total_gain > 0)
      for (auto &importance: feature_importance)
        importance /= total_gain;
    init_loss = final_loss + total_gain;
    relative_loss_reduction = 1.0 - final_loss / init_loss;
  }

  void WriteToCell(const TreeNode *node,
                   const int32_t cell_id,
                   const int32_t parent_id) {
    cell_type[cell_id] = node->Split()->type | node->Split()->feature_idx;
    feature_importance[node->Split()->feature_idx] += node->Split()->gain;
    switch (node->Split()->type) {
      case IsContinuous:
        cell_info[cell_id].float_point = node->Split()->info.float_type;
        break;
      case IsOrdinal:
        cell_info[cell_id].integer = node->Split()->info.uint32_type;
        break;
      case IsOneVsAll:
        cell_info[cell_id].integer = node->Split()->info.uint32_type;
        break;
      case IsLowCardinality:
        cell_info[cell_id].integer = node->Split()->info.uint32_type;
        break;
      case IsHighCardinality:
        cell_info[cell_id].integer = num_bitmask;
        bitmasks[num_bitmask++] = *node->Split()->info.ptr_type;
        break;
      default:
        break;
    }
    if (node->IsLeftChild()) left[parent_id] = cell_id;
    if (node->IsRightChild()) right[parent_id] = cell_id;
  }

  virtual void WriteToLeaf(const TreeNode *node,
                           const int32_t leaf_id,
                           const int32_t parent_id) {
    if (node->IsLeftChild()) left[parent_id] = -leaf_id;
    if (node->IsRightChild()) right[parent_id] = -leaf_id;
    final_loss += node->Stats()->Cost();
    max_depth = (max_depth > node->Depth())? max_depth : node->Depth();
  }
};

class ClassificationStoredTree: public StoredTree {
 public:
  vec_vec_dbl_t leaf_probability;

  ClassificationStoredTree():
          StoredTree::StoredTree(), leaf_probability() {}

  void Init(const Dataset &dataset,
            const uint32_t num_cell,
            const uint32_t num_leaf) override {
    StoredTree::Init(dataset, num_cell, num_leaf);
    leaf_probability.resize(num_leaf);
  }

  void WriteToLeaf(const TreeNode *node,
                   const int32_t leaf_id,
                   const int32_t parent_id) override {
    StoredTree::WriteToLeaf(node, leaf_id, parent_id);
    Maths::CastAndCopyVisitor<float> visitor;
    leaf_probability[leaf_id] = node->Stats()->Histogram();
    Maths::Normalize(leaf_probability[leaf_id]);
  }
};

class RegressionStoredTree: public StoredTree {
 public:
  vec_dbl_t leaf_mean;

  RegressionStoredTree():
          StoredTree::StoredTree(), leaf_mean() {}

  void Init(const Dataset &dataset,
            const uint32_t num_cell,
            const uint32_t num_leaf) override {
    StoredTree::Init(dataset, num_cell, num_leaf);
    leaf_mean.resize(num_leaf);
  }

  void WriteToLeaf(const TreeNode *node,
                   const int32_t leaf_id,
                   const int32_t parent_id) override {
    StoredTree::WriteToLeaf(node, leaf_id, parent_id);
    leaf_mean[leaf_id] = node->Stats()->Sum() / node->Stats()->NumSamples();
  }
};

#endif
