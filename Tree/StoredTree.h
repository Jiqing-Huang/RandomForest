
#ifndef DECISIONTREE_STOREDTREE_H
#define DECISIONTREE_STOREDTREE_H

#include <vector>
#include <cstdint>
#include <atomic>
#include "ParallelTreeNode.h"

using std::vector;
using std::copy;

class StoredTree {
 public:
  union Info {
    float float_point;
    uint32_t integer;
    Info():
            integer(0) {};
  };

  std::atomic<uint32_t> num_cell;
  std::atomic<uint32_t> num_leaf;
  uint32_t max_depth;
  std::atomic<uint32_t> num_bitmask;

  vector<uint32_t> cell_type;
  vector<Info> cell_info;
  vector<int32_t> left;
  vector<int32_t> right;

  vector<vector<uint32_t>> bitmasks;

  vector<vector<double>> feature_importance_by_thread;
  vector<double> feature_importance;
  double total_gain;
  double init_loss;
  double final_loss;
  double relative_loss_reduction;

  StoredTree():
          num_cell(0), num_leaf(0), num_bitmask(0), max_depth(0), cell_type(), cell_info(), left(), right(),
          bitmasks(), feature_importance_by_thread(), feature_importance(), total_gain(0.0), final_loss(0.0),
          relative_loss_reduction(0.0) {};

  virtual ~StoredTree() = default;

  virtual void Init(const Dataset &dataset,
                    const TreeParams &params,
                    uint32_t num_threads) {
    feature_importance_by_thread.resize(num_threads, vector<double>(dataset.Meta().num_features, 0.0));
    feature_importance.resize(dataset.Meta().num_features, 0.0);
    cell_type.resize(params.max_num_cell);
    cell_info.resize(params.max_num_cell);
    left.resize(params.max_num_cell);
    right.resize(params.max_num_cell);
    bitmasks.resize(params.max_num_node);
  }

  virtual void CleanUp() {
    NormalizeFeatureImportance();
    cell_type.resize(num_cell);
    cell_info.resize(num_cell);
    left.resize(num_cell);
    right.resize(num_cell);
    bitmasks.resize(num_bitmask);
    cell_type.shrink_to_fit();
    cell_info.shrink_to_fit();
    left.shrink_to_fit();
    right.shrink_to_fit();
    bitmasks.shrink_to_fit();

    init_loss = final_loss + total_gain;
    relative_loss_reduction = 1.0 - final_loss / init_loss;
  }

  void UpdateFeatureImportance(TreeNode *node) {
    uint32_t thread_id = 0;
    auto *parallel_node = dynamic_cast<ParallelTreeNode *>(node);
    if (parallel_node && parallel_node->IsParallelBuilding())
      thread_id = parallel_node->BuilderThreadId();
    feature_importance_by_thread[thread_id][node->Split()->feature_idx] += node->Split()->gain;
  }

  virtual void WriteToLeaf(const TreeNode *node,
                           uint32_t leaf_id) = 0;

 private:
  void NormalizeFeatureImportance() {
    uint32_t size = static_cast<uint32_t>(feature_importance.size());
    for (const auto &feature_importance_each: feature_importance_by_thread)
      Maths::VectorAddInPlace(size, feature_importance_each, 0, feature_importance);
    total_gain = accumulate(feature_importance.begin(), feature_importance.end(), 0.0);
    if (total_gain > 0)
      for (auto &importance: feature_importance)
        importance /= total_gain;
  }
};

class ClassificationStoredTree: public StoredTree {
 public:
  vector<uint32_t> leaf_decision;
  vector<vector<float>> leaf_probability;

  ClassificationStoredTree():
          StoredTree::StoredTree(), leaf_decision(), leaf_probability() {}

  void Init(const Dataset &dataset,
            const TreeParams &params,
            uint32_t num_threads) override {
    StoredTree::Init(dataset, params, num_threads);
    leaf_decision.resize(params.max_num_leaf);
    leaf_probability.resize(params.max_num_leaf);
  }

  void CleanUp() override {
    StoredTree::CleanUp();
    leaf_decision.resize(num_leaf);
    leaf_probability.resize(num_leaf);
    leaf_decision.shrink_to_fit();
    leaf_probability.shrink_to_fit();
  }

  void WriteToLeaf(const TreeNode *node,
                   uint32_t leaf_id) override {
    Maths::CastAndCopyVisitor<float> visitor;
    leaf_probability[leaf_id] = boost::apply_visitor(visitor, node->Stats()->Histogram());
    Maths::Normalize(leaf_probability[leaf_id]);
    leaf_decision[leaf_id] = Maths::Argmax(leaf_probability[leaf_id]);
    final_loss += node->Stats()->Cost();
  }
};

class RegressionStoredTree: public StoredTree {
 public:
  vector<double> leaf_mean;

  RegressionStoredTree():
          StoredTree::StoredTree(), leaf_mean() {}

  void Init(const Dataset &dataset,
            const TreeParams &params,
            uint32_t num_threads) override {
    StoredTree::Init(dataset, params, num_threads);
    leaf_mean.resize(params.max_num_leaf);
  }

  void CleanUp() override {
    StoredTree::CleanUp();
    leaf_mean.resize(num_leaf);
    leaf_mean.shrink_to_fit();
  }

  void WriteToLeaf(const TreeNode *node,
                   uint32_t leaf_id) override {
    leaf_mean[leaf_id] = node->Stats()->Sum() / node->Stats()->NumSamples();
    final_loss += node->Stats()->Cost();
  }
};

#endif
