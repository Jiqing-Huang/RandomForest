
#ifndef DECISIONTREE_STOREDTREE_H
#define DECISIONTREE_STOREDTREE_H

#include <vector>
#include <cstdint>
#include <atomic>
#include "../Tree/TreeNode.h"
#include "../Tree/ParallelTreeNode.h"

using std::vector;

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

  vector<uint32_t> leaf_decision;
  vector<vector<float>> leaf_probability;

  vector<vector<double>> numerical_feature_importance_by_thread;
  vector<vector<double>> discrete_feature_importance_by_thread;

  vector<double> numerical_feature_importance;
  vector<double> discrete_feature_importance;

  StoredTree():
          num_cell(0), num_leaf(0), num_bitmask(0), max_depth(0), cell_type(), cell_info(), left(), right(),
          bitmasks(), leaf_decision(), leaf_probability(), numerical_feature_importance_by_thread(),
          discrete_feature_importance_by_thread(), numerical_feature_importance(), discrete_feature_importance() {};

  void Init(uint32_t num_numerical_feature,
            uint32_t num_discrete_feature,
            uint32_t num_thread) {
    numerical_feature_importance_by_thread.resize(num_thread, vector<double>(num_numerical_feature, 0.0));
    discrete_feature_importance_by_thread.resize(num_thread, vector<double>(num_discrete_feature, 0.0));
    numerical_feature_importance.resize(num_numerical_feature, 0.0);
    discrete_feature_importance.resize(num_discrete_feature, 0.0);
  }

  void UpdateFeatureImportance(TreeNode *node) {
    uint32_t thread_id = 0;
    if (node->type & IsParallelBuilding) {
      auto *parallel_node = dynamic_cast<ParallelTreeNode *>(node);
      thread_id = parallel_node->thread_id_builder;
    }
    if (node->split_info->type == IsContinuous) {
      numerical_feature_importance_by_thread[thread_id][node->split_info->feature_idx] += node->split_info->gain;
    } else {
      discrete_feature_importance_by_thread[thread_id][node->split_info->feature_idx] += node->split_info->gain;
    }
  }

  void NormalizeFeatureImportance(const Maths &util,
                                  uint32_t num_numerical_feature,
                                  uint32_t num_discrete_feature) {
    for (const auto &feature_importance: numerical_feature_importance_by_thread)
      util.VectorAddInPlace(num_numerical_feature, feature_importance, 0, numerical_feature_importance);
    for (const auto &feature_importance: discrete_feature_importance_by_thread)
      util.VectorAddInPlace(num_discrete_feature, feature_importance, 0, discrete_feature_importance);
    double numerical_sum = accumulate(numerical_feature_importance.begin(), numerical_feature_importance.end(), 0.0);
    double discrete_sum = accumulate(discrete_feature_importance.begin(), discrete_feature_importance.end(), 0.0);
    double sum = numerical_sum + discrete_sum;
    if (sum > 0) {
      for (auto &importance: numerical_feature_importance)
        importance /= sum;
      for (auto &importance: discrete_feature_importance)
        importance /= sum;
    }
  }

  void Resize(const uint32_t max_num_node,
              const uint32_t max_num_leaf) {
    cell_type.resize(max_num_node);
    cell_info.resize(max_num_node);
    left.resize(max_num_node);
    right.resize(max_num_node);
    leaf_decision.resize(max_num_leaf);
    leaf_probability.resize(max_num_leaf);
    bitmasks.resize(max_num_leaf);
  }

  void ShrinkToFit() {
    cell_type.resize(num_cell);
    cell_info.resize(num_cell);
    left.resize(num_cell);
    right.resize(num_cell);

    bitmasks.resize(num_bitmask);

    leaf_decision.resize(num_leaf);
    leaf_probability.resize(num_leaf);

    cell_type.shrink_to_fit();
    cell_info.shrink_to_fit();
    left.shrink_to_fit();
    right.shrink_to_fit();
    bitmasks.shrink_to_fit();
    leaf_decision.shrink_to_fit();
    leaf_probability.shrink_to_fit();
  }
};


#endif
