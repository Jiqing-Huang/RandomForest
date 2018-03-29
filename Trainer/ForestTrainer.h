
#ifndef DECISIONTREE_FORESTBUILDER_H
#define DECISIONTREE_FORESTBUILDER_H

#include "../TreeBuilder/TreeBuilder.h"
#include "../Dataset/Dataset.h"
#include "../Dataset/IndexedFeature.h"
#include "TreeTrainer.h"

class ForestTrainer {
 public:
  ForestTrainer(uint32_t cost_function,
                uint32_t num_features_for_split,
                uint32_t min_leaf_node,
                uint32_t min_split_node,
                uint32_t max_depth,
                uint32_t max_num_nodes,
                uint32_t random_state,
                uint32_t num_threads,
                uint32_t num_trees):
    num_trees(num_trees), cost_function(cost_function),
    dataset(nullptr), presorted_indices(), total_sample_weights(), oob_count(), output_prob(), output_mean(),
    oob_output_prob(), oob_output_mean(), feature_importance(), feature_rank(), train_accuracy(0.0),
    train_loss(0.0), init_loss(0.0), final_loss(0.0), relative_loss_reduction(0.0), training_time(0.0),
    mean_depth(0.0), mean_num_cell(0.0), mean_num_leaf(0.0) {
    tree_trainers.reserve(num_trees);
    for (uint32_t tree_id = 0; tree_id != num_trees; ++tree_id)
      tree_trainers.emplace_back(std::make_unique<TreeTrainer>(cost_function, num_features_for_split, min_leaf_node,
                                                               min_split_node, max_depth, max_num_nodes,
                                                               random_state + tree_id, num_threads));
  };
  void LoadData(Dataset *dataset);
  void Train(bool to_report);
  void Predict();
  void Report();
  void Clear();

 private:
  uint32_t num_trees;
  const uint32_t cost_function;

  std::vector<std::unique_ptr<TreeTrainer>> tree_trainers;
  Dataset *dataset;
  vec_vec_uint32_t presorted_indices;
  vec_uint32_t total_sample_weights;
  vec_uint32_t oob_count;

  vec_vec_dbl_t output_prob;
  vec_dbl_t output_mean;

  vec_vec_dbl_t oob_output_prob;
  vec_dbl_t oob_output_mean;

  vec_dbl_t feature_importance;
  vec_uint32_t feature_rank;
  double train_accuracy;
  double train_loss;
  double oob_accuracy;
  double oob_loss;
  double init_loss;
  double final_loss;
  double relative_loss_reduction;

  double training_time;
  double mean_depth;
  double mean_num_cell;
  double mean_num_leaf;

  void Presort();

  template <typename feature_t>
  vec_uint32_t IndexSort(const std::vector<feature_t> &features);

  vec_uint32_t Bootstrap(uint32_t num_boot_samples);
  void Accumulate(uint32_t tree_id);
  void AccumulateClassification(uint32_t tree_id);
  void AccumulateRegression(uint32_t tree_id);
  void Reduce();
  void ReduceClassification();
  void ReduceRegression();
  void PredictClassification();
  void PredictRegression();
};

#endif
