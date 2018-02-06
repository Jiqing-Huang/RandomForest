
#ifndef DECISIONTREE_TREETRAINER_H
#define DECISIONTREE_TREETRAINER_H

#include "../TreeBuilder/TreeBuilder.h"
#include "../TreeBuilder/ParallelTreeBuilder.h"
#include "../Predictor/TreePredictor.h"
#include "../Tree/StoredTree.h"
#include <iostream>

class TreeTrainer {
  friend class ForestTrainer;

 public:
  TreeTrainer(uint32_t cost_function,
              uint32_t num_features_for_split,
              uint32_t min_leaf_node,
              uint32_t min_split_node,
              uint32_t max_depth,
              uint32_t random_state,
              uint32_t num_threads);
  void LoadData(Dataset *dataset);
  void LoadSampleWeights(vec_uint32_t &&sample_weights);
  void LoadDefaultSampleWeights();
  void Train(bool to_report);
  void Report();
  void ClearOutput();
  void ClearBuilder();
  void ClearTree();

 private:
  std::unique_ptr<TreeBuilder> tree_builder;
  std::unique_ptr<TreePredictor> tree_predictor;
  std::unique_ptr<StoredTree> tree;
  Dataset *dataset;

  const uint32_t cost_function;

  vec_dbl_t feature_importance;
  vec_uint32_t feature_rank;
  double train_accuracy;
  double train_loss;
  double init_loss;
  double final_loss;
  double relative_loss_reduction;

  vec_vec_dbl_t output_prob;
  vec_dbl_t output_mean;

  vec_vec_dbl_t oob_output_prob;
  vec_dbl_t oob_output_mean;

  double training_time;

  void Predict(bool get_output,
               bool get_oob_pred);
  void PredictClassification(bool get_output,
                             bool get_oob_pred);
  void PredictRegression(bool get_output,
                         bool get_oob_pred);
};

#endif
