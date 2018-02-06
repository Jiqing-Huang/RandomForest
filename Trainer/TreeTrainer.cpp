
#include "TreeTrainer.h"
#include "../TreeBuilder/TreeBuilder.h"
#include "../TreeBuilder/ParallelTreeBuilder.h"
#include "../Predictor/TreePredictor.h"
#include "../Predictor/ParallelTreePredictor.h"
#include "../Tree/StoredTree.h"
#include <iostream>

TreeTrainer::TreeTrainer(uint32_t cost_function,
                         uint32_t num_features_for_split,
                         uint32_t min_leaf_node,
                         uint32_t min_split_node,
                         uint32_t max_depth,
                         uint32_t random_state,
                         uint32_t num_threads):
  dataset(nullptr), cost_function(cost_function), train_accuracy(0.0), train_loss(0.0),
  init_loss(0.0), final_loss(0.0), relative_loss_reduction(0.0), feature_importance(), training_time(0.0) {
  if (num_threads == 1) {
    tree_builder = std::make_unique<TreeBuilder>(cost_function, min_leaf_node, min_split_node, max_depth,
                                                 num_features_for_split, random_state);
    tree_predictor = std::make_unique<TreePredictor>();
  } else {
    tree_builder = std::make_unique<ParallelTreeBuilder>(cost_function, min_leaf_node, min_split_node, max_depth,
                                                         num_features_for_split, random_state, num_threads);
    tree_predictor = std::make_unique<ParallelTreePredictor>(num_threads);
  }
  if (cost_function == GiniImpurity || cost_function == Entropy)
    tree = std::make_unique<ClassificationStoredTree>();
  if (cost_function == Variance)
    tree = std::make_unique<RegressionStoredTree>();
}

void TreeTrainer::LoadData(Dataset *dataset) {
  this->dataset = dataset;
}

void TreeTrainer::LoadSampleWeights(vec_uint32_t &&sample_weights) {
  dataset->AddSampleWeights(std::move(sample_weights));
}

void TreeTrainer::LoadDefaultSampleWeights() {
  vec_uint32_t default_sample_weights(dataset->Meta().size, 1);
  dataset->AddSampleWeights(std::move(default_sample_weights));
}

void TreeTrainer::Train(bool to_report = true) {
  auto begin = std::chrono::high_resolution_clock::now();
  tree_builder->LoadDataSet(*dataset);
  tree_builder->Build(*tree);

  init_loss = tree->init_loss;
  final_loss = tree->final_loss;
  relative_loss_reduction = tree->relative_loss_reduction;

  feature_importance = tree->feature_importance;
  feature_rank.resize(feature_importance.size(), 0);
  iota(feature_rank.begin(), feature_rank.end(), 0);
  const vec_dbl_t &local_feature_importance = feature_importance;
  std::sort(feature_rank.begin(), feature_rank.end(),
            [&local_feature_importance](uint32_t x, uint32_t y) {
              return local_feature_importance[x] > local_feature_importance[y];
            });
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_duration = end - begin;
  training_time = time_duration.count();

  if (to_report) {
    Predict(true, false);
    Report();
  }
}

void TreeTrainer::Predict(bool get_output,
                          bool get_oob_pred) {
  if (cost_function == GiniImpurity || cost_function == Entropy) {
    PredictClassification(get_output, get_oob_pred);
  } else {
    PredictRegression(get_output, get_oob_pred);
  }
}

void TreeTrainer::Report() {
  std::cout << "------------------------------" << std::endl;
  std::cout << "Training Time: " << training_time << " second(s)" << std::endl;
  std::cout << "------------------------------" << std::endl;
  std::cout << "Tree Description:" << std::endl;
  std::cout << "  Depth: " << tree->max_depth << std::endl;
  std::cout << "  Num Cells: " << tree->num_cell << std::endl;
  std::cout << "  Num Leaves: " << tree->num_leaf << std::endl;
  std::cout << "------------------------------" << std::endl;
  std::cout << "Loss as ";
  if (cost_function == GiniImpurity) {
    std::cout << "[Gini Impurity]";
  } else if (cost_function == Entropy) {
    std::cout << "[Entropy]";
  } else if (cost_function == Variance) {
    std::cout << "[Variance]";
  }
  std::cout << ":" << std::endl;
  std::cout << "  Initial Loss: " << init_loss << std::endl;
  std::cout << "  Final Loss: " << final_loss << std::endl;
  std::cout << "  Relative Loss Reduction: " << relative_loss_reduction << std::endl;
  std::cout << "------------------------------" << std::endl;
  std::cout << "Training Set Prediction: " << std::endl;
  if (cost_function == GiniImpurity || cost_function == Entropy) {
    std::cout << "  Training Accuracy: " << train_accuracy << std::endl;
  } else if (cost_function == Variance) {
    std::cout << "  Training Loss: " << train_loss << std::endl;
  }
  std::cout << "------------------------------" << std::endl;
  std::cout << "Top 10 Feature Importance: " << std::endl;
  for (uint32_t idx = 0; idx != 10; ++idx) {
    if (idx == 5) std::cout << std::endl;
    std::cout << "[" << feature_rank[idx] << ": " << feature_importance[feature_rank[idx]] << "]  ";
  }
  std::cout << std::endl << "------------------------------" << std::endl;
}

void TreeTrainer::ClearOutput() {
  output_prob.clear();
  output_prob.shrink_to_fit();
  output_mean.clear();
  output_mean.shrink_to_fit();
  oob_output_prob.clear();
  oob_output_prob.shrink_to_fit();
  oob_output_mean.clear();
  oob_output_mean.shrink_to_fit();
}

void TreeTrainer::ClearBuilder() {
  tree_builder.reset();
  tree_predictor.reset();
}

void TreeTrainer::ClearTree() {
  tree.reset();
}

void TreeTrainer::PredictClassification(bool get_output,
                                        bool get_oob_pred) {
  const auto *class_tree = dynamic_cast<const ClassificationStoredTree*>(tree.get());
  tree_predictor->BindToTree(*class_tree);
  output_prob = tree_predictor->PredictBatchByProbability(dataset, PredictPresent);
  if (get_oob_pred)
    oob_output_prob = tree_predictor->PredictBatchByProbability(dataset, PredictAbsent);
  if (get_output) {
    uint32_t correct_count = 0;
    uint32_t total_count = 0;
    const auto &labels = dataset->Labels();
    const auto &sample_weights = dataset->SampleWeights();
    for (uint32_t idx = 0; idx != dataset->Meta().size; ++idx) {
      if (sample_weights[idx] == 0) continue;
      total_count += sample_weights[idx];
      if (Maths::Argmax(output_prob[idx]) == Generics::RoundAt<uint32_t>(labels, idx))
        correct_count += sample_weights[idx];
    }
    train_accuracy = static_cast<double>(correct_count) / static_cast<double>(total_count);
  }
}

void TreeTrainer::PredictRegression(bool get_output,
                                    bool get_oob_pred) {
  const auto *regress_tree = dynamic_cast<const RegressionStoredTree*>(tree.get());
  tree_predictor->BindToTree(*regress_tree);
  output_mean = tree_predictor->PredictBatchByMean(dataset, PredictPresent);
  if (get_oob_pred)
    oob_output_mean = tree_predictor->PredictBatchByMean(dataset, PredictAbsent);
  if (get_output) {
    const auto &labels = dataset->Labels();
    const auto &sample_weights = dataset->SampleWeights();
    for (uint32_t idx = 0; idx != dataset->Meta().size; ++idx) {
      double diff = output_mean[idx] - Generics::RoundAt<double>(labels, idx);
      train_loss += sample_weights[idx] * diff * diff;
    }
    train_loss /= Generics::Round<double>(dataset->Meta().num_samples);
  }
}