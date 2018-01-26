
#ifndef DECISIONTREE_TREETRAINER_H
#define DECISIONTREE_TREETRAINER_H

#include "../TreeBuilder/TreeBuilder.h"
#include "../TreeBuilder/ParallelTreeBuilder.h"
#include "../Predictor/TreePredictor.h"
#include <iostream>

using std::cout;
using std::endl;

class TreeTrainer {
 public:
  TreeTrainer(uint32_t cost_function,
              uint32_t num_features_for_split,
              uint32_t min_leaf_node,
              uint32_t min_split_node,
              uint32_t max_depth,
              uint32_t random_state,
              uint32_t num_threads = 1):
          tree_predictor(), dataset(nullptr), cost_function(cost_function), train_accuracy(0.0), train_loss(0.0),
          init_loss(0.0), final_loss(0.0), relative_loss_reduction(0.0), feature_importance(), training_time(0.0) {
    if (num_threads == 1) {
      tree_builder = make_unique<TreeBuilder>(cost_function, min_leaf_node, min_split_node, max_depth,
                                              num_features_for_split, random_state);
    } else {
      tree_builder = make_unique<ParallelTreeBuilder>(cost_function, min_leaf_node, min_split_node, max_depth,
                                                      num_features_for_split, random_state, num_threads);
    }
    if (cost_function == GiniImpurity || cost_function == Entropy)
      tree = make_unique<ClassificationStoredTree>();
    if (cost_function == Variance)
      tree = make_unique<RegressionStoredTree>();
  }

  void LoadData(Dataset *dataset) {
    this->dataset = dataset;
  }

  void LoadDefaultSampleWeights(uint32_t size) {
    vec_uint32_t default_sample_weights(size, 1);
    dataset->AddSampleWeights(default_sample_weights);
  }

  void Train(bool predict_on_training_set = false) {
    tree_builder->LoadDataSet(*dataset);
    auto begin = std::chrono::high_resolution_clock::now();
    tree_builder->Build(*tree);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_duration = end - begin;
    training_time = time_duration.count();

    init_loss = tree->init_loss;
    final_loss = tree->final_loss;
    relative_loss_reduction = tree->relative_loss_reduction;

    feature_importance = tree->feature_importance;
    feature_rank.resize(feature_importance.size(), 0);
    iota(feature_rank.begin(), feature_rank.end(), 0);
    const vector<double> &local_feature_importance = feature_importance;
    sort(feature_rank.begin(), feature_rank.end(),
         [&local_feature_importance](uint32_t x, uint32_t y) {
           return local_feature_importance[x] > local_feature_importance[y];
         });

    if (predict_on_training_set) {
      if (cost_function == GiniImpurity || cost_function == Entropy) {
        PredictClassification();
      } else {
        PredictRegression();
      }
    }
  }

  void PredictClassification() {
    const auto *class_tree = dynamic_cast<const ClassificationStoredTree*>(tree.get());
    tree_predictor.BindToTree(*class_tree);

    prediction_majority = tree_predictor.PredictAllByMajority(dataset);

    uint32_t correct_count = 0;
    const auto &labels = boost::get<vec_uint32_t>(dataset->Labels());
    const auto &sample_weights = dataset->SampleWeights();
    for (uint32_t idx = 0; idx != dataset->Meta().size; ++idx)
      if (prediction_majority[idx] == labels[idx])
        correct_count += sample_weights[idx];
    train_accuracy = static_cast<double>(correct_count) / static_cast<double>(dataset->Meta().size);
  }

  void PredictRegression() {
    const auto *regress_tree = dynamic_cast<const RegressionStoredTree*>(tree.get());
    tree_predictor.BindToTree(*regress_tree);

    prediction_mean = tree_predictor.PredictAllByMean(dataset);

    const auto &labels = boost::get<vec_dbl_t>(dataset->Labels());
    const auto &sample_weights = dataset->SampleWeights();
    for (uint32_t idx = 0; idx != dataset->Meta().size; ++idx) {
      double diff = prediction_mean[idx] - labels[idx];
      train_loss += sample_weights[idx] * diff * diff;
    }
    train_loss /= Generics::Round<double>(dataset->Meta().num_samples);
  }

  void Report() {
    cout << "------------------------------" << endl;
    cout << "Training Time: " << training_time << " second(s)" << endl;
    cout << "------------------------------" << endl;
    cout << "Tree Description:" << endl;
    cout << "  Depth: " << tree->max_depth << endl;
    cout << "  Num Cells: " << tree->num_cell << endl;
    cout << "  Num Leaves: " << tree->num_leaf << endl;
    cout << "------------------------------" << endl;
    cout << "Loss as ";
    if (cost_function == GiniImpurity) {
      cout << "[Gini Impurity]";
    } else if (cost_function == Entropy) {
      cout << "[Entropy]";
    } else if (cost_function == Variance) {
      cout << "[Variance]";
    }
    cout << ":" << endl;
    cout << "  Initial Loss: " << init_loss << endl;
    cout << "  Final Loss: " << final_loss << endl;
    cout << "  Relative Loss Reduction: " << relative_loss_reduction << endl;
    cout << "------------------------------" << endl;
    cout << "Training Set Prediction: " << endl;
    if (cost_function == GiniImpurity || cost_function == Entropy) {
      cout << "  Training Accuracy: " << train_accuracy << endl;
    } else if (cost_function == Variance) {
      cout << "  Training Loss: " << train_loss << endl;
    }
    cout << "------------------------------" << endl;
    cout << "Top 10 Feature Importance: " << endl;
    for (uint32_t idx = 0; idx != 10; ++idx) {
      if (idx == 5) cout << endl;
      cout << "[" << feature_rank[idx] << ": " << feature_importance[feature_rank[idx]] << "]  ";
    }
    cout << endl << "------------------------------" << endl;
  }

 private:
  unique_ptr<TreeBuilder> tree_builder;
  TreePredictor tree_predictor;
  unique_ptr<StoredTree> tree;
  Dataset *dataset;

  const uint32_t cost_function;

  vector<double> feature_importance;
  vector<uint32_t> feature_rank;
  double train_accuracy;
  double train_loss;
  double init_loss;
  double final_loss;
  double relative_loss_reduction;

  vector<uint32_t> prediction_majority;
  vector<double> prediction_mean;

  double training_time;
};

#endif
