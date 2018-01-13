#include <iostream>
#include <vector>
#include <chrono>
#include <thread>

#include "TreeBuilder/ParallelTreeBuilder.h"
#include "Predictor/TreePredictor.h"
#include "Test/UnitTest.h"
#include "TreeBuilder/ForestBuilder.h"

using std::cout;
using std::endl;
using namespace std::chrono;

using std::vector;

int main() {
  // unit test
  UnitTest test;

  uint32_t num_samples = 1000000;
  uint32_t num_numerical_features = 120;
  uint32_t num_ordinal_features = 120;
  uint32_t num_one_vs_all_features = 120;
  uint32_t num_many_vs_many_features = 40;
  uint32_t min_ordinal_bin = 256;
  uint32_t max_ordinal_bin = 512;
  uint32_t min_one_vs_all_bin = 100;
  uint32_t max_one_vs_all_bin = 200;
  uint32_t min_many_vs_many_bin = 20;
  uint32_t max_many_vs_many_bin = 40;
  uint32_t num_classes = 4;

  test.RandomMixedDataset(num_samples,
                          num_numerical_features, num_ordinal_features,
                          num_one_vs_all_features, num_many_vs_many_features,
                          min_ordinal_bin, max_ordinal_bin,
                          min_one_vs_all_bin, max_one_vs_all_bin,
                          min_many_vs_many_bin, max_many_vs_many_bin,
                          num_classes);

  // Hyper parameters
  uint32_t cost_function = GiniCost;
  uint32_t num_features_for_split = 20;
  uint32_t min_leaf_node= 3;
  uint32_t min_split_node = 6;
  uint32_t max_depth = 0xffffffff;
  uint32_t random_state = 2;
  uint32_t num_trees = 20;
  uint32_t num_thread = 4;

  TreeBuilder ptb(cost_function, min_leaf_node, min_split_node, max_depth,
                  num_features_for_split, random_state);
  vector<uint32_t> sample_weights(num_samples, 1);
  test.dataset.LoadSampleWeights(sample_weights);
  ptb.LoadDataSet(test.dataset);
  StoredTree tree;
  auto begin = high_resolution_clock::now();
  ptb.Build(tree);
  auto end = high_resolution_clock::now();
  duration<double> elapsed= end - begin;

  uint32_t correct_count = 0;
  vector<uint32_t> prediction;
  prediction.resize(num_samples);
  TreePredictor tp;
  tp.BindToTree(tree);
  tp.PredictAllByMajority(&test.dataset, prediction);
  for (uint32_t idx = 0; idx < num_samples; ++idx)
    if (prediction[idx] == (*test.dataset.labels)[idx]) ++correct_count;
  double training_accuracy = static_cast<double>(correct_count) / static_cast<double>(num_samples);

  double numerical_feature_importance =
          accumulate(tree.numerical_feature_importance.begin(),
                     tree.numerical_feature_importance.end(), 0.0);
  double ordinal_feature_importance =
          accumulate(tree.discrete_feature_importance.begin(),
                     tree.discrete_feature_importance.begin() + num_ordinal_features, 0.0);
  double one_vs_all_feature_importance =
          accumulate(tree.discrete_feature_importance.begin() + num_ordinal_features,
                     tree.discrete_feature_importance.begin() + num_ordinal_features + num_one_vs_all_features, 0.0);
  double many_vs_many_feature_importance =
          accumulate(tree.discrete_feature_importance.begin() + num_ordinal_features + num_one_vs_all_features,
                     tree.discrete_feature_importance.end(), 0.0);

  cout << "Completed in " << elapsed.count() << " seconds." << endl;
  cout << "Depth: " << tree.max_depth << "; Num Cells: " << tree.num_cell << "; Num Leaves: " << tree.num_leaf << endl;
  cout << "training accuracy: " << training_accuracy << endl;
  cout << "   numerical feature importance: " << numerical_feature_importance << endl;
  cout << "   ordinal feature importance: " << ordinal_feature_importance << endl;
  cout << "   one_vs_all feature importance: " << one_vs_all_feature_importance << endl;
  cout << "   many_vs_many feature importance: " << many_vs_many_feature_importance << endl;


  /*
  ForestBuilder fb(cost_function, min_leaf_node, min_split_node, max_depth,
                   num_features_for_split, random_state, num_trees);

  fb.LoadDataSet(test.dataset);
  auto begin = high_resolution_clock::now();
  fb.Build();
  auto end = high_resolution_clock::now();
  duration<double> elapsed= end - begin;
  cout << "Completed in " << elapsed.count() << " seconds." << endl;

  uint32_t correct_count = 0;
  vector<vector<float>> training_prediction(num_samples, vector<float>(num_classes, 0.0));
  TreePredictor tp;
  for (const auto &tree: fb.forest) {
    tp.BindToTree(tree);
    tp.PredictAllByProbability(&test.dataset, training_prediction);
  }
  vector<uint32_t> training_vote;
  training_vote.reserve(num_samples);
  Maths util(0, 0);
  for (const auto &predict: training_prediction)
    training_vote.push_back(util.Argmax(predict, num_classes));
  for (uint32_t idx = 0; idx != num_samples; ++idx)
    if (training_vote[idx] == (*test.dataset.labels)[idx])
      ++correct_count;
  double training_accuracy = static_cast<double>(correct_count) / static_cast<double>(num_samples);
  cout << "training accuracy: " << training_accuracy << endl;

  correct_count = 0;
  for (uint32_t idx = 0; idx != num_samples; ++idx)
    if (fb.out_of_bag_vote[idx] == (*test.dataset.labels)[idx])
      ++correct_count;
  double out_of_bag_accuracy = static_cast<double>(correct_count) / static_cast<double>(num_samples);
  cout << "out of bag accuracy: " << out_of_bag_accuracy << endl;
  */
  return 0;

}
