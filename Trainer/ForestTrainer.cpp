
#include <algorithm>
#include <functional>
#include "ForestTrainer.h"
#include "../Predictor/TreePredictor.h"
#include "../Tree/StoredTree.h"
#include "../Util/Random.h"
#include "../Util/Maths.h"


void ForestTrainer::LoadData(Dataset *dataset) {
  this->dataset = dataset;
  total_sample_weights.resize(dataset->Meta().size, 0);
  oob_count.resize(dataset->Meta().size, 0);
  if (cost_function == GiniImpurity || cost_function == Entropy) {
    output_prob.resize(dataset->Meta().size, vec_dbl_t(dataset->Meta().num_classes, 0.0));
    oob_output_prob.resize(dataset->Meta().size, vec_dbl_t(dataset->Meta().num_classes, 0.0));
  } else {
    output_mean.resize(dataset->Meta().size, 0.0);
    oob_output_mean.resize(dataset->Meta().size, 0.0);
  }
  feature_importance.resize(dataset->Meta().num_features, 0.0);
}

void ForestTrainer::Train(bool to_report) {
  auto begin = std::chrono::high_resolution_clock::now();
  Presort();
  for (uint32_t tree_id = 0; tree_id < num_trees; ++tree_id) {
    if (tree_id % 10 == 0)
      std::cout << std::endl << "training tree: " << tree_id + 1;
    std::cout << "." << std::flush;
    TreeTrainer &trainer = *tree_trainers[tree_id];
    trainer.LoadData(dataset);
    vec_uint32_t sample_weights = Bootstrap(dataset->Meta().size);
    for (uint32_t idx = 0; idx != dataset->Meta().size; ++idx)
      if (sample_weights[idx] == 0) {
        ++oob_count[idx];
      } else {
        total_sample_weights[idx] += sample_weights[idx];
      }
    trainer.LoadSampleWeights(std::move(sample_weights));
    trainer.Train(false);
    trainer.Predict(false, true);
    Accumulate(tree_id);
    trainer.ClearOutput();
    trainer.ClearBuilder();
  }
  std::cout << std::endl;
  Reduce();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_duration = end - begin;
  training_time = time_duration.count();
  if (to_report) {
    Predict();
    Report();
  }
}

void ForestTrainer::Predict() {
  if (cost_function == GiniImpurity || cost_function == Entropy) {
    PredictClassification();
  } else {
    PredictRegression();
  }
}

void ForestTrainer::Report() {
  std::cout << "------------------------------" << std::endl;
  std::cout << "Training Time: " << training_time << " second(s)" << std::endl;
  std::cout << "------------------------------" << std::endl;
  std::cout << "Tree Description:" << std::endl;
  std::cout << "  Mean Depth: " << mean_depth << std::endl;
  std::cout << "  Mean Num Cells: " << mean_num_cell << std::endl;
  std::cout << "  Mean Num Leaves: " << mean_num_leaf << std::endl;
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
  std::cout << "Out of Bag Prediction: " << std::endl;
  if (cost_function == GiniImpurity || cost_function == Entropy) {
    std::cout << "  Out of Bag Accuracy: " << oob_accuracy << std::endl;
  } else if (cost_function == Variance) {
    std::cout << "  Out of Bag Loss: " << oob_loss << std::endl;
  }
  std::cout << "------------------------------" << std::endl;
  std::cout << "Top 10 Feature Importance: " << std::endl;
  for (uint32_t idx = 0; idx != 10; ++idx) {
    if (idx == 5) std::cout << std::endl;
    std::cout << "[" << feature_rank[idx] << ": " << feature_importance[feature_rank[idx]] << "]  ";
  }
  std::cout << std::endl << "------------------------------" << std::endl;
}

void ForestTrainer::Clear() {
  for (auto &trainer: tree_trainers)
    trainer.reset();
  presorted_indices.clear();
  presorted_indices.shrink_to_fit();
  total_sample_weights.clear();
  total_sample_weights.shrink_to_fit();
  oob_count.clear();
  oob_count.shrink_to_fit();
  output_prob.clear();
  output_prob.shrink_to_fit();
  output_mean.clear();
  output_mean.shrink_to_fit();
  oob_output_prob.clear();
  oob_output_prob.shrink_to_fit();
  oob_output_mean.clear();
  oob_output_mean.shrink_to_fit();
  feature_importance.clear();
  feature_importance.shrink_to_fit();
  feature_rank.clear();
  feature_rank.shrink_to_fit();
}

void ForestTrainer::Presort() {
  presorted_indices.resize(dataset->Meta().num_features);
  for (uint32_t idx = 0; idx != dataset->Meta().num_features; ++idx)
    if (dataset->FeatureType(idx) == IsContinuous)
      presorted_indices[idx] = boost::apply_visitor(
        [this](const auto &features) {
          return this->IndexSort(features);
        }, dataset->Features(idx));
}

template <typename feature_t>
vec_uint32_t ForestTrainer::IndexSort(const std::vector<feature_t> &features) {
  std::vector<IndexedFeature<feature_t>> indexed_features(features.size());
  for (uint32_t idx = 0; idx != features.size(); ++idx) {
    indexed_features[idx].feature = features[idx];
    indexed_features[idx].idx = idx;
  }
  std::sort(indexed_features.begin(), indexed_features.end());
  vec_uint32_t sorted_idx;
  sorted_idx.resize(features.size());
  uint32_t idx = 0;
  for (const auto &indexed_feature: indexed_features)
    sorted_idx[idx++] = indexed_feature.idx;
  return sorted_idx;
}

vec_uint32_t ForestTrainer::Bootstrap(uint32_t num_boot_samples) {
  vec_uint32_t sample_weights(num_boot_samples, 0);
  Random::SampleWithReplacement(num_boot_samples, num_boot_samples, sample_weights);
  return sample_weights;
}

void ForestTrainer::Accumulate(uint32_t tree_id) {
  if (cost_function == GiniImpurity || cost_function == Entropy) {
    AccumulateClassification(tree_id);
  } else {
    AccumulateRegression(tree_id);
  }
}

void ForestTrainer::AccumulateClassification(uint32_t tree_id) {
  TreeTrainer &trainer = *tree_trainers[tree_id];
  const vec_uint32_t &sample_weights = dataset->SampleWeights();
  for (uint32_t idx = 0; idx != dataset->Meta().size; ++idx) {
    uint32_t sample_weight = sample_weights[idx];
    if (sample_weight == 0) {
      std::transform(oob_output_prob[idx].begin(), oob_output_prob[idx].end(), trainer.oob_output_prob[idx].begin(),
                     oob_output_prob[idx].begin(), std::plus<>());
    } else {
      std::transform(output_prob[idx].begin(), output_prob[idx].end(),
                     trainer.output_prob[idx].begin(), output_prob[idx].begin(),
                     [&sample_weight](double &lhs, double &rhs) {
                       return lhs + sample_weight * rhs;
                     });
    }
  }
  std::transform(feature_importance.begin(), feature_importance.end(), trainer.feature_importance.begin(),
                 feature_importance.begin(), std::plus<>());
  trainer.ClearOutput();
}

void ForestTrainer::AccumulateRegression(uint32_t tree_id) {
  TreeTrainer &trainer = *tree_trainers[tree_id];
  const vec_uint32_t &sample_weights = dataset->SampleWeights();
  for (uint32_t idx = 0; idx != dataset->Meta().size; ++idx) {
    uint32_t sample_weight = sample_weights[idx];
    if (sample_weight == 0) {
      oob_output_mean[idx] += trainer.oob_output_mean[idx];
    } else {
      output_mean[idx] += sample_weight * trainer.output_mean[idx];
    }
  }
  std::transform(feature_importance.begin(), feature_importance.end(), trainer.feature_importance.begin(),
                 feature_importance.begin(), std::plus<>());
  trainer.ClearOutput();
}

void ForestTrainer::Reduce() {
  if (cost_function == GiniImpurity || cost_function == Entropy) {
    ReduceClassification();
  } else {
    ReduceRegression();
  }
  Maths::Normalize(feature_importance);
  feature_rank.resize(dataset->Meta().num_features, 0);
  std::iota(feature_rank.begin(), feature_rank.end(), 0);
  const vec_dbl_t &local_feature_importance = feature_importance;
  std::sort(feature_rank.begin(), feature_rank.end(),
            [&local_feature_importance](uint32_t x, uint32_t y) {
              return local_feature_importance[x] > local_feature_importance[y];
            });
  for (uint32_t tree_id = 0; tree_id != num_trees; ++tree_id) {
    init_loss += tree_trainers[tree_id]->init_loss;
    final_loss += tree_trainers[tree_id]->final_loss;
    mean_depth += tree_trainers[tree_id]->tree->max_depth;
    mean_num_cell += tree_trainers[tree_id]->tree->num_cell;
    mean_num_leaf += tree_trainers[tree_id]->tree->num_leaf;
  }
  init_loss /= num_trees;
  final_loss /= num_trees;
  relative_loss_reduction = 1 - final_loss / init_loss;
  mean_depth /= num_trees;
  mean_num_cell /= num_trees;
  mean_num_leaf /= num_trees;
}

void ForestTrainer::ReduceClassification() {
  for (auto &histogram: output_prob)
    Maths::Normalize(histogram);
  for (auto &histogram: oob_output_prob)
    Maths::Normalize(histogram);
}

void ForestTrainer::ReduceRegression() {
  for (uint32_t idx = 0; idx != dataset->Meta().size; ++idx) {
    if (total_sample_weights[idx] > 0)
      output_mean[idx] /= total_sample_weights[idx];
    if (oob_count[idx] > 0)
      oob_output_mean[idx] /= oob_count[idx];
  }
}

void ForestTrainer::PredictClassification() {
  uint32_t correct_count = 0;
  uint32_t total_count = 0;
  const auto &labels = dataset->Labels();
  for (uint32_t idx = 0; idx != dataset->Meta().size; ++idx) {
    if (total_sample_weights[idx] == 0) continue;
    total_count += total_sample_weights[idx];
    if (Maths::Argmax(output_prob[idx]) == Generics::RoundAt<uint32_t>(labels, idx))
      correct_count += total_sample_weights[idx];
  }
  train_accuracy = static_cast<double>(correct_count) / static_cast<double>(total_count);

  correct_count = 0;
  total_count = 0;
  for (uint32_t idx = 0; idx != dataset->Meta().size; ++idx) {
    if (oob_count[idx] == 0) continue;
    total_count += oob_count[idx];
    if (Maths::Argmax(oob_output_prob[idx]) == Generics::RoundAt<uint32_t>(labels, idx))
      correct_count += oob_count[idx];
  }
  oob_accuracy = static_cast<double>(correct_count) / static_cast<double>(total_count);
}

void ForestTrainer::PredictRegression() {
  const auto &labels = dataset->Labels();
  for (uint32_t idx = 0; idx != dataset->Meta().size; ++idx) {
    if (total_sample_weights[idx] == 0) continue;
    double diff = output_mean[idx] - Generics::RoundAt<double>(labels, idx);
    train_loss += total_sample_weights[idx] * diff * diff;
  }
  train_loss /= std::accumulate(total_sample_weights.begin(), total_sample_weights.end(), 0u);
  for (uint32_t idx = 0; idx != dataset->Meta().size; ++idx) {
    if (oob_count[idx] == 0) continue;
    double diff = oob_output_mean[idx] - Generics::RoundAt<double>(labels, idx);
    oob_loss += oob_count[idx] * diff * diff;
  }
  oob_loss /= std::accumulate(oob_count.begin(), oob_count.end(), 0u);
}