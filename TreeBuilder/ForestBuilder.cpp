
#include "ForestBuilder.h"
#include "../Predictor/TreePredictor.h"

using std::sort;
using std::fill;

void ForestBuilder::LoadDataSet(Dataset &dataset) {
  this->dataset = &dataset;
}

void ForestBuilder::UnloadDataSet() {
  dataset = nullptr;
}

void ForestBuilder::Presort() {
  vector<uint32_t> sample_ids;
  sample_ids.reserve(dataset->num_samples);
  for (uint32_t idx = 0; idx != dataset->num_samples; ++idx)
    sample_ids.push_back(idx);
  presorted_indices.reserve(dataset->num_numerical_features);
  for (uint32_t idx = 0; idx != dataset->num_numerical_features; ++idx) {
    presorted_indices.emplace_back(sample_ids);
    const vector<float> &feature = (*dataset->numerical_features)[idx];
    sort(presorted_indices[idx].begin(), presorted_indices[idx].end(),
         [&feature](uint32_t x, uint32_t y) {
           return feature[x] < feature[y];
         });
  }
}

void ForestBuilder::Bootstrap(uint32_t num_boot_samples,
                              vector<uint32_t> &sample_weights) {
  fill(sample_weights.begin(), sample_weights.end(), 0);
  util.SampleWithReplacement(dataset->num_samples, num_boot_samples, sample_weights);
}

void ForestBuilder::Build() {
  Presort();
  TreeBuilder tree_builder(params);
  tree_builder.LoadDataSet(*dataset);
  tree_builder.LoadPresortedIndices(presorted_indices);
  vector<uint32_t> sample_weights(dataset->num_samples, 0);

  out_of_bag_prediction.assign(dataset->num_samples, vector<float>(dataset->num_classes, 0.0));
  TreePredictor tree_predictor;

  for (uint32_t tree_id = 0; tree_id < params.num_trees; ++tree_id) {
    Bootstrap(dataset->num_samples, sample_weights);
    dataset->LoadSampleWeights(sample_weights);

    tree_builder.UpdateDataSet();
    tree_builder.Build(forest[tree_id]);

    tree_predictor.BindToTree(forest[tree_id]);
    tree_predictor.PredictAbsentByProbability(dataset, sample_weights, out_of_bag_prediction);
  }

  out_of_bag_vote.reserve(dataset->num_samples);
  for (const auto &prediction: out_of_bag_prediction)
    out_of_bag_vote.push_back(util.Argmax(prediction, dataset->num_classes));

  tree_builder.UnloadDataSet();
  tree_builder.UnloadPresortedIndices();
}