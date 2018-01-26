
#include <algorithm>
#include <functional>
#include "ForestBuilder.h"
#include "../Predictor/TreePredictor.h"
#include "../Util/Random.h"

using std::sort;
using std::fill;
using std::transform;

void ForestBuilder::LoadDataSet(Dataset *dataset) {
  this->dataset = dataset;
}

void ForestBuilder::UnloadDataSet() {
  dataset = nullptr;
}

void ForestBuilder::Presort() {
  presorted_indices.resize(dataset->Meta().num_features);
  IndexSortVisitor visitor(this);
  for (uint32_t idx = 0; idx != dataset->Meta().num_features; ++idx)
    if (dataset->FeatureType(idx) == IsContinuous)
      presorted_indices[idx] = boost::apply_visitor(visitor, dataset->Features(idx));
}

void ForestBuilder::Bootstrap(uint32_t num_boot_samples,
                              vector<uint32_t> &sample_weights) {
  fill(sample_weights.begin(), sample_weights.end(), 0);
  Random::SampleWithReplacement(dataset->Meta().size, num_boot_samples, sample_weights);
}

void ForestBuilder::Build() {
  Presort();
  TreeBuilder tree_builder(params);
  tree_builder.LoadDataSet(*dataset);
  tree_builder.LoadPresortedIndices(presorted_indices);
  vector<uint32_t> sample_weights(dataset->Meta().size, 0);

  out_of_bag_prediction.assign(dataset->Meta().size, vector<float>(dataset->Meta().num_classes, 0.0));
  TreePredictor tree_predictor;

  for (uint32_t tree_id = 0; tree_id < params.num_trees; ++tree_id) {
    Bootstrap(dataset->Meta().size, sample_weights);
    dataset->AddSampleWeights(sample_weights);
    tree_builder.UpdateDataSet();
    forest[tree_id] = make_unique<ClassificationStoredTree>();
    tree_builder.Build(*forest[tree_id]);
    tree_predictor.BindToTree(*dynamic_cast<ClassificationStoredTree*>(forest[tree_id].get()));
    auto prediction = tree_predictor.PredictSelectedByProbability(dataset, dataset->SampleWeights(), PredictAbsent);
    for (uint32_t idx = 0; idx != dataset->Meta().size; ++idx)
      transform(out_of_bag_prediction[idx].begin(), out_of_bag_prediction[idx].end(),
                prediction[idx].begin(), out_of_bag_prediction[idx].begin(), std::plus<float>());
  }

  out_of_bag_vote.reserve(dataset->Meta().size);
  for (const auto &prediction: out_of_bag_prediction)
    out_of_bag_vote.push_back(Maths::Argmax(prediction));

  tree_builder.UnloadDataSet();
  tree_builder.UnloadPresortedIndices();
}