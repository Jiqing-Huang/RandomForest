
#ifndef DECISIONTREE_SUBDATASET_H
#define DECISIONTREE_SUBDATASET_H

#include <vector>
#include <cstdint>
#include "Dataset.h"
#include "../Util/Maths.h"

using std::vector;

class SubDataset {
 public:

  // temporary elements
  vector<uint32_t> labels;
  vector<uint32_t> sample_weights;

  // persistent elements
  vector<uint32_t> sample_ids;
  vector<vector<uint32_t>> sorted_indices;
  vector<vector<float>> numerical_features;
  vector<vector<uint32_t>> discrete_features;

  SubDataset(const Dataset &dataset,
             const Maths &util):
          sorted_indices(dataset.num_numerical_features), numerical_features(dataset.num_numerical_features),
          discrete_features(dataset.num_discrete_features),
          labels(*dataset.labels), sample_weights(*dataset.sample_weights) {
    util.SequentialFillNonzero(dataset.num_samples, *dataset.sample_weights, sample_ids);
  }

  SubDataset(uint32_t num_numerical_features,
             uint32_t num_discrete_features,
             vector<uint32_t> &&sample_ids,
             vector<uint32_t> &&labels,
             vector<uint32_t> &&sample_weights):
          sorted_indices(num_numerical_features), numerical_features(num_numerical_features),
          discrete_features(num_discrete_features), sample_ids(move(sample_ids)), labels(move(labels)),
          sample_weights(move(sample_weights)) {};

  SubDataset(const SubDataset &set) = default;

  SubDataset(SubDataset &&set) noexcept:
          labels(move(set.labels)), sample_weights(move(set.sample_weights)), sample_ids(move(set.sample_ids)),
          sorted_indices(move(set.sorted_indices)), numerical_features(move(set.numerical_features)),
          discrete_features(move(set.discrete_features)) {};

  ~SubDataset() = default;

  SubDataset &operator=(const SubDataset &set) {
    labels = set.labels;
    sample_weights = set.sample_weights;
    sample_ids = set.sample_ids;
    sorted_indices = set.sorted_indices;
    numerical_features = set.numerical_features;
    discrete_features = set.discrete_features;
  }

  SubDataset &operator=(SubDataset &&set) noexcept {
    assert(this != &set);
    labels = move(set.labels);
    sample_weights = move(set.sample_weights);
    sample_ids = move(set.sample_ids);
    sorted_indices = move(set.sorted_indices);
    numerical_features = move(set.numerical_features);
    discrete_features = move(set.discrete_features);
  }

  void DiscardSubsetFeature() {
    labels.clear();
    labels.shrink_to_fit();
    sample_weights.clear();
    sample_weights.shrink_to_fit();
    numerical_features.clear();
    numerical_features.shrink_to_fit();
    discrete_features.clear();
    discrete_features.shrink_to_fit();
  }

  void DiscardSortedIdx(uint32_t feature_id) {
    sorted_indices[feature_id].clear();
    sorted_indices[feature_id].shrink_to_fit();
  }
};


#endif
