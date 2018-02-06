
#include "TreePredictor.h"
#include "../Tree/StoredTree.h"
#include "../Dataset/Dataset.h"

TreePredictor::TreePredictor() :
  class_tree(nullptr), regress_tree(nullptr) {}

TreePredictor::~TreePredictor() = default;

void TreePredictor::BindToTree(const ClassificationStoredTree &tree) {
  class_tree = &tree;
}

void TreePredictor::BindToTree(const RegressionStoredTree &tree) {
  regress_tree = &tree;
}

double TreePredictor::PredictOneByMean(const Dataset *dataset,
                                       uint32_t sample_id) {
  if (regress_tree->num_cell == 0)
    return regress_tree->leaf_mean[0];
  int32_t cell_id = 0;
  do {
    cell_id = NextNode(regress_tree, dataset, cell_id, sample_id);
  } while (cell_id > 0);
  int32_t leaf_id = -cell_id;
  return regress_tree->leaf_mean[leaf_id];
}

vec_dbl_t TreePredictor::PredictOneByProbability(const Dataset *dataset,
                                                 uint32_t sample_id) {
  if (class_tree->num_cell == 0)
    return class_tree->leaf_probability[0];
  int32_t cell_id = 0;
  do {
    cell_id = NextNode(class_tree, dataset, cell_id, sample_id);
  } while (cell_id > 0);
  int32_t leaf_id = -cell_id;
  return class_tree->leaf_probability[leaf_id];
}

vec_dbl_t TreePredictor::PredictBatchByMean(const Dataset *dataset,
                                            const uint32_t filter) {
  vec_dbl_t predictions(dataset->Meta().size, 0.0);
  const vec_uint32_t &sample_weights = dataset->SampleWeights();
  for (uint32_t idx = 0; idx != dataset->Meta().size; ++idx)
    if (ToPredict(sample_weights, idx, filter))
      predictions[idx] = PredictOneByMean(dataset, idx);
  return predictions;
}

vec_vec_dbl_t TreePredictor::PredictBatchByProbability(const Dataset *dataset,
                                                       const uint32_t filter) {
  vec_vec_dbl_t predictions(dataset->Meta().size);
  const vec_uint32_t &sample_weights = dataset->SampleWeights();
  for (uint32_t idx = 0; idx != dataset->Meta().size; ++idx)
    if (ToPredict(sample_weights, idx, filter))
      predictions[idx] = PredictOneByProbability(dataset, idx);
  return predictions;
}

bool TreePredictor::ToPredict(const vec_uint32_t &sample_weights,
                              const uint32_t idx,
                              const uint32_t filter) {
  return filter == PredictAll ||
         (sample_weights[idx] > 0 && filter == PredictPresent) ||
         (sample_weights[idx] == 0 && filter == PredictAbsent);
}

int32_t TreePredictor::NextNode(const StoredTree *tree,
                                const Dataset *dataset,
                                int32_t cell_id,
                                uint32_t sample_id) {
  uint32_t cell_type = tree->cell_type[cell_id];
  uint32_t feature_idx = cell_type & GetFeatureIdx;
  uint32_t feature_type = cell_type & GetFeatureType;

  const StoredTree::Info &info = tree->cell_info[cell_id];
  int32_t left_id = tree->left[cell_id];
  int32_t right_id = tree->right[cell_id];

  const generic_vec_t &features = dataset->Features(feature_idx);
  switch (feature_type) {
    case IsContinuous:
      return (Generics::RoundAt<float>(features, sample_id) < info.float_point) ? left_id : right_id;
    case IsOrdinal:
      return (Generics::RoundAt<uint32_t>(features, sample_id) <= info.integer) ? left_id : right_id;
    case IsOneVsAll:
      return (Generics::RoundAt<uint32_t>(features, sample_id) == info.integer) ? left_id : right_id;
    case IsLowCardinality:
      return (1 << (Generics::RoundAt<uint32_t>(features, sample_id)) & info.integer) ? left_id : right_id;
    case IsHighCardinality: {
      uint32_t feature = Generics::RoundAt<uint32_t>(features, sample_id);
      uint32_t mask_idx = feature >> GetMaskIdx;
      uint32_t mask_shift = feature & GetMaskShift;
      return (tree->bitmasks[info.integer][mask_idx] & (1 << mask_shift)) ? left_id : right_id;
    }
    default:
      return 0;
  }
}