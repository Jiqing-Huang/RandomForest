
#ifndef DECISIONTREE_TREEPREDICTOR_H
#define DECISIONTREE_TREEPREDICTOR_H

#include "../Dataset/StoredTree.h"
#include "../Dataset/Dataset.h"
#include "../Global/GlobalConsts.h"

class TreePredictor {

 public:
  TreePredictor():
          tree(nullptr) {}

  void BindToTree(const StoredTree &tree) {
    this->tree = &tree;
  }

  void PredictOneByMajority(const Dataset *dataset,
                            uint32_t sample_id,
                            uint32_t &prediction) {
    if (tree->num_cell == 0) {
      prediction = tree->leaf_decision[0];
      return;
    }
    int32_t cell_id = 0;
    do {
      cell_id = NextNode(dataset, cell_id, sample_id);
    } while (cell_id > 0);
    int32_t leaf_id = -cell_id;
    prediction = tree->leaf_decision[leaf_id];
  };

  void PredictOneByProbability(const Dataset *dataset,
                               uint32_t sample_id,
                               vector<float> &prediction) {
    if (tree->num_cell == 0) {
      for (uint32_t idx = 0; idx < prediction.size(); ++idx)
        prediction[idx] += tree->leaf_probability[0][idx];
      return;
    }
    int32_t cell_id = 0;
    do {
      cell_id = NextNode(dataset, cell_id, sample_id);
    } while (cell_id > 0);
    int32_t leaf_id = -cell_id;
    for (uint32_t idx = 0; idx < prediction.size(); ++idx)
      prediction[idx] += tree->leaf_probability[leaf_id][idx];
  };

  void PredictBatchByMajority(const Dataset *dataset,
                              const vector<uint32_t> &sample_ids,
                              vector<uint32_t> &predictions) {
    for (const auto &sample_id: sample_ids)
      PredictOneByMajority(dataset, sample_id, predictions[sample_id]);
  };

  void PredictBatchByProbability(const Dataset *dataset,
                                 const vector<uint32_t> &sample_ids,
                                 vector<vector<float>> &predictions) {
    for (const auto &sample_id: sample_ids)
      PredictOneByProbability(dataset, sample_id, predictions[sample_id]);
  };

  void PredictAllByMajority(const Dataset *dataset,
                            vector<uint32_t> &predictions) {
    for (uint32_t idx = 0; idx != dataset->num_samples; ++idx)
      PredictOneByMajority(dataset, idx, predictions[idx]);
  }

  void PredictAllByProbability(const Dataset *dataset,
                               vector<vector<float>> &predictions) {
    for (uint32_t idx = 0; idx != dataset->num_samples; ++idx)
      PredictOneByProbability(dataset, idx, predictions[idx]);
  }

  void PredictAbsentByMajority(const Dataset *dataset,
                               const vector<uint32_t> &sample_weights,
                               vector<uint32_t> &predictions) {
    for (uint32_t idx = 0; idx != dataset->num_samples; ++idx)
      if (!sample_weights[idx])
        PredictOneByMajority(dataset, idx, predictions[idx]);
  }

  void PredictAbsentByProbability(const Dataset *dataset,
                                  const vector<uint32_t> &sample_weights,
                                  vector<vector<float>> &predictions) {
    for (uint32_t idx = 0; idx != dataset->num_samples; ++idx)
      if (!sample_weights[idx])
        PredictOneByProbability(dataset, idx, predictions[idx]);
  }

  void PredictPresentByMajority(const Dataset *dataset,
                                const vector<uint32_t> &sample_weights,
                                vector<uint32_t> &predictions) {
    for (uint32_t idx = 0; idx != dataset->num_samples; ++idx)
      if (sample_weights[idx])
        PredictOneByMajority(dataset, idx, predictions[idx]);
  }

  void PredictPresentByProbability(const Dataset *dataset,
                                   const vector<uint32_t> &sample_weights,
                                   vector<vector<float>> &predictions) {
    for (uint32_t idx = 0; idx != dataset->num_samples; ++idx)
      if (sample_weights[idx])
        PredictOneByProbability(dataset, idx, predictions[idx]);
  }

 private:
  const StoredTree *tree;

  int32_t NextNode(const Dataset *dataset,
                   int32_t cell_id,
                   int32_t sample_id) {
    uint32_t cell_type = tree->cell_type[cell_id];
    uint32_t feature_idx = cell_type & GetFeatureIdx;
    uint32_t feature_type = cell_type & GetFeatureType;

    const StoredTree::Info &info = tree->cell_info[cell_id];
    int32_t left_id = tree->left[cell_id];
    int32_t right_id = tree->right[cell_id];

    if (feature_type & IsContinuous)
      return ((*dataset->numerical_features)[feature_idx][sample_id] < info.float_point)? left_id : right_id;
    if (feature_type & IsOrdinal)
      return ((*dataset->discrete_features)[feature_idx][sample_id] <= info.integer)? left_id : right_id;
    if (feature_type & IsOneVsAll)
      return ((*dataset->discrete_features)[feature_idx][sample_id] == info.integer)? left_id : right_id;
    if (feature_type & IsLowCardinality)
      return ((1 << (*dataset->discrete_features)[feature_idx][sample_id]) & info.integer)? left_id : right_id;
    if (feature_type & IsHighCardinality) {
      uint32_t feature = (*dataset->discrete_features)[feature_idx][sample_id];
      uint32_t mask_idx = feature >> GetMaskIdx;
      uint32_t mask_shift = feature & GetMaskShift;
      return (tree->bitmasks[info.integer][mask_idx] & (1 << mask_shift))? left_id : right_id;
    }
    return 0;
  }
};

#endif
