
#ifndef DECISIONTREE_TREEPREDICTOR_H
#define DECISIONTREE_TREEPREDICTOR_H

#include "../Tree/StoredTree.h"
#include "../Dataset/Dataset.h"
#include "../Global/GlobalConsts.h"
#include "../Generics/Generics.h"

namespace TreePredictorNonMember {
inline int32_t NextNode(const StoredTree *tree,
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
  if (feature_type & IsContinuous)
    return (Generics::At<float>(features, sample_id) < info.float_point) ? left_id : right_id;
  if (feature_type & IsOrdinal)
    return (Generics::At<uint32_t>(features, sample_id) <= info.integer) ? left_id : right_id;
  if (feature_type & IsOneVsAll)
    return (Generics::At<uint32_t>(features, sample_id) == info.integer) ? left_id : right_id;
  if (feature_type & IsLowCardinality)
    return (1 << (Generics::At<uint32_t>(features, sample_id)) & info.integer) ? left_id : right_id;
  if (feature_type & IsHighCardinality) {
    uint32_t feature = Generics::At<uint32_t>(features, sample_id);
    uint32_t mask_idx = feature >> GetMaskIdx;
    uint32_t mask_shift = feature & GetMaskShift;
    return (tree->bitmasks[info.integer][mask_idx] & (1 << mask_shift)) ? left_id : right_id;
  }
  return 0;
}
} // namespace TreePredictorNonMember

class TreePredictor {

 public:
  TreePredictor() :
          class_tree(nullptr), regress_tree(nullptr) {}

  void BindToTree(const ClassificationStoredTree &tree) {
    class_tree = &tree;
  }

  void BindToTree(const RegressionStoredTree &tree) {
    regress_tree = &tree;
  }

  uint32_t PredictOneByMajority(const Dataset *dataset,
                                uint32_t sample_id) {
    if (class_tree->num_cell == 0)
       return class_tree->leaf_decision[0];
    int32_t cell_id = 0;
    do {
      cell_id = TreePredictorNonMember::NextNode(class_tree, dataset, cell_id, sample_id);
    } while (cell_id > 0);
    int32_t leaf_id = -cell_id;
    return class_tree->leaf_decision[leaf_id];
  }

  double PredictOneByMean(const Dataset *dataset,
                          uint32_t sample_id) {
    if (regress_tree->num_cell == 0)
      return regress_tree->leaf_mean[0];
    int32_t cell_id = 0;
    do {
      cell_id = TreePredictorNonMember::NextNode(regress_tree, dataset, cell_id, sample_id);
    } while (cell_id > 0);
    int32_t leaf_id = -cell_id;
    return regress_tree->leaf_mean[leaf_id];
  }

  vec_flt_t PredictOneByProbability(const Dataset *dataset,
                                    uint32_t sample_id) {
    if (class_tree->num_cell == 0)
      return class_tree->leaf_probability[0];
    int32_t cell_id = 0;
    do {
      cell_id = TreePredictorNonMember::NextNode(class_tree, dataset, cell_id, sample_id);
    } while (cell_id > 0);
    int32_t leaf_id = -cell_id;
    return class_tree->leaf_probability[leaf_id];
  }

  vec_uint32_t PredictAllByMajority(const Dataset *dataset) {
    vec_uint32_t predictions(dataset->Meta().size, 0);
    for (uint32_t idx = 0; idx != dataset->Meta().size; ++idx)
      predictions[idx] = PredictOneByMajority(dataset, idx);
    return predictions;
  }

  vec_dbl_t PredictAllByMean(const Dataset *dataset) {
    vec_dbl_t predictions(dataset->Meta().size, 0.0);
    for (uint32_t idx = 0; idx != dataset->Meta().size; ++idx)
      predictions[idx] = PredictOneByMean(dataset, idx);
    return predictions;
  }

  vector<vec_flt_t> PredictAllByProbability(const Dataset *dataset) {
    vector<vec_flt_t> predictions(dataset->Meta().size, vec_flt_t());
    for (uint32_t idx = 0; idx != dataset->Meta().size; ++idx)
      predictions[idx] = PredictOneByProbability(dataset, idx);
    return predictions;
  }

  vec_uint32_t PredictSelectedByMajority(const Dataset *dataset,
                                         const vec_uint32_t &sample_weights,
                                         const uint32_t selection = PredictAll) {
    vec_uint32_t predictions(dataset->Meta().size, 0);
    for (uint32_t idx = 0; idx != dataset->Meta().size; ++idx)
      if (ToPredict(sample_weights, idx, selection))
        predictions[idx] = PredictOneByMajority(dataset, idx);
    return predictions;
  }

  vec_dbl_t PredictSelectedByMean(const Dataset *dataset,
                                  const vec_uint32_t &sample_weights,
                                  const uint32_t selection = PredictAll) {
    vec_dbl_t predictions(dataset->Meta().size, 0.0);
    for (uint32_t idx = 0; idx != dataset->Meta().size; ++idx)
      if (ToPredict(sample_weights, idx, selection))
        predictions[idx] = PredictOneByMean(dataset, idx);
    return predictions;
  }

  vector<vec_flt_t> PredictSelectedByProbability(const Dataset *dataset,
                                                 const vec_uint32_t &sample_weights,
                                                 const uint32_t selection = PredictAll) {
    vector<vec_flt_t> predictions(dataset->Meta().size);
    for (uint32_t idx = 0; idx != dataset->Meta().size; ++idx)
      if (ToPredict(sample_weights, idx, selection))
        predictions[idx] = PredictOneByProbability(dataset, idx);
    return predictions;
  }

 private:
  const ClassificationStoredTree *class_tree;
  const RegressionStoredTree *regress_tree;

  bool ToPredict(const vec_uint32_t &sample_weights,
                 const uint32_t idx,
                 const uint32_t selection) {
    return selection == PredictAll ||
           (sample_weights[idx] > 0 && selection == PredictPresent) ||
           (sample_weights[idx] == 0 && selection == PredictAbsent);
  }
};
#endif
