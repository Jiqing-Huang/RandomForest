
#ifndef DECISIONTREE_TREEPREDICTOR_H
#define DECISIONTREE_TREEPREDICTOR_H

#include "../Global/GlobalConsts.h"
#include "../Generics/TypeDefs.h"

class StoredTree;
class ClassificationStoredTree;
class RegressionStoredTree;
class Dataset;

/// Predict output of test sample(s) with a trained tree

class TreePredictor {

 public:
  /// Empty predictor
  TreePredictor();

  /// Destructor
  virtual ~TreePredictor();

  /// Bind to Classification Tree
  void BindToTree(const ClassificationStoredTree &tree);

  /// Bind to Regression Tree
  void BindToTree(const RegressionStoredTree &tree);

  /// Predict one sample in a dataset by mean label value in regression task
  double PredictOneByMean(const Dataset *dataset,
                          uint32_t sample_id);

  /// Predict one sample in a dataset by probability in classification task
  vec_dbl_t PredictOneByProbability(const Dataset *dataset,
                                    uint32_t sample_id);

  /// Predict samples in a dataset by mean value in regression task
  /// filter == PredictAll: predict all samples
  /// filter == PredictPresent: predict samples whose weight are nonzero
  /// filter == PredictAbsent: predict samples whose weight are zero
  virtual vec_dbl_t PredictBatchByMean(const Dataset *dataset,
                                       const uint32_t filter = PredictAll);

  /// Predict samples in a dataset by probability in classification task
  /// filter == PredictAll: predict all samples
  /// filter == PredictPresent: predict samples whose weight are nonzero
  /// filter == PredictAbsent: predict samples whose weight are zero
  virtual vec_vec_dbl_t PredictBatchByProbability(const Dataset *dataset,
                                                  const uint32_t filter = PredictAll);

 protected:
  const ClassificationStoredTree *class_tree;
  const RegressionStoredTree *regress_tree;

  bool ToPredict(const vec_uint32_t &sample_weights,
                 const uint32_t idx,
                 const uint32_t selection);
  int32_t NextNode(const StoredTree *tree,
                   const Dataset *dataset,
                   int32_t cell_id,
                   uint32_t sample_id);
};
#endif
