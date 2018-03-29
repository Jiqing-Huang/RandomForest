
#ifndef DECISIONTREE_SPLITTERIMPLEMENTOR_H
#define DECISIONTREE_SPLITTERIMPLEMENTOR_H

#include "../Generics/Generics.h"
#include "ClaSplitManipulator.h"
#include "RegSplitManipulator.h"

#define IS_VALID_LABEL \
  (std::is_integral<label_t>::value ^ std::is_same<SplitManipulatorType, VarianceSplitManipulator>::value)
#define IS_INTEGRAL_FEATURE (std::is_integral<feature_t>::value)

class Dataset;
class TreeParams;
class TreeNode;

class BaseSplitterImpl {
 public:
  BaseSplitterImpl() = default;
  virtual ~BaseSplitterImpl() = default;
  virtual void Split(const uint32_t feature_idx,
                     const uint32_t feature_type,
                     const Dataset *dataset,
                     TreeNode *node) = 0;
};

template <typename SplitManipulatorType>
class SplitterImpl: public BaseSplitterImpl {
 public:
  SplitterImpl(const Dataset *dataset,
               const TreeParams &params);
  void Split(const uint32_t feature_idx,
             const uint32_t feature_type,
             const Dataset *dataset,
             TreeNode *node) override;

 private:
  static thread_local std::unique_ptr<SplitManipulatorType> split_manipulator;
  const TreeParams &params;
  uint32_t cost_function;
  uint32_t num_classes;

  template <typename feature_t, typename label_t>
  std::enable_if_t<IS_VALID_LABEL && !IS_INTEGRAL_FEATURE, void>
  ContinuousSplit(const vector<feature_t> &features,
                  const vector<label_t> &labels,
                  const vec_uint32_t &sample_weights,
                  const uint32_t feature_idx,
                  TreeNode *node);
  template <typename feature_t, typename label_t>
  std::enable_if_t<!IS_VALID_LABEL || IS_INTEGRAL_FEATURE, void>
  ContinuousSplit(const vector<feature_t> &features,
                  const vector<label_t> &labels,
                  const vec_uint32_t &sample_weights,
                  const uint32_t feature_idx,
                  TreeNode *node);
  template <typename feature_t, typename label_t>
  std::enable_if_t<IS_VALID_LABEL && IS_INTEGRAL_FEATURE, void>
  DiscreteSplit(const vector<feature_t> &features,
                const vector<label_t> &labels,
                const vec_uint32_t &sample_weights,
                const uint32_t feature_idx,
                const uint32_t feature_type,
                TreeNode *node);
  template <typename feature_t, typename label_t>
  std::enable_if_t<!IS_VALID_LABEL || !IS_INTEGRAL_FEATURE, void>
  DiscreteSplit(const vector<feature_t> &features,
                const vector<label_t> &labels,
                const vec_uint32_t &sample_weights,
                const uint32_t feature_idx,
                const uint32_t feature_type,
                TreeNode *node);
  template <typename feature_t, typename label_t>
  std::enable_if_t<IS_VALID_LABEL && !IS_INTEGRAL_FEATURE, void>
  NumericalSplitter(const vector<feature_t> &features,
                    const vector<label_t> &labels,
                    const vec_uint32_t &sample_weights,
                    const uint32_t feature_idx,
                    TreeNode *node);
  void OrdinalSplitter(const uint32_t feature_idx,
                       TreeNode *node);
  void OneVsAllSplitter(const uint32_t feature_idx,
                        TreeNode *node);
  void LinearSplitter(const uint32_t feature_idx,
                      TreeNode *node);
  void BruteSplitter(const uint32_t feature_idx,
                     TreeNode *node);
  void GreedySplitter(const uint32_t feature_idx,
                      TreeNode *node);
  void UpdateManyVsManySplit(const vec_uint32_t &indicators,
                             const uint32_t feature_idx,
                             const double gain,
                             TreeNode *node);
};

using GiniSplitter = SplitterImpl<GiniSplitManipulator>;
using EntropySplitter = SplitterImpl<EntropySplitManipulator>;
using VarianceSplitter = SplitterImpl<VarianceSplitManipulator>;

#endif
