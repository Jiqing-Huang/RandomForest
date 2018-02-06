
#ifndef DECISIONTREE_DATASET_H
#define DECISIONTREE_DATASET_H

#include <cstdint>
#include <algorithm>
#include <memory>
#include <vector>

#include "MetaData.h"
#include "../Generics/TypeDefs.h"
#include "../Generics/Generics.h"

/// The class that holds training / testing dataset for either classification or regression.
/// features and labels are of generic vector type, which can take uin8_t, uint16_t, uint32_t,
/// float or double.
class Dataset {

 public:

  /// Empty dataset
  Dataset():
    features(), feature_types(), labels(nullptr), sample_weights(), class_weights(), meta() {}

  /// Add a feature vector by copying
  template <typename feature_t>
  void AddFeature(const std::vector<feature_t> &feature,
                  uint32_t feature_type);

  /// Add a feature vector by moving
  template <typename feature_t>
  void AddFeature(std::vector<feature_t> &&feature,
                  uint32_t feature_type);

  /// Add a number of features vectors by copying
  template <typename feature_t>
  void AddFeatures(const std::vector<std::vector<feature_t>> &features,
                   const vec_uint32_t &feature_types);

  /// Add a number of features vectors by moving
  template <typename feature_t>
  void AddFeatures(std::vector<std::vector<feature_t>> &&features,
                   const vec_uint32_t &feature_types);

  /// Add label vector by copying
  template <typename label_t>
  void AddLabel(const std::vector<label_t> &labels);

  /// Add label vector by moving
  template <typename label_t>
  void AddLabel(std::vector<label_t> &&labels);

  /// Add sample weights by copying
  void AddSampleWeights(const vec_uint32_t &sample_weights);

  /// Add sample weights by moving
  void AddSampleWeights(vec_uint32_t &&sample_weights);

  /// Add class weights
  void AddClassWeights(const vec_dbl_t &class_weights);

  ///////////
  /// Getters
  const MetaData &Meta() const;
  uint32_t FeatureType(uint32_t feature_idx) const;
  const generic_vec_t &Features(uint32_t feature_idx) const;
  const generic_vec_t &Labels() const;
  const vec_dbl_t &ClassWeights() const;
  const vec_uint32_t &SampleWeights() const;
  ///////////

 private:
  std::vector<std::unique_ptr<generic_vec_t>> features;
  vec_uint32_t feature_types;
  std::unique_ptr<generic_vec_t> labels;
  vec_uint32_t sample_weights;
  vec_dbl_t class_weights;
  MetaData meta;

  /// Update metadata on added feature
  template <typename feature_t>
  void UpdateFeature(const std::vector<feature_t> &feature,
                     uint32_t feature_type);

  /// Update metadata on a number of added features
  template <typename feature_t>
  void UpdateFeatures(const std::vector<std::vector<feature_t>> &features,
                      const vec_uint32_t &feature_types);

  /// Update number of output classes if label is of integral type.
  /// Assume label is encoded as 0, 1 ... N. The number of classes should be N + 1.
  template <typename label_t>
  std::enable_if_t<std::is_integral<label_t>::value, void>
  UpdateLabel(const std::vector<label_t> &labels);

  template <typename label_t>
  std::enable_if_t<!std::is_integral<label_t>::value, void>
  UpdateLabel(const std::vector<label_t> &labels) { /* do nothing */ }

  /// Number of samples weighted on both sample weights and class weights.
  /// Enabled for integral type of label only, i.e. classification.
  /// Regression has no output classes so wnum_samples is nonsensical.
  template <typename label_t>
  std::enable_if_t<std::is_integral<label_t>::value, double>
  WNumSamples(const std::vector<label_t> &labels);

  template <typename label_t>
  std::enable_if_t<!std::is_integral<label_t>::value, double>
  WNumSamples(const std::vector<label_t> &labels) { /* do nothing */ return 0.0; }

  /// WNumSamples invoker for generic vector type
  double ComputeWNumSamples();
};

#endif
