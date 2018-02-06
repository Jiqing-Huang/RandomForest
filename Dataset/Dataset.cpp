
#include <cstdint>
#include <vector>
#include <boost/variant.hpp>
#include "Dataset.h"
#include "../Generics/Generics.h"
#include "../Global/GlobalConsts.h"

/// Implementations of Dataset Class

template <typename feature_t>
void Dataset::AddFeature(const std::vector<feature_t> &feature,
                         uint32_t feature_type) {
  UpdateFeature(feature, feature_type);
  features.emplace_back(std::make_unique<generic_vec_t>(feature));
  feature_types.push_back(feature_type);
}

template <typename feature_t>
void Dataset::AddFeature(std::vector<feature_t> &&feature,
                         uint32_t feature_type) {
  UpdateFeature(feature, feature_type);
  features.emplace_back(std::make_unique<generic_vec_t>(move(feature)));
  feature_types.push_back(feature_type);
}

template <typename feature_t>
void Dataset::AddFeatures(const std::vector<std::vector<feature_t>> &features,
                          const vec_uint32_t &feature_types) {
  if (features.empty()) return;
  UpdateFeatures(features, feature_types);
  for (uint32_t idx = 0; idx != features.size(); ++idx) {
    this->features.emplace_back(std::make_unique<generic_vec_t>(features[idx]));
    this->feature_types.push_back(feature_types[idx]);
  }
}

template <typename feature_t>
void Dataset::AddFeatures(std::vector<std::vector<feature_t>> &&features,
                          const vec_uint32_t &feature_types) {
  if (features.empty()) return;
  UpdateFeatures(features, feature_types);
  for (uint32_t idx = 0; idx != features.size(); ++idx) {
    this->features.emplace_back(std::make_unique<generic_vec_t>(std::move(features[idx])));
    this->feature_types.push_back(feature_types[idx]);
  }
}

template <typename label_t>
void Dataset::AddLabel(const std::vector<label_t> &labels) {
  UpdateLabel(labels);
  this->labels = std::make_unique<generic_vec_t>(labels);
}

template <typename label_t>
void Dataset::AddLabel(std::vector<label_t> &&labels) {
  UpdateLabel(labels);
  this->labels = std::make_unique<generic_vec_t>(move(labels));
}

void Dataset::AddSampleWeights(const vec_uint32_t &sample_weights) {
  meta.num_samples = accumulate(sample_weights.cbegin(), sample_weights.cend(), 0u);
  this->sample_weights = sample_weights;
  if (!class_weights.empty())
    meta.wnum_samples = ComputeWNumSamples();
}

void Dataset::AddSampleWeights(vec_uint32_t &&sample_weights) {
  meta.num_samples = accumulate(sample_weights.cbegin(), sample_weights.cend(), 0u);
  this->sample_weights = move(sample_weights);
  if (!class_weights.empty())
    meta.wnum_samples = ComputeWNumSamples();
}

void Dataset::AddClassWeights(const vec_dbl_t &class_weights) {
  this->class_weights = class_weights;
  meta.wnum_samples = ComputeWNumSamples();
}

const MetaData &Dataset::Meta() const {
  return meta;
}

uint32_t Dataset::FeatureType(uint32_t feature_idx) const {
  return feature_types[feature_idx];
}

const generic_vec_t &Dataset::Features(uint32_t feature_idx) const {
  return *features[feature_idx];
}

const generic_vec_t &Dataset::Labels() const {
  return *labels;
}

const vec_dbl_t &Dataset::ClassWeights() const {
  return class_weights;
}

const vec_uint32_t &Dataset::SampleWeights() const {
  return sample_weights;
}

template <typename feature_t>
void Dataset::UpdateFeature(const std::vector<feature_t> &feature,
                            uint32_t feature_type) {
  if (meta.size == 0)
    meta.size = static_cast<uint32_t>(feature.size());
  ++meta.num_features;
  uint32_t num_bins = 0;
  // assume discrete feature is encoded as 0, 1 ... N, so that the cardinality should be N + 1
  if (feature_type != IsContinuous)
    num_bins = *std::max_element(feature.cbegin(), feature.cend()) + 1;
  meta.num_bins.push_back(num_bins);
  if (num_bins > meta.max_num_bins) meta.max_num_bins = num_bins;
}

template <typename feature_t>
void Dataset::UpdateFeatures(const std::vector<std::vector<feature_t>> &features,
                             const vec_uint32_t &feature_types) {
  if (meta.size == 0)
    meta.size = static_cast<uint32_t>(features[0].size());
  meta.num_features += features.size();
  for (uint32_t idx = 0; idx != features.size(); ++idx) {
    uint32_t num_bins = 0;
    if (feature_types[idx] != IsContinuous)
      num_bins = *std::max_element(features[idx].cbegin(), features[idx].cend()) + 1;
    meta.num_bins.push_back(num_bins);
    if (num_bins > meta.max_num_bins) meta.max_num_bins = num_bins;
  }
}

template <typename label_t>
std::enable_if_t<std::is_integral<label_t>::value, void>
Dataset::UpdateLabel(const std::vector<label_t> &labels) {
  meta.num_classes = *std::max_element(labels.cbegin(), labels.cend()) + 1;
}

template <typename label_t>
std::enable_if_t<std::is_integral<label_t>::value, double>
Dataset::WNumSamples(const std::vector<label_t> &labels) {
  double wnum_samples = 0.0;
  for (uint32_t idx = 0; idx != labels.size(); ++idx)
    wnum_samples += sample_weights[idx] * class_weights[labels[idx]];
  return wnum_samples;
}

double Dataset::ComputeWNumSamples() {
  return boost::apply_visitor(
    [this] (const auto &labels) {
      return this->WNumSamples(labels);
    }, *labels);
}

/// explicit instantiation of public template functions
#define DATASET_ADDFEATURE_COPY(type) \
template void Dataset::AddFeature(const std::vector<type> &, uint32_t);
DATASET_ADDFEATURE_COPY(uint8_t)
DATASET_ADDFEATURE_COPY(uint16_t)
DATASET_ADDFEATURE_COPY(uint32_t)
DATASET_ADDFEATURE_COPY(float)
DATASET_ADDFEATURE_COPY(double)

#define DATASET_ADDFEATURE_MOVE(type) \
template void Dataset::AddFeature(std::vector<type> &&, uint32_t);
DATASET_ADDFEATURE_MOVE(uint8_t)
DATASET_ADDFEATURE_MOVE(uint16_t)
DATASET_ADDFEATURE_MOVE(uint32_t)
DATASET_ADDFEATURE_MOVE(float)
DATASET_ADDFEATURE_MOVE(double)

#define DATASET_ADDFEATURES_COPY(type) \
template void Dataset::AddFeatures(const std::vector<std::vector<type>> &, const vec_uint32_t &);
DATASET_ADDFEATURES_COPY(uint8_t)
DATASET_ADDFEATURES_COPY(uint16_t)
DATASET_ADDFEATURES_COPY(uint32_t)
DATASET_ADDFEATURES_COPY(float)
DATASET_ADDFEATURES_COPY(double)

#define DATASET_ADDFEATURES_MOVE(type) \
template void Dataset::AddFeatures(std::vector<std::vector<type>> &&, const vec_uint32_t &);
DATASET_ADDFEATURES_MOVE(uint8_t)
DATASET_ADDFEATURES_MOVE(uint16_t)
DATASET_ADDFEATURES_MOVE(uint32_t)
DATASET_ADDFEATURES_MOVE(float)
DATASET_ADDFEATURES_MOVE(double)

#define DATASET_ADDLABEL_COPY(type) \
template void Dataset::AddLabel(const std::vector<type> &);
DATASET_ADDLABEL_COPY(uint8_t)
DATASET_ADDLABEL_COPY(uint16_t)
DATASET_ADDLABEL_COPY(uint32_t)
DATASET_ADDLABEL_COPY(float)
DATASET_ADDLABEL_COPY(double)

#define DATASET_ADDLABEL_MOVE(type) \
template void Dataset::AddLabel(std::vector<type> &&);
DATASET_ADDLABEL_MOVE(uint8_t)
DATASET_ADDLABEL_MOVE(uint16_t)
DATASET_ADDLABEL_MOVE(uint32_t)
DATASET_ADDLABEL_MOVE(float)
DATASET_ADDLABEL_MOVE(double)