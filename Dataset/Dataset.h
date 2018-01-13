
#ifndef DECISIONTREE_DATASET_H
#define DECISIONTREE_DATASET_H

#include <vector>
#include <cstdint>
#include <cmath>

#include "../Global/GlobalConsts.h"

using std::vector;

class Dataset {

  static const float MaxMultiplier = 100.0;

 public:
  // pointer to data
  const vector<vector<float>> *numerical_features;
  const vector<vector<uint32_t>> *discrete_features;
  const vector<uint32_t> *labels;
  const vector<uint32_t> *discrete_feature_types;
  const vector<double> *class_weights;

  // metadata
  uint32_t num_samples;

  const vector<uint32_t> *sample_weights;
  uint32_t weighted_num_samples;

  vector<uint32_t> integral_class_weights;
  uint32_t num_features;
  uint32_t num_numerical_features;
  uint32_t num_discrete_features;
  uint32_t num_classes;
  vector<uint32_t> num_bins;
  uint32_t max_num_bins;
  double multiplier;

  Dataset() = default;

  void LoadDataset(const vector<vector<float>> &numerical_features,
                   const vector<vector<uint32_t>> &discrete_features,
                   const vector<uint32_t> &labels,
                   const vector<uint32_t> &distint_feature_types,
                   const vector<double> &class_weights) {
    this->numerical_features = &numerical_features;
    num_numerical_features = static_cast<uint32_t>(numerical_features.size());
    this->discrete_features = &discrete_features;
    num_discrete_features = static_cast<uint32_t>(discrete_features.size());
    num_features = num_numerical_features + num_discrete_features;
    this->discrete_feature_types = &distint_feature_types;

    num_bins.resize(num_discrete_features, 0);
    for (uint32_t idx = 0; idx != num_discrete_features; ++idx) {
      uint32_t max_feature = 0;
      for (const auto &feature: discrete_features[idx])
        if (feature > max_feature) max_feature = feature;
      num_bins[idx] = max_feature + 1;
    }
    max_num_bins = 0;
    for (const auto &num_bin: num_bins)
      if (num_bin > max_num_bins) max_num_bins = num_bin;

    this->num_samples = static_cast<uint32_t>(labels.size());

    this->labels = &labels;
    uint32_t max_label = 0;
    for (const auto &label: labels)
      if (label > max_label) max_label = label;
    num_classes = max_label + 1;
    this->class_weights = &class_weights;

    multiplier = GetMultiplier();
    integral_class_weights.reserve(num_classes);
    for (const auto &weight: class_weights)
      integral_class_weights.push_back(static_cast<uint32_t>(0.5 + weight * multiplier));
  }

  void LoadSampleWeights(const vector<uint32_t> &sample_weights) {
    this->sample_weights = &sample_weights;
    this->weighted_num_samples = 0;
    for (uint32_t sample_id = 0; sample_id != this->num_samples; ++sample_id)
      this->weighted_num_samples += (*this->sample_weights)[sample_id];
  }

  double GetMultiplier() {
    for (double multiplier = 1.0; multiplier <= MaxMultiplier; multiplier += 1.0) {
      bool valid_multiplier = true;
      for (const auto &weight: *class_weights) {
        double approximated_weight = round(weight * multiplier) / multiplier;
        if (fabs(approximated_weight - weight) > FloatError)
          valid_multiplier = false;
      }
      if (valid_multiplier) return multiplier;
    }
    return 0.0;
  }
};

#endif
