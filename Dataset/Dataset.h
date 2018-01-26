
#ifndef DECISIONTREE_DATASET_H
#define DECISIONTREE_DATASET_H

#include <cstdint>
#include <algorithm>
#include <memory>
#include <vector>
#include <boost/variant/multivisitors.hpp>

#include "../Global/GlobalConsts.h"
#include "../Generics/Generics.h"

using std::vector;
using std::accumulate;
using std::max_element;
using std::move;
using std::unique_ptr;
using std::make_unique;

struct MetaData {
  MetaData():
    size(0), num_samples(0), num_features(0), num_bins(), max_num_bins(0), num_classes(0), wnum_samples(0),
    multiplier(1.0) {};

  // common to both regression and classification dataset
  uint32_t size;
  uint32_t num_samples;
  uint32_t num_features;
  vector<uint32_t> num_bins;
  uint32_t max_num_bins;

  // specific to classification dataset
  uint32_t num_classes;
  uint32_t wnum_samples;
  double multiplier;
};

class Dataset {

  static constexpr double MaxMultiplier = 100.0;

 public:
  Dataset():
    features(), feature_types(), labels(nullptr), sample_weights(), class_weights(nullptr), meta() {}

  template <typename feature_t>
  void AddFeature(const vector<feature_t> &feature,
                  uint32_t feature_type) {
    UpdateFeature(feature, feature_type);
    features.emplace_back(make_unique<generic_vec_t>(feature));
    feature_types.push_back(feature_type);
  }

  template <typename feature_t>
  void AddFeature(vector<feature_t> &&feature,
                  uint32_t feature_type) {
    UpdateFeature(feature, feature_type);
    features.emplace_back(make_unique<generic_vec_t>(move(feature)));
    feature_types.push_back(feature_type);
  }

  template <typename feature_t>
  void AddFeatures(const vector<vector<feature_t>> &features,
                   const vec_uint32_t &feature_types) {
    if (features.empty()) return;
    UpdateFeatures(features, feature_types);
    for (uint32_t idx = 0; idx != features.size(); ++idx) {
      this->features.emplace_back(make_unique<generic_vec_t>(features[idx]));
      this->feature_types.push_back(feature_types[idx]);
    }
  }

  template <typename feature_t>
  void AddFeatures(vector<vector<feature_t>> &&features,
                   const vec_uint32_t &feature_types) {
    if (features.empty()) return;
    UpdateFeatures(features, feature_types);
    for (uint32_t idx = 0; idx != features.size(); ++idx) {
      this->features.emplace_back(make_unique<generic_vec_t>(move(features[idx])));
      this->feature_types.push_back(feature_types[idx]);
    }
  }

  template <typename label_t>
  void AddLabel(const vector<label_t> &labels) {
    UpdateLabel(labels);
    this->labels = make_unique<generic_vec_t>(labels);
  }

  template <typename label_t>
  void AddLabel(vector<label_t> &&labels) {
    UpdateLabel(labels);
    this->labels = make_unique<generic_vec_t>(move(labels));
  }

  void AddSampleWeights(const vec_uint32_t &sample_weights) {
    meta.num_samples = accumulate(sample_weights.cbegin(), sample_weights.cend(), 0u);
    this->sample_weights = sample_weights;
    meta.wnum_samples = WNumSamples();
  }

  void AddSampleWeights(vec_uint32_t &&sample_weights) {
    meta.num_samples = accumulate(sample_weights.cbegin(), sample_weights.cend(), 0u);
    this->sample_weights = move(sample_weights);
    meta.wnum_samples = WNumSamples();
  }

  template <typename class_weight_t>
  void AddClassWeights(const vector<class_weight_t> &class_weights) {
    assert(meta.num_classes == class_weights.size());
    this->class_weights.reset();
    this->class_weights = make_unique<generic_vec_t>(class_weights);
    meta.wnum_samples = WNumSamples();
  }

  void IntegrateClassWeights() {
    assert(class_weights);
    assert(meta.num_classes != 0);
    IntegrateClassWeightsVisitor visitor(this);
    bool integratable = boost::apply_visitor(visitor, *class_weights);
    assert(integratable);
  }

  const MetaData &Meta() const {
    return meta;
  }

  uint32_t FeatureType(uint32_t feature_idx) const {
    return feature_types[feature_idx];
  }

  const generic_vec_t &Features(uint32_t feature_idx) const {
    return *features[feature_idx];
  }

  const generic_vec_t &Labels() const {
    return *labels;
  }

  const generic_vec_t &ClassWeights() const {
    return *class_weights;
  }

  const vec_uint32_t &SampleWeights() const {
    return sample_weights;
  }

 private:
  vector<unique_ptr<generic_vec_t>> features;
  vec_uint32_t feature_types;
  vec_uint32_t sample_weights;
  unique_ptr<generic_vec_t> labels;
  unique_ptr<generic_vec_t> class_weights;
  MetaData meta;

  template <typename feature_t>
  void UpdateFeature(const vector<feature_t> &feature,
                     uint32_t feature_type) {
    if (meta.size == 0)
      meta.size = static_cast<uint32_t>(feature.size());
    ++meta.num_features;
    uint32_t num_bins = 0;
    if (feature_type != IsContinuous)
      num_bins = max_element(feature.cbegin(), feature.cend());
    meta.num_bins.push_back(num_bins);
    if (num_bins > meta.max_num_bins) meta.max_num_bins = num_bins;
  }

  template <typename feature_t>
  void UpdateFeatures(const vector<vector<feature_t>> &features,
                      const vector<uint32_t> &feature_types) {
    if (meta.size == 0)
      meta.size = static_cast<uint32_t>(features[0].size());
    meta.num_features += features.size();
    for (uint32_t idx = 0; idx != features.size(); ++idx) {
      uint32_t num_bins = 0;
      if (feature_types[idx] != IsContinuous)
        num_bins = *max_element(features[idx].cbegin(), features[idx].cend()) + 1;
      meta.num_bins.push_back(num_bins);
      if (num_bins > meta.max_num_bins) meta.max_num_bins = num_bins;
    }
  }

  template <typename label_t>
  void UpdateLabel(const vector<label_t> &labels) {
    meta.num_classes = *max_element(labels.cbegin(), labels.cend()) + 1;
  }

  struct IntegrateClassWeightsVisitor: public boost::static_visitor<bool> {
    Dataset *dataset;

    explicit IntegrateClassWeightsVisitor(Dataset *dataset):
      dataset(dataset) {}

    template <typename class_weight_t>
    bool operator()(const vector<class_weight_t> &class_weights) {
      return dataset->IntegrateClassWeights<class_weight_t>(class_weights);
    }
  };

  template <typename class_weight_t>
  bool IntegrateClassWeights(const vector<class_weight_t> &class_weights) {
    meta.multiplier = 1.0;
    while (meta.multiplier <= MaxMultiplier) {
      bool valid_multiplier = true;
      for (const auto &weight: class_weights) {
        class_weight_t approximated_weight = round(weight * meta.multiplier) / meta.multiplier;
        class_weight_t error = (weight > approximated_weight)? weight - approximated_weight :
                                                               approximated_weight - weight;
        if (error > FloatError) {
          valid_multiplier = false;
          break;
        }
      }
      if (valid_multiplier) break;
      meta.multiplier += 1.0;
    }
    if (meta.multiplier > MaxMultiplier) return false;
    vector<uint32_t> integral_class_weights;
    for (const auto &weight: class_weights) {
      auto integral_weight = static_cast<uint32_t>(0.5 + weight * meta.multiplier);
      integral_class_weights.push_back(integral_weight);
    }
    this->class_weights.reset();
    this->class_weights = make_unique<generic_vec_t>(move(integral_class_weights));
    return true;
  }

  uint32_t WNumSamples() {
    if (class_weights) {
      WNumSamplesVisitor visitor(sample_weights);
      return boost::apply_visitor(visitor, *labels, *class_weights);
    } else {
      return Generics::Round<uint32_t>(meta.num_samples);
    }
  }

  struct WNumSamplesVisitor: public boost::static_visitor<uint32_t> {
    const vec_uint32_t &sample_weights;

    explicit WNumSamplesVisitor(const vec_uint32_t &sample_weights):
      sample_weights(sample_weights) {};

    template <typename label_t, typename class_weight_t>
    uint32_t operator()(const vector<label_t> &labels,
                        const vector<class_weight_t> &class_weights) {
      return WNumSamples(labels, sample_weights, class_weights);
    }
  };

  template <typename label_t, typename class_weight_t>
  static uint32_t WNumSamples(const vector<label_t> &labels,
                              const vec_uint32_t &sample_weights,
                              const vector<class_weight_t> &class_weights) {
    class_weight_t wnum_samples = 0;
    for (uint32_t idx = 0; idx != labels.size(); ++idx)
      wnum_samples += sample_weights[idx] * class_weights[labels[idx]];
    return static_cast<uint32_t>(0.5 + wnum_samples);
  }
};

#endif
