
#ifndef DECISIONTREE_UNITTEST_H
#define DECISIONTREE_UNITTEST_H

#include <cstdint>
#include <random>
#include "../Dataset/Dataset.h"

class UnitTest {
 public:

  Dataset dataset;
  vector<vector<float>> numerical_feature;
  vector<vector<uint32_t>> discrete_feature;
  vector<uint32_t> discrete_feature_type;
  vector<uint32_t> labels;
  vector<double> class_weights;

  void RandomNumericDataset(uint32_t num_samples,
                            uint32_t num_numerical_features,
                            uint32_t num_classes) {
    std::mt19937 generator(0);
    std::uniform_real_distribution<float>  randfeat(0.0, 1.0);
    std::uniform_int_distribution<uint32_t> randlabel(0, num_classes - 1);

    numerical_feature.resize(num_numerical_features);
    for (uint32_t i = 0; i < num_numerical_features; ++i) {
      numerical_feature[i].reserve(num_samples);
      for (uint32_t j = 0; j < num_samples; ++j)
        numerical_feature[i].push_back(randfeat(generator));
    }

    labels.reserve(num_samples);
    for (uint32_t i = 0; i < num_samples; ++i)
      labels.push_back(randlabel(generator));

    class_weights.assign(num_classes, 1.0);

    dataset.LoadDataset(numerical_feature, discrete_feature, labels,
                        discrete_feature_type, class_weights);
  }

  void RandomOrdinalDataset(uint32_t num_samples,
                            uint32_t num_ordinal_features,
                            uint32_t num_bin,
                            uint32_t num_classes) {
    std::mt19937 generator(0);
    std::uniform_int_distribution<uint32_t>  randfeat(0, num_bin - 1);
    std::uniform_int_distribution<uint32_t> randlabel(0, num_classes - 1);

    discrete_feature.resize(num_ordinal_features);
    for (uint32_t i = 0; i < num_ordinal_features; ++i) {
      discrete_feature[i].reserve(num_samples);
      for (uint32_t j = 0; j < num_samples; ++j)
        discrete_feature[i].push_back(randfeat(generator));
    }

    labels.reserve(num_samples);
    for (uint32_t i = 0; i < num_samples; ++i)
      labels.push_back(randlabel(generator));

    class_weights.assign(num_classes, 1.0);
    discrete_feature_type.assign(num_ordinal_features, IsOrdinal);

    dataset.LoadDataset(numerical_feature, discrete_feature, labels,
                        discrete_feature_type, class_weights);
  }

  void RandomOenVsAllDataset(uint32_t num_samples,
                             uint32_t num_on_vs_all_features,
                             uint32_t num_bin,
                             uint32_t num_classes) {
    std::mt19937 generator(0);
    std::uniform_int_distribution<uint32_t>  randfeat(0, num_bin - 1);
    std::uniform_int_distribution<uint32_t> randlabel(0, num_classes - 1);

    discrete_feature.resize(num_on_vs_all_features);
    for (uint32_t i = 0; i < num_on_vs_all_features; ++i) {
      discrete_feature[i].reserve(num_samples);
      for (uint32_t j = 0; j < num_samples; ++j)
        discrete_feature[i].push_back(randfeat(generator));
    }

    labels.reserve(num_samples);
    for (uint32_t i = 0; i < num_samples; ++i)
      labels.push_back(randlabel(generator));

    class_weights.assign(num_classes, 1.0);
    discrete_feature_type.assign(num_on_vs_all_features, IsOneVsAll);

    dataset.LoadDataset(numerical_feature, discrete_feature, labels,
                        discrete_feature_type, class_weights);
  }

  void RandomManyVsManyDataset(uint32_t num_samples,
                               uint32_t num_many_vs_many_features,
                               uint32_t num_bin,
                               uint32_t num_classes) {
    std::mt19937 generator(0);
    std::uniform_int_distribution<uint32_t>  randfeat(0, num_bin - 1);
    std::uniform_int_distribution<uint32_t> randlabel(0, num_classes - 1);

    discrete_feature.resize(num_many_vs_many_features);
    for (uint32_t i = 0; i < num_many_vs_many_features; ++i) {
      discrete_feature[i].reserve(num_samples);
      for (uint32_t j = 0; j < num_samples; ++j)
        discrete_feature[i].push_back(randfeat(generator));
    }

    labels.reserve(num_samples);
    for (uint32_t i = 0; i < num_samples; ++i)
      labels.push_back(randlabel(generator));

    class_weights.assign(num_classes, 1.0);
    discrete_feature_type.assign(num_many_vs_many_features, IsManyVsMany);

    dataset.LoadDataset(numerical_feature, discrete_feature, labels,
                        discrete_feature_type, class_weights);
  }

  void RandomMixedDataset(uint32_t num_samples,
                          uint32_t num_numerical_features,
                          uint32_t num_ordinal_features,
                          uint32_t num_one_vs_all_features,
                          uint32_t num_many_vs_many_features,
                          uint32_t min_ordinal_bin,
                          uint32_t max_ordinal_bin,
                          uint32_t min_one_vs_all_bin,
                          uint32_t max_one_vs_all_bin,
                          uint32_t min_many_vs_many_bin,
                          uint32_t max_many_vs_many_bin,
                          uint32_t num_classes) {

    uint32_t num_discrete_features = num_ordinal_features + num_one_vs_all_features + num_many_vs_many_features;

    std::mt19937 generator(0);
    std::uniform_int_distribution<uint32_t> rand_label(0, num_classes - 1);
    std::uniform_real_distribution<float>  rand_nume_feat(0.0, 1.0);

    numerical_feature.resize(num_numerical_features);
    for (uint32_t i = 0; i < num_numerical_features; ++i) {
      numerical_feature[i].reserve(num_samples);
      for (uint32_t j = 0; j < num_samples; ++j)
        numerical_feature[i].push_back(rand_nume_feat(generator));
    }

    discrete_feature.resize(num_discrete_features);

    std::uniform_int_distribution<uint32_t> rand_ordinal_bin(min_ordinal_bin, max_ordinal_bin);
    for (uint32_t i = 0; i < num_ordinal_features; ++i) {
      discrete_feature[i].reserve(num_samples);
      std::uniform_int_distribution<uint32_t> rand_disc_feat(0, rand_ordinal_bin(generator) - 1);
      for (uint32_t j = 0; j < num_samples; ++j)
        discrete_feature[i].push_back(rand_disc_feat(generator));
    }

    std::uniform_int_distribution<uint32_t> rand_one_vs_all_bin(min_one_vs_all_bin, max_one_vs_all_bin);
    for (uint32_t i = num_ordinal_features; i < num_ordinal_features + num_one_vs_all_features; ++i) {
      discrete_feature[i].reserve(num_samples);
      std::uniform_int_distribution<uint32_t> rand_disc_feat(0, rand_one_vs_all_bin(generator) - 1);
      for (uint32_t j = 0; j < num_samples; ++j)
        discrete_feature[i].push_back(rand_disc_feat(generator));
    }

    std::uniform_int_distribution<uint32_t> rand_many_vs_many_bin(min_many_vs_many_bin, max_many_vs_many_bin);
    for (uint32_t i = num_ordinal_features + num_one_vs_all_features; i < num_discrete_features; ++i) {
      discrete_feature[i].reserve(num_samples);
      std::uniform_int_distribution<uint32_t> rand_disc_feat(0, rand_many_vs_many_bin(generator) - 1);
      for (uint32_t j = 0; j < num_samples; ++j)
        discrete_feature[i].push_back(rand_disc_feat(generator));
    }

    labels.reserve(num_samples);
    for (uint32_t i = 0; i < num_samples; ++i)
      labels.push_back(rand_label(generator));

    class_weights.assign(num_classes, 1.0);
    discrete_feature_type.resize(num_discrete_features, 0);
    fill(discrete_feature_type.begin(),
         discrete_feature_type.begin() + num_ordinal_features,
         IsOrdinal);
    fill(discrete_feature_type.begin() + num_ordinal_features,
         discrete_feature_type.begin() + num_ordinal_features + num_one_vs_all_features,
         IsOneVsAll);
    fill(discrete_feature_type.begin() + num_ordinal_features + num_one_vs_all_features,
         discrete_feature_type.end(),
         IsManyVsMany);

    dataset.LoadDataset(numerical_feature, discrete_feature, labels,
                        discrete_feature_type, class_weights);
  }
};

#endif
