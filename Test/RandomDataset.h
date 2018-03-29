
#ifndef DECISIONTREE_UNITTEST_H
#define DECISIONTREE_UNITTEST_H

#include <cstdint>
#include <random>
#include <memory>
#include "../Dataset/Dataset.h"
#include "../Global/GlobalConsts.h"

template <typename numerical_feature_t, typename discrete_feature_t, typename label_t>
class RandomDataset {
 public:

  std::unique_ptr<Dataset> dataset;
  std::vector<std::vector<numerical_feature_t>> numerical_feature;
  std::vector<std::vector<discrete_feature_t>> discrete_feature;
  vec_uint32_t discrete_feature_type;
  std::vector<label_t> labels;

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

    vec_dbl_t class_weights(num_classes, 1.0);

    dataset.reset();
    dataset = std::make_unique<Dataset>();

    dataset->AddFeatures(std::move(numerical_feature), vec_uint32_t(num_numerical_features, IsContinuous));
    dataset->AddFeatures(std::move(discrete_feature), discrete_feature_type);
    dataset->AddLabel(std::move(labels));
    dataset->AddSampleWeights(std::move(vec_uint32_t(num_samples, 1)));
    dataset->AddClassWeights(class_weights);
  }

  void RandomMixedRegDataset(uint32_t num_samples,
                             uint32_t num_numerical_features,
                             uint32_t num_ordinal_features,
                             uint32_t num_one_vs_all_features,
                             uint32_t num_many_vs_many_features,
                             uint32_t min_ordinal_bin,
                             uint32_t max_ordinal_bin,
                             uint32_t min_one_vs_all_bin,
                             uint32_t max_one_vs_all_bin,
                             uint32_t min_many_vs_many_bin,
                             uint32_t max_many_vs_many_bin) {

    uint32_t num_discrete_features = num_ordinal_features + num_one_vs_all_features + num_many_vs_many_features;

    std::mt19937 generator(0);
    std::uniform_real_distribution<double> rand_label(0.0, 1.0);
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

    dataset.reset();
    dataset = std::make_unique<Dataset>();

    dataset->AddFeatures(std::move(numerical_feature), vec_uint32_t(num_numerical_features, IsContinuous));
    dataset->AddFeatures(std::move(discrete_feature), discrete_feature_type);
    dataset->AddLabel(std::move(labels));
    dataset->AddSampleWeights(std::move(vec_uint32_t(num_samples, 1)));
  }
};

#endif
