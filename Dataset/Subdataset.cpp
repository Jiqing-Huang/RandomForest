
#include "Dataset.h"
#include "Subdataset.h"
#include "IndexedFeature.h"
#include "../Predictor/Discriminator.h"
#include "../Splitter/SplitInfo.h"

/// Implementation of Subdataset Class

Subdataset::Subdataset(const Dataset *dataset) {
  num_features = dataset->Meta().num_features;
  trios.resize(num_features);
  sorted_indices.resize(num_features);
  boost::apply_visitor([this, &dataset] (const auto &labels) {
    return this->MakeRoot(labels, dataset->SampleWeights(), dataset->Meta().size);
  }, dataset->Labels());
}

Subdataset::Subdataset(const uint32_t num_features,
                       const uint32_t size,
                       vec_uint32_t &&sample_ids,
                       generic_vec_t &&labels,
                       vec_uint32_t &&sample_weights):
  size(size), num_features(num_features), sample_ids(std::move(sample_ids)),
  labels(std::make_unique<generic_vec_t>(std::move(labels))),
  sample_weights(std::move(sample_weights)),
  trios(num_features), sorted_indices(num_features) {}

uint32_t Subdataset::Size() const {
  return size;
}

uint32_t Subdataset::NumFeatures() const {
  return num_features;
}

const vec_uint32_t &Subdataset::SampleIds() const {
  return sample_ids;
}

const vec_uint32_t &Subdataset::SampleWeights() const {
  return sample_weights;
}

const generic_vec_t &Subdataset::Labels() const {
  return *labels;
}

const generic_vec_t &Subdataset::Features(const uint32_t feature_idx) const {
  return trios[feature_idx]->features;
}

const vec_uint32_t &Subdataset::SortedIdx(const uint32_t feature_idx) const {
  return sorted_indices[feature_idx];
}

const generic_vec_t &Subdataset::SortedLabels(const uint32_t feature_idx) const {
  return trios[feature_idx]->labels;
}

const vec_uint32_t &Subdataset::SortedSampleWeights(const uint32_t feature_idx) const {
  return trios[feature_idx]->sample_weights;
}

void Subdataset::Gather(const Dataset *dataset,
                        const uint32_t feature_idx) {
  /// Call the private overloaded gather to do the work
  trios[feature_idx] = std::make_unique<Trio>(Gather(dataset->Features(feature_idx), sample_ids));
}

void Subdataset::Sort(const Dataset *dataset,
                      const uint32_t feature_idx) {
  /// Sort index, and then reorder labels and sample_weights by the sorted index
  IndexSort(dataset->Features(feature_idx), feature_idx);
  trios[feature_idx] = std::make_unique<Trio>(Gather(*labels, sorted_indices[feature_idx]),
                                              Gather(sample_weights, sorted_indices[feature_idx]));
}

void Subdataset::Subset(const Subdataset *subset,
                        const uint32_t feature_idx) {
  /// Subset sorted index from ancestor node, and then reorder labels and sample_weights by the sorted index
  IndexSubset(subset, feature_idx);
  trios[feature_idx] = std::make_unique<Trio>(Gather(*labels, sorted_indices[feature_idx]),
                                              Gather(sample_weights, sorted_indices[feature_idx]));
}

void Subdataset::Subset(const Dataset *dataset,
                        const std::vector<vec_uint32_t> *presorted_indices,
                        const uint32_t feature_idx) {
  /// Subset from presorted index, and then reorder labels and sample_weights by the sorted index
  PresortedIndexSubset(dataset, (*presorted_indices)[feature_idx], feature_idx);
  trios[feature_idx] = std::make_unique<Trio>(Gather(*labels, sorted_indices[feature_idx]),
                                              Gather(sample_weights, sorted_indices[feature_idx]));
}

void Subdataset::Partition(const generic_vec_t &features,
                           const SplitInfo *split_info,
                           std::unique_ptr<Subdataset> &left_subset,
                           std::unique_ptr<Subdataset> &right_subset) const {
  boost::apply_visitor([this, &split_info, &left_subset, &right_subset] (const auto &features) {
    if (split_info->type == IsContinuous) {
      this->PartitionByContinuousFeature(split_info, features, left_subset, right_subset);
    } else {
      this->PartitionByDiscreteFeature(split_info, features, left_subset, right_subset);
    }
  }, features);
}

void Subdataset::DiscardSortedIdx(const uint32_t feature_idx) {
  sorted_indices[feature_idx].clear();
  sorted_indices[feature_idx].shrink_to_fit();
}

void Subdataset::DiscardTemporaryElements() {
  labels.reset();
  sample_weights.clear();
  sample_weights.shrink_to_fit();
  for (auto &trio: trios)
    trio.reset();
}

bool Subdataset::Empty(uint32_t feature_idx) const {
  return sorted_indices[feature_idx].empty();
}

template <typename label_t>
void Subdataset::MakeRoot(const std::vector<label_t> &source_labels,
                          const vec_uint32_t &source_sample_weights,
                          const uint32_t source_size) {
  /// collect all samples whose sample weights are non-zero
  std::vector<label_t> target_labels;
  target_labels.reserve(source_size);
  sample_weights.reserve(source_size);
  sample_ids.reserve(source_size);
  size = 0;
  for (uint32_t sample_id = 0; sample_id != source_size; ++sample_id)
    if (source_sample_weights[sample_id] > 0) {
      sample_ids.push_back(sample_id);
      target_labels.push_back(source_labels[sample_id]);
      sample_weights.push_back(source_sample_weights[sample_id]);
      ++size;
    }
  sample_weights.shrink_to_fit();
  sample_ids.shrink_to_fit();
  labels = std::make_unique<generic_vec_t>(std::move(target_labels));
}

generic_vec_t Subdataset::Gather(const generic_vec_t &source,
                                 const vec_uint32_t &index) {
  return boost::apply_visitor([this, &index] (const auto &source) {
    auto target = this->Gather(source, index);
    return generic_vec_t(std::move(target));
  }, source);
}

template <typename data_t>
std::vector<data_t> Subdataset::Gather(const std::vector<data_t> &source,
                                       const vec_uint32_t &random_indices) {
  std::vector<data_t> target;
  target.resize(size);
  uint32_t idx = 0;
  for (const auto &random_idx: random_indices)
    target[idx++] = source[random_idx];
  return target;
}

template <typename feature_t>
std::enable_if_t<!std::is_integral<feature_t>::value, void>
Subdataset::PartitionByContinuousFeature(const SplitInfo *split_info,
                                         const std::vector<feature_t> &features,
                                         std::unique_ptr<Subdataset> &left_subset,
                                         std::unique_ptr<Subdataset> &right_subset) const {
  const ContinuousDiscriminator<feature_t> discriminator(split_info->info.float_type, features);
  PartitionExecutor(discriminator, left_subset, right_subset);
}

template <typename feature_t>
std::enable_if_t<std::is_integral<feature_t>::value, void>
Subdataset::PartitionByDiscreteFeature(const SplitInfo *split_info,
                                       const std::vector<feature_t> &features,
                                       std::unique_ptr<Subdataset> &left_subset,
                                       std::unique_ptr<Subdataset> &right_subset) const {
  if (split_info->type == IsOrdinal) {
    const OrdinalDiscriminator<feature_t> discriminator(split_info->info.uint32_type, features);
    PartitionExecutor(discriminator, left_subset, right_subset);
  } else if (split_info->type == IsOneVsAll) {
    const OneVsAllDiscriminator<feature_t> discriminator(split_info->info.uint32_type, features);
    PartitionExecutor(discriminator, left_subset, right_subset);
  } else if (split_info->type == IsLowCardinality) {
    const LowCardDiscriminator<feature_t> discriminator(split_info->info.uint32_type, features);
    PartitionExecutor(discriminator, left_subset, right_subset);
  } else if (split_info->type == IsHighCardinality) {
    const HighCardDiscriminator<feature_t> discriminator(*(split_info->info.ptr_type), features);
    PartitionExecutor(discriminator, left_subset, right_subset);
  }
}

template <typename Discriminator>
void Subdataset::PartitionExecutor(const Discriminator discriminator,
                                   std::unique_ptr<Subdataset> &left_subset,
                                   std::unique_ptr<Subdataset> &right_subset) const {
  /// parition sample ids
  vec_uint32_t left_sample_ids, right_sample_ids;
  left_sample_ids.reserve(size);
  right_sample_ids.reserve(size);
  for (uint32_t idx = 0; idx != size; ++idx) {
    uint32_t sample_id = sample_ids[idx];
    if (discriminator(sample_id)) {
      left_sample_ids.push_back(sample_id);
    } else {
      right_sample_ids.push_back(sample_id);
    }
  }
  left_sample_ids.shrink_to_fit();
  right_sample_ids.shrink_to_fit();
  uint32_t left_size = static_cast<uint32_t>(left_sample_ids.size());

  /// partition labels and sample_weights by the partitioned sample_ids
  auto sample_weights_pair = PartitionBySampleIds(left_sample_ids, sample_weights);
  std::pair<generic_vec_t, generic_vec_t> labels_pair = boost::apply_visitor(
    [this, &left_sample_ids] (const auto &labels) {
      auto data_pair = this->PartitionBySampleIds(left_sample_ids, labels);
      return std::make_pair<generic_vec_t, generic_vec_t>(std::move(data_pair.first), std::move(data_pair.second));
    }, *labels);

  /// construct subsets for left and right child
  left_subset = std::make_unique<Subdataset>(num_features, left_size, std::move(left_sample_ids),
                                             std::move(labels_pair.first), std::move(sample_weights_pair.first));
  right_subset = std::make_unique<Subdataset>(num_features, size - left_size, std::move(right_sample_ids),
                                              std::move(labels_pair.second), std::move(sample_weights_pair.second));
}

template <typename data_t>
std::pair<std::vector<data_t>, std::vector<data_t>>
Subdataset::PartitionBySampleIds(const vec_uint32_t &left_sample_ids,
                                 const std::vector<data_t> &parent_data) const {
  uint32_t left_idx = 0;
  uint32_t left_size = static_cast<uint32_t>(left_sample_ids.size());
  std::vector<data_t> left_data, right_data;
  left_data.reserve(left_size);
  right_data.reserve(size -left_size);
  for (uint32_t idx = 0; idx != size; ++idx)
    if (left_idx != left_size && left_sample_ids[left_idx] == sample_ids[idx]) {
      left_data.push_back(parent_data[idx]);
      ++left_idx;
    } else {
      right_data.push_back(parent_data[idx]);
    }
  return std::make_pair(std::move(left_data), std::move(right_data));
}

void Subdataset::IndexSort(const generic_vec_t &features,
                           const uint32_t feature_idx) {
  boost::apply_visitor([this, &feature_idx] (const auto &features) {
    this->IndexSort(features, feature_idx);
  }, features);
}

template <typename feature_t>
std::enable_if_t<!std::is_integral<feature_t>::value, void>
Subdataset::IndexSort(const std::vector<feature_t> &features,
                      const uint32_t feature_idx) {
  /// Pair features with indices, sort the pair and collect sorted index into resulting vector
  /// This seems to do more work than directly sorting a index vector using a customised comparator,
  /// but this is actually faster because of better memory locality
  std::vector<IndexedFeature<feature_t>> indexed_features(size);
  for (uint32_t idx = 0; idx != size; ++idx) {
    indexed_features[idx].feature = features[sample_ids[idx]];
    indexed_features[idx].idx = idx;
  }
  std::sort(indexed_features.begin(), indexed_features.end());
  sorted_indices[feature_idx].resize(size);
  for (uint32_t idx = 0; idx != size; ++idx)
    sorted_indices[feature_idx][idx] = indexed_features[idx].idx;
}

void Subdataset::IndexSubset(const Subdataset *ancestor_subset,
                             const uint32_t feature_idx) {
  /// Map superset index to subset index by two pointer walk-through
  /// This is possible because sample ids are always in ascending order
  uint32_t super_size = ancestor_subset->Size();
  vec_uint32_t super_to_sub_mapping(super_size, UINT32_MAX /* UINT32_MAX indicates absense in the subset */);
  uint32_t super_idx = 0;
  const vec_uint32_t &ancestor_sample_ids = ancestor_subset->sample_ids;
  for (uint32_t sub_idx = 0; sub_idx != size; ++sub_idx) {
    uint32_t sample_id = sample_ids[sub_idx];
    while (ancestor_sample_ids[super_idx] != sample_id) ++super_idx;
    super_to_sub_mapping[super_idx] = sub_idx;
  }

  /// Walk through sorted index in the superset
  /// Collect the mapped subset index in sorted order
  sorted_indices[feature_idx].resize(size, 0);
  vec_uint32_t &target = sorted_indices[feature_idx];
  uint32_t idx = 0;
  for (const auto &sorted_idx: ancestor_subset->SortedIdx(feature_idx)) {
    uint32_t sub_idx = super_to_sub_mapping[sorted_idx];
    if (sub_idx != UINT32_MAX)
      target[idx++] = sub_idx;
  }
}

void Subdataset::PresortedIndexSubset(const Dataset *dataset,
                                      const vec_uint32_t &presorted_idx,
                                      const uint32_t feature_idx) {
  /// Map superset index to subset index
  /// The superset is the whole dataset so that we just loop from 0 to N to implicitly visit the superset
  uint32_t super_size = dataset->Meta().size;
  vec_uint32_t super_to_sub_mapping(super_size, UINT32_MAX /* UINT32_MAX indicates absense in subset */);
  for (uint32_t sub_idx = 0; sub_idx != size; ++sub_idx)
    super_to_sub_mapping[sample_ids[sub_idx]] = sub_idx;

  /// Walk through sorted index in the superset
  /// Collect the mapped subset index in sorted order
  sorted_indices[feature_idx].resize(size, 0);
  vec_uint32_t &target = sorted_indices[feature_idx];
  uint32_t idx = 0;
  for (const auto &sorted_idx: presorted_idx) {
    uint32_t sub_idx = super_to_sub_mapping[sorted_idx];
    if (sub_idx != UINT32_MAX)
      target[idx++] = sub_idx;
  }
}