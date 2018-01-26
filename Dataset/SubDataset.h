
#ifndef DECISIONTREE_SUBDATASET_H
#define DECISIONTREE_SUBDATASET_H

#include <vector>
#include <cstdint>
#include <boost/variant/multivisitors.hpp>
#include "../Generics/Generics.h"
#include "../Predictor/Discriminator.h"
#include "../Splitter/SplitInfo.h"
#include "Dataset.h"

using std::vector;
using std::unique_ptr;
using std::make_unique;
using std::sort;
using std::pair;
using std::make_pair;

class SubDataset {
 public:
  explicit SubDataset(const uint32_t num_features):
    size(0), num_features(num_features), sample_ids(), labels(nullptr), sample_weights(),
    trios(num_features), sorted_indices(num_features) {}

  explicit SubDataset(const Dataset *dataset) {
    MakeRoot(dataset);
  }

  template <typename label_t>
  SubDataset(const uint32_t num_features,
             const uint32_t size,
             vec_uint32_t &&sample_ids,
             vector<label_t> &&labels,
             vec_uint32_t &&sample_weights):
    size(size), num_features(num_features), sample_ids(move(sample_ids)),
    labels(make_unique<generic_vec_t>(move(labels))),
    sample_weights(move(sample_weights)),
    trios(num_features), sorted_indices(num_features) {}

  uint32_t Size() const {
    return size;
  }

  uint32_t NumFeatures() const {
    return num_features;
  }

  const vec_uint32_t &SampleIds() const {
    return sample_ids;
  }

  const vec_uint32_t &SampleWeights() const {
    return sample_weights;
  }

  const generic_vec_t &Labels() const {
    return *labels;
  }

  const generic_vec_t &Features(const uint32_t feature_idx) const {
    return trios[feature_idx]->features;
  }

  const vec_uint32_t &SortedIdx(const uint32_t feature_idx) const {
    return sorted_indices[feature_idx];
  }

  const generic_vec_t &SortedLabels(const uint32_t feature_idx) const {
    return trios[feature_idx]->labels;
  }

  const vec_uint32_t &SortedSampleWeights(const uint32_t feature_idx) const {
    return trios[feature_idx]->sample_weights;
  }

  void MakeRoot(const Dataset *dataset) {
    num_features = dataset->Meta().num_features;
    trios.resize(num_features);
    sorted_indices.resize(num_features);
    MakeRootVisitor visitor(this, dataset->SampleWeights(), dataset->Meta().size);
    boost::apply_visitor(visitor, dataset->Labels());
  }

  void Gather(const Dataset *dataset,
              const uint32_t feature_idx) {
    generic_vec_t features = Gather(dataset->Features(feature_idx), sample_ids);
    trios[feature_idx] = make_unique<Trio>(std::move(features));
  }

  void Sort(const Dataset *dataset,
            const uint32_t feature_idx) {
    IndexSort(dataset->Features(feature_idx), feature_idx);
    GatherLabelsAndSampleWeights(feature_idx);
  }

  void Subset(const SubDataset *subset,
              const uint32_t feature_idx) {
    IndexSubset(subset, feature_idx);
    GatherLabelsAndSampleWeights(feature_idx);
  }

  void Subset(const Dataset *dataset,
              const vector<vec_uint32_t> *presorted_indices,
              const uint32_t feature_idx) {
    PresortedIndexSubset(dataset, (*presorted_indices)[feature_idx], feature_idx);
    GatherLabelsAndSampleWeights(feature_idx);
  }

  void Partition(const generic_vec_t &features,
                 const SplitInfo *split_info,
                 unique_ptr<SubDataset> &left_subset,
                 unique_ptr<SubDataset> &right_subset) const {
    PartitionVisitor visitor(this, split_info, sample_weights, left_subset, right_subset);
    boost::apply_visitor(visitor, features, *labels);
  }

  void DiscardSortedIdx(const uint32_t feature_idx) {
    sorted_indices[feature_idx].clear();
    sorted_indices[feature_idx].shrink_to_fit();
  }

  void DiscardTemporaryElements() {
    labels.reset();
    sample_weights.clear();
    sample_weights.shrink_to_fit();
    for (auto &trio: trios)
      trio.reset();
  }

  bool Empty(uint32_t feature_idx) const {
    return sorted_indices[feature_idx].empty();
  }

 private:
  uint32_t size;
  uint32_t num_features;
  vec_uint32_t sample_ids;
  unique_ptr<generic_vec_t> labels;
  vec_uint32_t sample_weights;
  vector<vec_uint32_t> sorted_indices;

  struct Trio {
    Trio(generic_vec_t &&labels,
         vec_uint32_t &&sample_weights):
      features(), labels(std::move(labels)), sample_weights(std::move(sample_weights)) {}

    explicit Trio(generic_vec_t &&features):
      features(std::move(features)), labels(), sample_weights() {}

    generic_vec_t features;
    generic_vec_t labels;
    vec_uint32_t sample_weights;
  };

  vector<unique_ptr<Trio>> trios;

  struct MakeRootVisitor: public boost::static_visitor<> {
    SubDataset *subset;
    const vec_uint32_t &source_sample_weights;
    const uint32_t size;

    explicit MakeRootVisitor(SubDataset *subset,
                             const vec_uint32_t &source_sample_weights,
                             uint32_t size):
      subset(subset), source_sample_weights(source_sample_weights), size(size) {}

    template <typename label_t>
    void operator()(const vector<label_t> &source_labels) {
      subset->MakeRoot(source_labels, source_sample_weights, size);
    };
  };

  template <typename label_t>
  void MakeRoot(const vector<label_t> &source_labels,
                const vec_uint32_t &source_sample_weights,
                const uint32_t source_size) {
    vector<label_t> target_labels;
    vec_uint32_t target_sample_weights;
    target_labels.reserve(source_size);
    target_sample_weights.reserve(source_size);
    sample_ids.reserve(source_size);
    for (uint32_t sample_id = 0; sample_id != source_size; ++sample_id)
      if (source_sample_weights[sample_id] > 0) {
        sample_ids.push_back(sample_id);
        target_labels.push_back(source_labels[sample_id]);
        target_sample_weights.push_back(source_sample_weights[sample_id]);
      }
    target_labels.shrink_to_fit();
    target_sample_weights.shrink_to_fit();
    sample_ids.shrink_to_fit();
    labels = make_unique<generic_vec_t>(move(target_labels));
    sample_weights = move(target_sample_weights);
    size = static_cast<uint32_t>(sample_ids.size());
  }

  generic_vec_t Gather(const generic_vec_t &source,
                       const vec_uint32_t &index) {
    GatherVisitor visitor(this, index);
    return boost::apply_visitor(visitor, source);
  }

  struct GatherVisitor: public boost::static_visitor<generic_vec_t> {
    SubDataset *subset;
    const vec_uint32_t &index;

    GatherVisitor(SubDataset *subset,
                  const vec_uint32_t &index):
      subset(subset), index(index) {}

    template <typename data_t>
    generic_vec_t operator()(const vector<data_t> &source) {
      return subset->Gather(source, index);
    }
  };

  template <typename data_t>
  vector<data_t> Gather(const vector<data_t> &source,
                        const vec_uint32_t &index) {
    vector<data_t> target;
    target.reserve(size);
    for (const auto &idx: index)
      target.push_back(source[idx]);
    return target;
  }

  struct PartitionVisitor: public boost::static_visitor<> {
    const SubDataset *const subset;
    const SplitInfo *const split_info;
    const vec_uint32_t &sample_weights;
    unique_ptr<SubDataset> &left_subset;
    unique_ptr<SubDataset> &right_subset;

    PartitionVisitor(const SubDataset *const subset,
                     const SplitInfo *const split_info,
                     const vec_uint32_t &sample_weights,
                     unique_ptr<SubDataset> &left_subset,
                     unique_ptr<SubDataset> &right_subset):
      subset(subset), split_info(split_info), sample_weights(sample_weights),
      left_subset(left_subset), right_subset(right_subset) {}

    template <typename feature_t, typename label_t>
    void operator()(const vector<feature_t> &features,
                    const vector<label_t> &labels) {
      subset->Partition(split_info, features, labels, sample_weights, left_subset, right_subset);
    }
  };

  template <typename feature_t, typename label_t>
  void Partition(const SplitInfo *split_info,
                 const vector<feature_t> &features,
                 const vector<label_t> &parent_labels,
                 const vec_uint32_t &parent_sample_weights,
                 unique_ptr<SubDataset> &left_subset,
                 unique_ptr<SubDataset> &right_subset) const {
    if (split_info->type == IsContinuous) {
      const ContinuousDiscriminator<feature_t> discriminator(split_info->info.float_type, features);
      PartitionExecutor(parent_labels, parent_sample_weights, discriminator, left_subset, right_subset);
    } else if (split_info->type == IsOrdinal) {
      const OrdinalDiscriminator<feature_t> discriminator(split_info->info.uint32_type, features);
      PartitionExecutor(parent_labels, parent_sample_weights, discriminator, left_subset, right_subset);
    } else if (split_info->type == IsOneVsAll) {
      const OneVsAllDiscriminator<feature_t> discriminator(split_info->info.uint32_type, features);
      PartitionExecutor(parent_labels, parent_sample_weights, discriminator, left_subset, right_subset);
    } else if (split_info->type == IsLowCardinality) {
      const LowCardDiscriminator<feature_t> discriminator(split_info->info.uint32_type, features);
      PartitionExecutor(parent_labels, parent_sample_weights, discriminator, left_subset, right_subset);
    } else if (split_info->type == IsHighCardinality) {
      const HighCardDiscriminator<feature_t> discriminator(*(split_info->info.ptr_type), features);
      PartitionExecutor(parent_labels, parent_sample_weights, discriminator, left_subset, right_subset);
    }
  }

  template <typename label_t, typename Discriminator>
  void PartitionExecutor(const vector<label_t> &parent_labels,
                         const vec_uint32_t &parent_sample_weights,
                         const Discriminator discriminator,
                         unique_ptr<SubDataset> &left_subset,
                         unique_ptr<SubDataset> &right_subset) const {
    vector<uint32_t> left_sample_ids, right_sample_ids;
    vector<label_t> left_labels, right_labels;
    vec_uint32_t left_sample_weights, right_sample_weights;

    left_sample_ids.reserve(size);
    right_sample_ids.reserve(size);
    left_labels.reserve(size);
    right_labels.reserve(size);
    left_sample_weights.reserve(size);
    right_sample_weights.reserve(size);

    uint32_t left_size = 0;
    uint32_t right_size = 0;

    for (uint32_t idx = 0; idx != size; ++idx) {
      uint32_t sample_id = sample_ids[idx];
      if (discriminator(sample_id)) {
        left_sample_ids.push_back(sample_id);
        left_labels.push_back(parent_labels[idx]);
        left_sample_weights.push_back(parent_sample_weights[idx]);
        ++left_size;
      } else {
        right_sample_ids.push_back(sample_id);
        right_labels.push_back(parent_labels[idx]);
        right_sample_weights.push_back(parent_sample_weights[idx]);
        ++right_size;
      }
    }

    left_sample_ids.shrink_to_fit();
    right_sample_ids.shrink_to_fit();
    left_labels.shrink_to_fit();
    right_labels.shrink_to_fit();
    left_sample_weights.shrink_to_fit();
    right_sample_weights.shrink_to_fit();

    left_subset = make_unique<SubDataset>(num_features, left_size, move(left_sample_ids), move(left_labels),
                                          move(left_sample_weights));
    right_subset = make_unique<SubDataset>(num_features, right_size, move(right_sample_ids), move(right_labels),
                                           move(right_sample_weights));
  }

  void IndexSort(const generic_vec_t &features,
                 const uint32_t feature_idx) {
    IndexSortVisitor visitor(this, feature_idx);
    boost::apply_visitor(visitor, features);
  }

  struct IndexSortVisitor: public boost::static_visitor<> {
    SubDataset *subset;
    const uint32_t feature_idx;

    IndexSortVisitor(SubDataset *subset,
                     const uint32_t feature_idx):
      subset(subset), feature_idx(feature_idx) {}

    template <typename feature_t>
    void operator()(const vector<feature_t> &features) {
      subset->IndexSort(features, feature_idx);
    }
  };

  template <typename feature_t>
  void IndexSort(const vector<feature_t> &features,
                 const uint32_t feature_idx) {
    using pair_t = pair<feature_t, uint32_t>;
    vector<pair_t> pairs;
    pairs.reserve(size);
    uint32_t idx = 0;
    for (const auto &sample_id: sample_ids)
      pairs.emplace_back(make_pair(features[sample_id], idx++));
    sort(pairs.begin(), pairs.end(),
         [](pair_t x, pair_t y) {
           return x.first < y.first;
         });
    sorted_indices[feature_idx].reserve(size);
    for (const auto &pair: pairs)
      sorted_indices[feature_idx].push_back(pair.second);
  }

  void IndexSubset(const SubDataset *parent_subset,
                   const uint32_t feature_idx) {
    uint32_t super_size = parent_subset->Size();
    vec_uint32_t super_to_sub(super_size, UINT32_MAX);
    uint32_t super_idx = 0;
    for (uint32_t sub_idx = 0; sub_idx != size; ++sub_idx) {
      uint32_t sample_id = sample_ids[sub_idx];
      while (parent_subset->sample_ids[super_idx] != sample_id) ++super_idx;
      super_to_sub[super_idx++] = sub_idx;
    }
    sorted_indices[feature_idx].reserve(size);
    const vec_uint32_t &source = parent_subset->SortedIdx(feature_idx);
    vec_uint32_t &target = sorted_indices[feature_idx];
    for (super_idx = 0; super_idx != super_size; ++super_idx) {
      uint32_t sub_idx = super_to_sub[source[super_idx]];
      if (sub_idx != UINT32_MAX)
        target.push_back(sub_idx);
    }
  }

  void PresortedIndexSubset(const Dataset *dataset,
                            const vec_uint32_t &presorted_idx,
                            const uint32_t feature_idx) {
    uint32_t super_size = dataset->Meta().size;
    vec_uint32_t super_to_sub(super_size, UINT32_MAX);
    uint32_t super_idx = 0;
    for (uint32_t sub_idx = 0; sub_idx != size; ++sub_idx)
      super_to_sub[sample_ids[sub_idx]] = sub_idx;
    sorted_indices[feature_idx].reserve(size);
    vec_uint32_t &target = sorted_indices[feature_idx];
    for (super_idx = 0; super_idx != super_size; ++super_idx) {
      uint32_t sub_idx = super_to_sub[presorted_idx[super_idx]];
      if (sub_idx != UINT32_MAX)
        target.push_back(sub_idx);
    }
  }

  void GatherLabelsAndSampleWeights(const uint32_t feature_idx) {
    generic_vec_t sorted_labels = Gather(*labels, sorted_indices[feature_idx]);
    vec_uint32_t sorted_sample_weights = Gather<uint32_t>(sample_weights, sorted_indices[feature_idx]);
    trios[feature_idx] = make_unique<Trio>(std::move(sorted_labels), std::move(sorted_sample_weights));
  }
};

#endif