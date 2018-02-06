
#ifndef DECISIONTREE_SUBDATASET_H
#define DECISIONTREE_SUBDATASET_H

#include <vector>
#include <cstdint>
#include "../Generics/Generics.h"

class SplitInfo;

/// A subset of the original dataset a tree node represents
/// Each TreeNode object has one Subdataset object as its component
class Subdataset {
 public:

  /// Construct subset from the original dataset.
  /// Any sample with a non-zero sample weight is subsetted.
  /// Used to construct subset for root.
  explicit Subdataset(const Dataset *dataset);

  /// Construct subset from a given set of sample ids and their corresponding labels and sample weights
  /// These are obtained by partitioning the subset of parent tree node.
  Subdataset(const uint32_t num_features,
             const uint32_t size,
             vec_uint32_t &&sample_ids,
             generic_vec_t &&labels,
             vec_uint32_t &&sample_weights);

  ///////////
  /// Getters
  uint32_t Size() const;
  uint32_t NumFeatures() const;
  const vec_uint32_t &SampleIds() const;
  const vec_uint32_t &SampleWeights() const;
  const generic_vec_t &Labels() const;
  const generic_vec_t &Features(const uint32_t feature_idx) const;
  const vec_uint32_t &SortedIdx(const uint32_t feature_idx) const;
  const generic_vec_t &SortedLabels(const uint32_t feature_idx) const;
  const vec_uint32_t &SortedSampleWeights(const uint32_t feature_idx) const;
  ///////////

  /// Gather a feature from the original dataset by sample ids this subset holds
  void Gather(const Dataset *dataset,
              const uint32_t feature_idx);

  /// Index sort a numerical feature for later split finding
  void Sort(const Dataset *dataset,
            const uint32_t feature_idx);

  /// Subset the already sorted index of a numerical feature from an ancestor subset
  void Subset(const Subdataset *subset,
              const uint32_t feature_idx);

  /// Subset the already sorted index of a numerical feature from presorted index
  void Subset(const Dataset *dataset,
              const vec_vec_uint32_t *presorted_indices,
              const uint32_t feature_idx);

  /// Partition this subset into two subsets by the best split found
  /// Store them in the two unique_ptr arguments passed in
  void Partition(const generic_vec_t &features,
                 const SplitInfo *split_info,
                 std::unique_ptr<Subdataset> &left_subset,
                 std::unique_ptr<Subdataset> &right_subset) const;

  /// Clear and free memory for the sorted index of a numerical feature
  /// Whether this will be called depends on the memory saving strategy
  void DiscardSortedIdx(const uint32_t feature_idx);

  /// Clear and free memory for sample ids, labels, sample weights and discrete features
  /// This will be called right after the subset is partitioned
  void DiscardTemporaryElements();

  /// Whether the sorted index of a numerical feature is available (constructed and not discarded)
  bool Empty(uint32_t feature_idx) const;

 private:
  /// Size of the subset, only used to estimate computational complexity
  /// Not necessarily equals to number of samples
  uint32_t size;

  /// Number of features, always same as the original dataset
  uint32_t num_features;

  /// Ids of samples this subset holds, stored in ascending order
  vec_uint32_t sample_ids;

  /// Labels in the same order as sample ids
  std::unique_ptr<generic_vec_t> labels;

  /// Sample weights in the same order as sample ids
  vec_uint32_t sample_weights;

  /// Sorted index for the all features, ith element corresponds to the ith feature
  /// For numerical feature, it is constructed by sorting or by subsetting, when that feature is chosen for splitting
  /// For discrete feature, it is always empty
  vec_vec_uint32_t sorted_indices;

  /// Feature, label, sample_weight vectors for a feature, used in two contexts:
  /// 1. Numerical feature
  ///   Labels and sample weights in the order of sorted feature
  ///   They will be accessed sequentially to find the best split
  ///   Features are not used
  /// 2. Discrete feature
  ///   Features are subsetted from the original dataset in ascending order of sample ids
  ///   Labels and sample weights are not used
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
  std::vector<std::unique_ptr<Trio>> trios;

  /// Visitor template functions to the root constructor
  template <typename label_t>
  void MakeRoot(const std::vector<label_t> &source_labels,
                const vec_uint32_t &source_sample_weights,
                const uint32_t source_size);

  /// Generic gather function used to
  /// 1. Gather discrete feature from dataset
  /// 2. Gather labels and sample weights by sorted index of numerical feature
  generic_vec_t Gather(const generic_vec_t &source,
                       const vec_uint32_t &index);

  /// Visitor template functions to the generic gather
  template <typename data_t>
  std::vector<data_t> Gather(const std::vector<data_t> &source,
                             const vec_uint32_t &index);

  /// Visitor template partition functions to the public partition function
  /// Select a discriminator specific to the split type and call the partition executor
  template <typename feature_t>
  std::enable_if_t<!std::is_integral<feature_t>::value, void>
  PartitionByContinuousFeature(const SplitInfo *split_info,
                               const std::vector<feature_t> &features,
                               std::unique_ptr<Subdataset> &left_subset,
                               std::unique_ptr<Subdataset> &right_subset) const;
  template <typename feature_t>
  std::enable_if_t<std::is_integral<feature_t>::value, void>
  PartitionByContinuousFeature(const SplitInfo *split_info,
                               const std::vector<feature_t> &features,
                               std::unique_ptr<Subdataset> &left_subset,
                               std::unique_ptr<Subdataset> &right_subset) const { /* do nothing */ }
  template <typename feature_t>
  std::enable_if_t<std::is_integral<feature_t>::value, void>
  PartitionByDiscreteFeature(const SplitInfo *split_info,
                             const std::vector<feature_t> &features,
                             std::unique_ptr<Subdataset> &left_subset,
                             std::unique_ptr<Subdataset> &right_subset) const;
  template <typename feature_t>
  std::enable_if_t<!std::is_integral<feature_t>::value, void>
  PartitionByDiscreteFeature(const SplitInfo *split_info,
                             const std::vector<feature_t> &features,
                             std::unique_ptr<Subdataset> &left_subset,
                             std::unique_ptr<Subdataset> &right_subset) const { /* do nothing */ }

  /// Actual partition executor to partition the sample ids
  template <typename Discriminator>
  void PartitionExecutor(const Discriminator discriminator,
                         std::unique_ptr<Subdataset> &left_subset,
                         std::unique_ptr<Subdataset> &right_subset) const;

  /// Called by PartitionExecutor to partition labels and sample_weights according to sample_ids
  /// The implementation of partition is separated in three different functions in order to decouple template
  /// arguments to reduce the amount of codes generated
  template <typename data_t>
  std::pair<std::vector<data_t>, std::vector<data_t>>
  PartitionBySampleIds(const vec_uint32_t &left_sample_ids,
                       const std::vector<data_t> &parent_data) const;

  /// Sort index in order of the feature
  void IndexSort(const generic_vec_t &features,
                 const uint32_t feature_idx);

  /// Visitor template indexsort functions to the generic indexsort
  template <typename feature_t>
  std::enable_if_t<!std::is_integral<feature_t>::value, void>
  IndexSort(const std::vector<feature_t> &features,
            const uint32_t feature_idx);

  /// Subset sorted index from an ancestor tree node
  void IndexSubset(const Subdataset *ancestor_subset,
                   const uint32_t feature_idx);

  /// Subset sorted index from the original dataset if index is pre-sorted
  void PresortedIndexSubset(const Dataset *dataset,
                            const vec_uint32_t &presorted_idx,
                            const uint32_t feature_idx);
};

#endif