
#ifndef DECISIONTREE_FORESTBUILDER_H
#define DECISIONTREE_FORESTBUILDER_H

#include "TreeBuilder.h"
#include "../Tree/ForestParams.h"
#include "../Dataset/Dataset.h"

class ForestBuilder {
 public:
  vector<vec_flt_t> out_of_bag_prediction;
  vector<uint32_t> out_of_bag_vote;
  vector<unique_ptr<StoredTree>> forest;

  ForestBuilder(uint32_t cost_function,
                uint32_t min_leaf_node,
                uint32_t min_split_node,
                uint32_t max_depth,
                uint32_t num_features_for_split,
                uint32_t random_state,
                uint32_t num_trees):
          params(cost_function, min_leaf_node, min_split_node, max_depth,
                 num_features_for_split, random_state, num_trees),
          forest(num_trees), dataset(nullptr), presorted_indices() {};

  void LoadDataSet(Dataset *dataset);
  void UnloadDataSet();
  void Build();

 private:
  ForestParams params;

  Dataset *dataset;
  vector<vec_uint32_t> presorted_indices;

  void Presort();

  struct IndexSortVisitor: public boost::static_visitor<vec_uint32_t> {
    ForestBuilder *forest_builder;

    explicit IndexSortVisitor(ForestBuilder *forest_builder):
      forest_builder(forest_builder) {}

    template <typename feature_t>
    vec_uint32_t operator()(const vector<feature_t> &features) {
      return forest_builder->IndexSort(features);
    }
  };

  template <typename feature_t>
  vec_uint32_t IndexSort(const vector<feature_t> &features) {
    using pair_t = pair<feature_t, uint32_t>;
    vector<pair_t> pairs;
    pairs.reserve(features.size());
    for (uint32_t idx = 0; idx != features.size(); ++idx)
      pairs.emplace_back(make_pair(features[idx], idx));
    sort(pairs.begin(), pairs.end(),
         [](pair_t x, pair_t y) {
           return x.first < y.first;
         });
    vec_uint32_t sorted_idx;
    sorted_idx.reserve(features.size());
    for (const auto &pair: pairs)
      sorted_idx.push_back(pair.second);
    return sorted_idx;
  }

  void Bootstrap(uint32_t num_boot_samples,
                 vector<uint32_t> &sample_weights);
};


#endif
