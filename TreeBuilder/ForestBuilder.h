
#ifndef DECISIONTREE_FORESTBUILDER_H
#define DECISIONTREE_FORESTBUILDER_H

#include "TreeBuilder.h"
#include "../Tree/ForestParams.h"
#include "../Dataset/Dataset.h"

class ForestBuilder {
 public:
  vector<vector<float>> out_of_bag_prediction;
  vector<uint32_t> out_of_bag_vote;
  vector<StoredTree> forest;

  ForestBuilder(uint32_t cost_function,
                uint32_t min_leaf_node,
                uint32_t min_split_node,
                uint32_t max_depth,
                uint32_t num_features_for_split,
                uint32_t random_state,
                uint32_t num_trees):
          params(cost_function, min_leaf_node, min_split_node, max_depth,
                 num_features_for_split, random_state, num_trees),
          forest(num_trees), dataset(nullptr), presorted_indices(), util(cost_function, random_state) {};

  void LoadDataSet(Dataset &dataset);
  void UnloadDataSet();
  void Build();

 private:
  ForestParams params;

  Dataset *dataset;
  vector<vector<uint32_t>> presorted_indices;
  Maths util;

  void Presort();
  void Bootstrap(uint32_t num_boot_samples,
                 vector<uint32_t> &sample_weights);
};


#endif
