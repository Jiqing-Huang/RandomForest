
#ifndef DECISIONTREE_TREEPARAMS_H
#define DECISIONTREE_TREEPARAMS_H

struct TreeParams {

  // input hyperparamters
  const uint32_t cost_function;
  const uint32_t min_leaf_node;
  const uint32_t min_split_node;
  const uint32_t max_depth;
  const uint32_t num_features_for_split;
  const uint32_t random_state;

  // calculated
  uint32_t max_num_leaf;
  uint32_t max_num_cell;
  uint32_t max_num_node;

  TreeParams(uint32_t cost_function,
             uint32_t min_leaf_node,
             uint32_t min_split_node,
             uint32_t max_depth,
             uint32_t num_features_for_split,
             uint32_t random_state):
          cost_function(cost_function), min_leaf_node(min_leaf_node), min_split_node(min_split_node),
          max_depth(max_depth), num_features_for_split(num_features_for_split), random_state(random_state),
          max_num_leaf(0), max_num_cell(0), max_num_node(0) {};

  void SetMaxNums(const double wnum_samples) {
    max_num_leaf = static_cast<uint32_t>(0.5 + wnum_samples) / min_leaf_node;
    max_num_cell = max_num_leaf - 1;
    max_num_node = max_num_cell + max_num_leaf;
  }

  void ResetMaxNums() {
    max_num_leaf = 0;
    max_num_cell = 0;
    max_num_node = 0;
  }
};

#endif