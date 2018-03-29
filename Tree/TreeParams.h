
#ifndef DECISIONTREE_TREEPARAMS_H
#define DECISIONTREE_TREEPARAMS_H

struct TreeParams {

  const uint32_t cost_function;
  const uint32_t min_leaf_node;
  const uint32_t min_split_node;
  const uint32_t max_depth;
  const uint32_t max_num_nodes;
  const uint32_t num_features_for_split;
  const uint32_t random_state;

  TreeParams(uint32_t cost_function,
             uint32_t min_leaf_node,
             uint32_t min_split_node,
             uint32_t max_depth,
             uint32_t max_num_nodes,
             uint32_t num_features_for_split,
             uint32_t random_state):
          cost_function(cost_function), min_leaf_node(min_leaf_node), min_split_node(min_split_node),
          max_depth(max_depth), max_num_nodes(max_num_nodes), num_features_for_split(num_features_for_split),
          random_state(random_state) {}
};

#endif