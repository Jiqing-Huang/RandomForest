
#ifndef DECISIONTREE_FORESTPARAMS_H
#define DECISIONTREE_FORESTPARAMS_H

#include "TreeParams.h"

class ForestParams: public TreeParams {
 public:
  ForestParams(uint32_t cost_function,
               uint32_t min_leaf_node,
               uint32_t min_split_node,
               uint32_t max_depth,
               uint32_t num_features_for_split,
               uint32_t random_state,
               uint32_t num_trees):
          TreeParams::TreeParams(cost_function, min_leaf_node, min_split_node, max_depth,
                                 num_features_for_split, random_state),
          num_trees(num_trees) {};

  uint32_t num_trees;
};


#endif
