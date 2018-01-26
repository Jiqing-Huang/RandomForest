

#ifndef DECISIONTREE_HYPERPARAMS_H
#define DECISIONTREE_HYPERPARAMS_H

#include <cstdint>
#include <boost/variant.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <boost/variant/static_visitor.hpp>
#include <boost/variant/multivisitors.hpp>
#include "../Dataset/Dataset.h"
#include "../Util/Maths.h"

struct TreeParams {

  // input hyperparamters
  const uint32_t cost_function;
  const uint32_t min_leaf_node;
  const uint32_t min_split_node;
  const uint32_t max_depth;
  const uint32_t num_features_for_split;
  const uint32_t random_state;

  // calculated

  uint32_t effective_min_leaf_node;
  uint32_t effective_min_split_node;

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
          effective_min_leaf_node(0), effective_min_split_node(0), max_num_leaf(0), max_num_cell(0), max_num_node(0) {};

  void SetMaxNums(const uint32_t wnum_samples,
                  const double multiplier) {
    effective_min_leaf_node = static_cast<uint32_t>(0.5 + min_leaf_node * multiplier);
    effective_min_split_node = static_cast<uint32_t>(0.5 + min_split_node * multiplier);
    max_num_leaf = wnum_samples / min_leaf_node;
    max_num_cell = max_num_leaf - 1;
    max_num_node = max_num_cell + max_num_leaf;
  }

  void ResetMaxNums() {
    max_num_leaf = 0;
    max_num_cell = 0;
    max_num_node = 0;
  }

 private:

};

#endif