
#ifndef DECISIONTREE_PARALLELTREENODE_H
#define DECISIONTREE_PARALLELTREENODE_H

#include "TreeNode.h"

class ParallelTreeNode: public TreeNode {
 public:
  uint32_t thread_id_builder;
  vector<uint32_t> thread_id_splitter;

  ParallelTreeNode():
          TreeNode::TreeNode(), thread_id_builder(0), thread_id_splitter() {};

  ParallelTreeNode(const Dataset &dataset,
                   const Maths &util):
          TreeNode::TreeNode(dataset, util), thread_id_builder(0), thread_id_splitter() {};

  ParallelTreeNode(uint32_t node_id,
                   uint32_t type,
                   uint32_t depth,
                   ParallelTreeNode *parent,
                   const Dataset &dataset,
                   vector<uint32_t> &&sample_ids,
                   vector<uint32_t> &&labels,
                   vector<uint32_t> &&sample_weights):
          TreeNode::TreeNode(node_id, type, depth, parent, dataset, move(sample_ids),
                             move(labels), move(sample_weights)),
          thread_id_builder(parent->thread_id_builder), thread_id_splitter() {};

  ParallelTreeNode(ParallelTreeNode &&node) noexcept:
          TreeNode::TreeNode(move(node)),
          thread_id_builder(node.thread_id_builder), thread_id_splitter(move(node.thread_id_splitter)) {};

  ParallelTreeNode &operator=(ParallelTreeNode &&node) noexcept {
    thread_id_builder = node.thread_id_builder;
    thread_id_splitter = move(node.thread_id_splitter);
    TreeNode::operator=(move(node));
  }

  void SetToParallelBuilding(uint32_t thread_id) {
    type &= ~IsParallelSplitting;
    type |= IsParallelBuilding;
    thread_id_builder = thread_id;
  }

  void SetToParallelSplitting(uint32_t thread_id,
                              uint32_t feature) {
    type &= ~IsParallelBuilding;
    type |= IsParallelSplitting;
    thread_id_splitter[feature] = thread_id;
  }
};


#endif
