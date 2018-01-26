
#ifndef DECISIONTREE_PARALLELTREENODE_H
#define DECISIONTREE_PARALLELTREENODE_H

#include "TreeNode.h"

class ParallelTreeNode: public TreeNode {
 public:
  ParallelTreeNode():
          TreeNode::TreeNode(), parallel_type(0), thread_id_builder(0), thread_id_splitter() {};

  ParallelTreeNode(const Dataset *dataset):
          TreeNode::TreeNode(dataset), parallel_type(0), thread_id_builder(0), thread_id_splitter() {};

  ParallelTreeNode(uint32_t node_id,
                   uint32_t type,
                   ParallelTreeNode *parent):
          TreeNode::TreeNode(node_id, type, parent), parallel_type(parent->parallel_type),
          thread_id_builder(parent->thread_id_builder), thread_id_splitter() {};

  void SetParallelBuilding(uint32_t thread_id) {
    parallel_type = IsParallelSplittingType;
    thread_id_builder = thread_id;
  }

  void InitParallelSplitting(uint32_t num_features) {
    thread_id_splitter.resize(num_features, UINT32_MAX);
  }

  void SetParallelSplitting(uint32_t thread_id,
                            uint32_t feature) {
    parallel_type = IsParallelBuildingType;
    thread_id_splitter[feature] = thread_id;
  }

  bool InParallel() const {
    return (parallel_type & IsParallelBuildingType) || (parallel_type & IsParallelSplittingType);
  }

  bool IsParallelBuilding() const {
    return (parallel_type & IsParallelBuildingType) > 0;
  }

  bool IsParallelSplitting() const {
    return (parallel_type & IsParallelSplittingType) > 0;
  }

  uint32_t BuilderThreadId() const {
    return thread_id_builder;
  }

  uint32_t SplitterThreadId(uint32_t feature_idx) const {
    return thread_id_splitter[feature_idx];
  }

 private:
  uint32_t parallel_type;
  uint32_t thread_id_builder;
  vector<uint32_t> thread_id_splitter;
};

namespace ParallelTreeNodeNonMember {

inline bool IsParallelNode(const TreeNode *node) {
  const auto *parallel_node = dynamic_cast<const ParallelTreeNode*>(node);
  return parallel_node != nullptr;
}

inline uint32_t GetThreadId(const uint32_t feature_idx,
                            const TreeNode *node) {
  const auto *parallel_node = dynamic_cast<const ParallelTreeNode*>(node);
  if (!parallel_node) return 0;
  if (!parallel_node->InParallel()) return 0;
  if (parallel_node->IsParallelBuilding()) return parallel_node->BuilderThreadId();
  if (parallel_node->IsParallelSplitting()) return parallel_node->SplitterThreadId(feature_idx);
  return 0;
}
} // namespace ParallelTreeNodeNonMember
#endif
