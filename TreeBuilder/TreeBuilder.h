
#ifndef DECISIONTREE_TREEBUILDER_H
#define DECISIONTREE_TREEBUILDER_H

#include <memory>
#include <atomic>
#include <mutex>
#include "../Generics/TypeDefs.h"
#include "../Tree/TreeParams.h"
#include "../Splitter/Splitter.h"

class Dataset;
class StoredTree;
class TreeNode;

class TreeBuilder {
 public:
  TreeBuilder(uint32_t cost_function,
              uint32_t min_leaf_node,
              uint32_t min_split_node,
              uint32_t max_depth,
              uint32_t max_num_nodes,
              uint32_t num_features_for_split,
              uint32_t random_state);
  explicit TreeBuilder(const TreeParams &params);
  ~TreeBuilder();
  void LoadDataSet(const Dataset *dataset,
                   const vec_vec_uint32_t *presorted_indices = nullptr);
  TreeNode *SetupRoot();
  uint32_t InitSplit(TreeNode *node);
  std::pair<vec_uint32_t::iterator, vec_uint32_t::iterator> GetFeatureSet();
  void FindSplitOnAllFeatures(TreeNode *node);
  void FindSplitOnOneFeature(uint32_t feature_idx,
                             TreeNode *node);
  bool FindSplitFinished(TreeNode *node);
  bool DoSplit(TreeNode *node);
  bool MakeLeaf(TreeNode *node);
  void WriteToTree(StoredTree *tree);

 private:
  TreeParams params;
  const Dataset *dataset;
  const vec_vec_uint32_t *presorted_indices;
  std::unique_ptr<TreeNode> root;
  std::atomic<uint32_t> cell_count;
  std::atomic<uint32_t> leaf_count;
  std::mutex update_mut;
  bool finish;

  bool PrepareSubset(uint32_t feature_type,
                     uint32_t feature_idx,
                     TreeNode *node);
  TreeNode* LookForAncestor(uint32_t feature_idx,
                            TreeNode *node);
  bool UpdateStatus(TreeNode *node);
};

#endif
