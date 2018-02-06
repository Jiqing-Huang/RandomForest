
#ifndef DECISIONTREE_TREEBUILDER_H
#define DECISIONTREE_TREEBUILDER_H

#include <mutex>
#include <atomic>
#include <memory>
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
              uint32_t num_features_for_split,
              uint32_t random_state);
  explicit TreeBuilder(const TreeParams &params);
  virtual ~TreeBuilder();
  void LoadDataSet(const Dataset &dataset);
  void UpdateDataSet();
  void UnloadDataSet();
  void LoadPresortedIndices(const vec_vec_uint32_t &presorted_indices);
  void UnloadPresortedIndices();
  virtual void Build(StoredTree &tree);
 protected:
  TreeParams params;
  const Dataset *dataset;
  const vec_vec_uint32_t *presorted_indices;
  Splitter splitter;
  vec_vec_uint32_t feature_sets;
  std::vector<std::unique_ptr<TreeNode>> nodes;
  std::atomic<uint32_t> node_top;
  bool finish;
  void Init(uint32_t num_threads,
            StoredTree &tree);
  void BuildAllNodes(uint32_t node_id,
                     StoredTree &tree);
  bool Splittable(const TreeNode *node);
  void GetBestSplit(TreeNode *node);
  const vec_uint32_t &ShuffleFeatures(TreeNode *node);
  void Split(uint32_t feature_type,
             uint32_t feature_idx,
             TreeNode *node);
  virtual bool UpdateStatus(TreeNode *node);
  void PrepareLeaf(TreeNode *node,
                   StoredTree &tree);
  void PrepareCell(TreeNode *node,
                   StoredTree &tree);
  virtual void InsertChildNodes(TreeNode *node);
  virtual void CleanUp(StoredTree &tree);
 private:
  bool PrepareSubset(uint32_t feature_type,
                     uint32_t feature_idx,
                     TreeNode *node);
  TreeNode* LookForAncestor(uint32_t feature_idx,
                            TreeNode *node);
  uint32_t MakeLeaf(TreeNode *node,
                    StoredTree &tree);
  void LinkLeaf(const TreeNode *node,
                uint32_t leaf_id,
                StoredTree &tree);
  uint32_t MakeCell(TreeNode *node,
                    StoredTree &tree);
  void LinkCell(const TreeNode *node,
                uint32_t cell_id,
                StoredTree &tree);
};

#endif
