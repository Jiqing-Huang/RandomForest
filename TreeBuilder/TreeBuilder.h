
#ifndef DECISIONTREE_TREEBUILDER_H
#define DECISIONTREE_TREEBUILDER_H

#include <cstdint>

#include "../Util/Cost.h"
#include "../Util/Random.h"
#include "../Tree/TreeParams.h"
#include "../Dataset/Dataset.h"
#include "../Tree/StoredTree.h"
#include "../Splitter/TreeNodeSplitter.h"

using std::mutex;
using std::unique_lock;

class TreeBuilder {
 public:
  TreeBuilder(uint32_t cost_function,
              uint32_t min_leaf_node,
              uint32_t min_split_node,
              uint32_t max_depth,
              uint32_t num_features_for_split,
              uint32_t random_state):
          params(cost_function, min_leaf_node, min_split_node, max_depth,
                 num_features_for_split, random_state),
          dataset(nullptr), presorted_indices(nullptr), nodes(), node_top(0),
          splitter_ptr(cost_function), finish(false) {
    Cost::Init(cost_function);
    Random::Init(random_state);
  };
  explicit TreeBuilder(const TreeParams &params):
          params(params), dataset(nullptr), presorted_indices(nullptr), nodes(), node_top(0),
          splitter_ptr(params.cost_function), finish(false) {};
  virtual ~TreeBuilder() = default;
  void LoadDataSet(const Dataset &dataset);
  void UpdateDataSet();
  void UnloadDataSet();
  void LoadPresortedIndices(const vector<vector<uint32_t>> &presorted_indices);
  void UnloadPresortedIndices();
  virtual void Build(StoredTree &tree);
 protected:
  TreeParams params;
  const Dataset *dataset;
  const vector<vector<uint32_t>> *presorted_indices;
  SplitterPtr splitter_ptr;
  vector<vector<uint32_t>> feature_sets;
  vector<unique_ptr<TreeNode>> nodes;
  std::atomic<uint32_t> node_top;
  bool finish;
  void Init(uint32_t num_threads,
            StoredTree &tree);
  void BuildAllNodes(uint32_t node_id,
                     StoredTree &tree);
  void GetBestSplit(TreeNode *node);
  const vector<uint32_t> &ShuffleFeatures(TreeNode *node);
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
