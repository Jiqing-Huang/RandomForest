
#ifndef DECISIONTREE_TREEBUILDER_H
#define DECISIONTREE_TREEBUILDER_H

#include <cstdint>

#include "../Tree/TreeParams.h"
#include "../Dataset/Dataset.h"
#include "../Dataset/StoredTree.h"
#include "../Util/Maths.h"
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
          dataset(nullptr), presorted_indices(nullptr), nodes(), node_top(0), splitter_ptr(cost_function),
          util(cost_function, random_state), finish(false) {};

  explicit TreeBuilder(const TreeParams &params):
          params(params), dataset(nullptr), presorted_indices(nullptr), nodes(), node_top(0),
          splitter_ptr(params.cost_function), util(params.cost_function, params.random_state), finish(false) {};

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
  vector<uint32_t> feature_set;

  using GiniSplitter = TreeNodeSplitter<double, GiniCostComputer>;
  using EntropySplitter = TreeNodeSplitter<uint32_t, EntropyCostComputer>;
  struct SplitterPtr {
    unique_ptr<GiniSplitter> gini_splitter;
    unique_ptr<EntropySplitter> entropy_splitter;

    explicit SplitterPtr(uint32_t cost_function) {
      if (cost_function == GiniCost)
        gini_splitter = make_unique<GiniSplitter>();
      if (cost_function == EntropyCost)
        entropy_splitter = make_unique<EntropySplitter>();
    }

    ~SplitterPtr() {
      gini_splitter.reset();
      entropy_splitter.reset();
    }
  };

  SplitterPtr splitter_ptr;

  vector<unique_ptr<TreeNode>> nodes;
  std::atomic<uint32_t> node_top;
  Maths util;
  bool finish;

  virtual void SetupRoot(StoredTree &tree);

  void BuildAllNodes(uint32_t node_id,
                     StoredTree &tree);

  void GetBestSplit(TreeNode *node);

  template <typename Splitter_Type>
  void SplitOnFeature(uint32_t feature_type,
                      uint32_t feature_idx,
                      TreeNode *node,
                      Splitter_Type &splitter);

  void GetFeatureTypeAndIndex(uint32_t feature,
                              uint32_t &feature_type,
                              uint32_t &feature_idx);

  virtual bool UpdateStatus(TreeNode *node);

  void PrepareLeaf(TreeNode *node,
                   StoredTree &tree);

  void PrepareCell(TreeNode *node,
                   StoredTree &tree);

  virtual void InsertChildNodes(TreeNode *node,
                                vector<uint32_t> &sample_ids_left,
                                vector<uint32_t> &sample_ids_right,
                                vector<uint32_t> &labels_left,
                                vector<uint32_t> &labels_right,
                                vector<uint32_t> &sample_weights_left,
                                vector<uint32_t> &sample_weights_right);

  virtual void CleanUp(StoredTree &tree);

 private:
  bool PrepareSubset(uint32_t feature_type,
                     uint32_t feature_idx,
                     TreeNode *node);

  TreeNode* LookForAncestor(uint32_t feature_idx,
                            TreeNode *node);

  void GetSortedIdxBySorting(uint32_t feature_idx,
                             TreeNode *node);

  void GetSortedIdxBySubsetting(uint32_t feature_idx,
                                TreeNode *node,
                                TreeNode *ancestor);

  void SubsetNumericalFeature(uint32_t feature_idx,
                              TreeNode *node);

  void SubsetDiscreteFeature(uint32_t feature_idx,
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

  void PartitionNode(const TreeNode *node,
                     vector<uint32_t> &sample_ids_left,
                     vector<uint32_t> &sample_ids_right,
                     vector<uint32_t> &labels_left,
                     vector<uint32_t> &labels_right,
                     vector<uint32_t> &sample_weights_left,
                     vector<uint32_t> &sample_weights_right);

  template <typename Comparator>
  void BatchPartition(const vector<uint32_t> &sample_ids,
                      Comparator &comparator,
                      vector<uint32_t> &sample_ids_left,
                      vector<uint32_t> &sample_ids_right,
                      vector<uint32_t> &labels_left,
                      vector<uint32_t> &labels_right,
                      vector<uint32_t> &sample_weights_left,
                      vector<uint32_t> &sample_weights_right);
};

#endif
