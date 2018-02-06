
#include "TreeBuilder.h"
#include "../Util/Random.h"
#include "../Util/Cost.h"
#include "../Tree/StoredTree.h"

TreeBuilder::TreeBuilder(uint32_t cost_function,
                         uint32_t min_leaf_node,
                         uint32_t min_split_node,
                         uint32_t max_depth,
                         uint32_t num_features_for_split,
                         uint32_t random_state):
  params(cost_function, min_leaf_node, min_split_node, max_depth, num_features_for_split, random_state),
  dataset(nullptr), presorted_indices(nullptr), splitter(cost_function), nodes(), node_top(0), finish(false) {
  Random::Init(random_state);
}

TreeBuilder::TreeBuilder(const TreeParams &params):
  params(params), dataset(nullptr), presorted_indices(nullptr), splitter(params.cost_function), nodes(), node_top(0),
  finish(false) {
  Random::Init(params.random_state);
}

TreeBuilder::~TreeBuilder() = default;

void TreeBuilder::LoadDataSet(const Dataset &dataset) {
  this->dataset = &dataset;
  params.SetMaxNums(dataset.Meta().wnum_samples);
  if (params.cost_function == Entropy)
    Cost::Init(params.cost_function, this->dataset->ClassWeights(), this->dataset->Meta().wnum_samples);
}

void TreeBuilder::UpdateDataSet() {
  params.SetMaxNums(dataset->Meta().wnum_samples);
  if (params.cost_function == Entropy)
    Cost::Init(params.cost_function, dataset->ClassWeights(), dataset->Meta().wnum_samples);
}

void TreeBuilder::UnloadDataSet() {
  dataset = nullptr;
  params.ResetMaxNums();
  Cost::CleanUp();
}

void TreeBuilder::LoadPresortedIndices(const vec_vec_uint32_t &presorted_indices) {
  this->presorted_indices = (presorted_indices.empty())? nullptr : &presorted_indices;
}

void TreeBuilder::UnloadPresortedIndices() {
  presorted_indices = nullptr;
}

void TreeBuilder::Init(uint32_t num_threads,
                       StoredTree &tree) {
  splitter.Init(num_threads, dataset, params);
  tree.Init(*dataset, params, num_threads);

  feature_sets.resize(num_threads, vector<uint32_t>(dataset->Meta().num_features, 0));
  for (uint32_t idx = 0; idx != num_threads; ++idx)
    iota(feature_sets[idx].begin(), feature_sets[idx].end(), 0);

  nodes.resize(params.max_num_node);
}

void TreeBuilder::Build(StoredTree &tree) {
  Init(1, tree);
  nodes[node_top++] = std::make_unique<TreeNode>(dataset);
  BuildAllNodes(0, tree);
  CleanUp(tree);
}

void TreeBuilder::CleanUp(StoredTree &tree) {
  tree.CleanUp();
  splitter.CleanUp();
  nodes.clear();
  nodes.shrink_to_fit();
  node_top = 0;
  finish = false;
}

void TreeBuilder::BuildAllNodes(uint32_t node_id,
                                StoredTree &tree) {
  TreeNode *root = nodes[node_id].get();
  TreeNode *node = root;

  while (!root->RightChildProcessed()) {

    if (tree.max_depth < node->Depth())
      tree.max_depth = node->Depth();

    if (!node->LeftChildProcessed()) {
      node->SetStats(dataset, params.cost_function);
      bool splittable = Splittable(node);
      if (splittable) {
        GetBestSplit(node);
        if (node->Split()->type == IsLeaf)
          splittable = false;
      }
      if (!splittable) {
        PrepareLeaf(node, tree);
      } else {
        PrepareCell(node, tree);
        node = node->Left();
      }
    } else if (!node->RightChildProcessed()) {
      node = node->Right();
    } else {
      node = node->Parent();
    }
  }
}

bool TreeBuilder::Splittable(const TreeNode *node) {
  if (node->Stats()->Cost() <= FloatError || node->Depth() == params.max_depth) return false;
  return (params.cost_function == Variance)? node->Stats()->NumSamples() >= params.min_split_node :
                                             node->Stats()->WNumSamples() >= params.min_split_node;
}

void TreeBuilder::GetBestSplit(TreeNode *node) {
  node->InitSplitInfo();
  const vector<uint32_t> &feature_set = ShuffleFeatures(node);
  for (uint32_t idx = 0; idx != params.num_features_for_split; ++idx) {
    uint32_t feature_idx = feature_set[idx];
    Split(dataset->FeatureType(feature_idx), feature_idx, node);
  }
  node->Split()->FinishUpdate();
}

const vec_uint32_t &TreeBuilder::ShuffleFeatures(TreeNode *node) {
  uint32_t thread_id = ParallelTreeNodeNonMember::GetThreadId(0, node);
  vec_uint32_t &feature_set = feature_sets[thread_id];
  Random::PartialShuffle(dataset->Meta().num_features, params.num_features_for_split, feature_set);
  return feature_set;
}

void TreeBuilder::Split(uint32_t feature_type,
                        uint32_t feature_idx,
                        TreeNode *node) {
  bool to_delete_sorted_idx = PrepareSubset(feature_type, feature_idx, node);
  splitter.Split(feature_idx, feature_type, dataset, node);
  if (to_delete_sorted_idx) node->DiscardSortedIdx(feature_idx);
}

bool TreeBuilder::PrepareSubset(uint32_t feature_type,
                                uint32_t feature_idx,
                                TreeNode *node) {
  if (feature_type == IsContinuous) {
    TreeNode *ancestor = LookForAncestor(feature_idx, node);
    uint32_t size_ancestor = (ancestor)? ancestor->Size() : (presorted_indices)? dataset->Meta().size : UINT32_MAX;
    auto max_size = static_cast<uint32_t>(node->Size() * log2(node->Size()) * SubsetToSortRatio);
    if (size_ancestor > max_size) {
      node->Subset()->Sort(dataset, feature_idx);
    } else if (ancestor) {
      node->Subset()->Subset(ancestor->Subset(), feature_idx);
    } else {
      node->Subset()->Subset(dataset, presorted_indices, feature_idx);
    }
    return node->Size() * MemorySavingFactor >= size_ancestor;
  } else {
    node->Subset()->Gather(dataset, feature_idx);
    return false;
  }
}

TreeNode *TreeBuilder::LookForAncestor(uint32_t feature_idx,
                                       TreeNode *node) {
  TreeNode *ancestor = node->Parent();
  while (ancestor) {
    if (ancestor->Subset() && !ancestor->Subset()->Empty(feature_idx))
      return ancestor;
    ancestor = ancestor->Parent();
  }
  return nullptr;
}

void TreeBuilder::PrepareLeaf(TreeNode *node,
                              StoredTree &tree) {
  uint32_t leaf_id = MakeLeaf(node, tree);
  LinkLeaf(node, leaf_id, tree);
  node->DiscardTemporaryElements();
  node->DiscardSubset();
  finish = UpdateStatus(node);
}

uint32_t TreeBuilder::MakeLeaf(TreeNode *node,
                               StoredTree &tree) {
  node->ProcessedLeft();
  node->ProcessedRight();
  uint32_t leaf_id = tree.num_leaf++;
  tree.WriteToLeaf(node, leaf_id);
  return leaf_id;
}

void TreeBuilder::LinkLeaf(const TreeNode *node,
                           uint32_t leaf_id,
                           StoredTree &tree) {
  if (node->IsLeftChild())
    tree.left[node->Parent()->CellId()] = -leaf_id;
  if (node->IsRightChild())
    tree.right[node->Parent()->CellId()] = -leaf_id;
}

bool TreeBuilder::UpdateStatus(TreeNode *node) {
  while (node) {
    node->DiscardSubset();
    if (node->IsRoot())
      return true;
    if (node->IsLeftChild()) {
      node->Parent()->ProcessedLeft();
      if (!node->Parent()->RightChildProcessed())
        return false;
    }
    if (node->IsRightChild()) {
      node->Parent()->ProcessedRight();
      if (!node->Parent()->LeftChildProcessed())
        return false;
    }
    node = node->Parent();
  }
  return true;
}

void TreeBuilder::PrepareCell(TreeNode *node,
                              StoredTree &tree) {
  tree.UpdateFeatureImportance(node);
  uint32_t cell_id = MakeCell(node, tree);
  LinkCell(node, cell_id, tree);
  InsertChildNodes(node);
  node->DiscardTemporaryElements();
}

uint32_t TreeBuilder::MakeCell(TreeNode *node,
                               StoredTree &tree) {
  uint32_t cell_id = tree.num_cell++;

  uint32_t bitmask_id = 0;
  if (node->Split()->type == IsHighCardinality)
    bitmask_id = tree.num_bitmask++;

  node->SetCellId(cell_id);
  uint32_t cell_type = node->Split()->type | node->Split()->feature_idx;

  StoredTree::Info info;
  if (node->Split()->type == IsContinuous) info.float_point = node->Split()->info.float_type;
  if (node->Split()->type == IsOrdinal || node->Split()->type == IsOneVsAll ||
      node->Split()->type == IsLowCardinality)
    info.integer = node->Split()->info.uint32_type;
  if (node->Split()->type == IsHighCardinality) {
    tree.bitmasks[bitmask_id] = *node->Split()->info.ptr_type;
    info.integer = bitmask_id;
  }
  tree.cell_type[cell_id] = cell_type;
  tree.cell_info[cell_id] = info;

  return cell_id;
}

void TreeBuilder::LinkCell(const TreeNode *node,
                           uint32_t cell_id,
                           StoredTree &tree) {
  if (node->IsLeftChild()) tree.left[node->Parent()->CellId()] = cell_id;
  if (node->IsRightChild()) tree.right[node->Parent()->CellId()] = cell_id;
}

void TreeBuilder::InsertChildNodes(TreeNode *node) {
  uint32_t left_child_id = node_top++;
  uint32_t right_child_id = node_top++;
  nodes[left_child_id] = std::make_unique<TreeNode>(left_child_id, IsLeftChildType, node);
  nodes[right_child_id] = std::make_unique<TreeNode>(right_child_id, IsRightChildType, node);
  TreeNode *left = nodes[left_child_id].get();
  TreeNode *right = nodes[right_child_id].get();
  node->LinkChildren(left, right);
  node->PartitionSubset(dataset, left, right);
}