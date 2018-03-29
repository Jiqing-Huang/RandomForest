
#include "TreeBuilder.h"
#include "../Util/Random.h"
#include "../Util/Cost.h"
#include "../Tree/StoredTree.h"
#include "../Parallel/Job.h"

TreeBuilder::TreeBuilder(uint32_t cost_function,
                         uint32_t min_leaf_node,
                         uint32_t min_split_node,
                         uint32_t max_depth,
                         uint32_t max_num_nodes,
                         uint32_t num_features_for_split,
                         uint32_t random_state):
  params(cost_function, min_leaf_node, min_split_node, max_depth, max_num_nodes, num_features_for_split, random_state),
  dataset(nullptr), presorted_indices(nullptr), root(nullptr),
  cell_count(0), leaf_count(0), finish(false) {
  Random::Init(random_state);
}

TreeBuilder::TreeBuilder(const TreeParams &params):
  params(params), dataset(nullptr), presorted_indices(nullptr), root(nullptr),
  cell_count(0), leaf_count(0), finish(false) {
  Random::Init(params.random_state);
}

TreeBuilder::~TreeBuilder() = default;

void TreeBuilder::LoadDataSet(const Dataset *dataset, const vec_vec_uint32_t *presorted_indices) {
  this->dataset = dataset;
  if (params.cost_function == Entropy)
    Cost::Init(params.cost_function, this->dataset->ClassWeights(), this->dataset->Meta().wnum_samples);
  this->presorted_indices = presorted_indices;
}

TreeNode *TreeBuilder::SetupRoot() {
  root = std::make_unique<TreeNode>(dataset);
  return root.get();
}

uint32_t TreeBuilder::InitSplit(TreeNode *node) {
  node->SetStats(dataset, params.cost_function);
  if (node->Stats()->Cost() <= FloatError || cell_count >= params.max_num_nodes || node->Depth() == params.max_depth)
    return Job::MakeLeaf;
  if (params.cost_function == Variance && node->Stats()->NumSamples() < params.min_split_node)
    return Job::MakeLeaf;
  if (params.cost_function != Variance && node->Stats()->WNumSamples() < params.min_split_node)
    return Job::MakeLeaf;
  return (node->Size() <= MaxSizeForSerialSplit)? Job::FindSplitOnAllFeatures : Job::FindSplitOnOneFeature;
}

std::pair<vec_uint32_t::iterator, vec_uint32_t::iterator> TreeBuilder::GetFeatureSet() {
  static thread_local vec_uint32_t feature_set;
  if (feature_set.empty()) {
    feature_set = std::vector<uint32_t>(dataset->Meta().num_features);
    std::iota(feature_set.begin(), feature_set.end(), 0);
  }
  Random::PartialShuffle(dataset->Meta().num_features, params.num_features_for_split, feature_set);
  return {feature_set.begin(), feature_set.begin() + params.num_features_for_split};
}

void TreeBuilder::FindSplitOnAllFeatures(TreeNode *node) {
  node->InitSplitInfo();
  const auto feature_iters = GetFeatureSet();
  for (auto iter = feature_iters.first; iter != feature_iters.second; ++iter)
    FindSplitOnOneFeature(*iter, node);
  node->Split()->FinishUpdate();
}

void TreeBuilder::FindSplitOnOneFeature(uint32_t feature_idx,
                                        TreeNode *node) {
  if (!node->Split())
    node->InitSplitInfo();
  uint32_t feature_type = dataset->FeatureType(feature_idx);
  bool to_delete_sorted_idx = PrepareSubset(feature_type, feature_idx, node);
  Splitter &splitter = Splitter::GetInstance(dataset, params);
  splitter.Split(feature_idx, feature_type, dataset, node);
  if (to_delete_sorted_idx) node->DiscardSortedIdx(feature_idx);
}

bool TreeBuilder::FindSplitFinished(TreeNode *node) {
  return node->Split()->num_updates == params.num_features_for_split;
}

bool TreeBuilder::DoSplit(TreeNode *node) {
  if (cell_count >= params.max_num_nodes) {
    node->Split()->type = IsLeaf;
    return false;
  } else {
    ++cell_count;
    node->SpawnChildren(dataset);
    node->DiscardTemporaryElements();
    return true;
  }
}

bool TreeBuilder::MakeLeaf(TreeNode *node) {
  ++leaf_count;
  node->ProcessedLeft();
  node->ProcessedRight();
  node->DiscardTemporaryElements();
  node->DiscardSubset();
  finish = UpdateStatus(node);
  return finish;
}

void TreeBuilder::WriteToTree(StoredTree *tree) {
  struct NumberedNode {
    TreeNode *node;
    int32_t id;
    int32_t parent_id;
    uint32_t next;

    NumberedNode(TreeNode *node,
                 int32_t parent_id):
      node(node), id(0), parent_id(parent_id), next(IsLeftChildType) {}
  };

  tree->Init(*dataset, cell_count, leaf_count);

  std::vector<NumberedNode> stack;
  stack.reserve(cell_count + leaf_count);

  int32_t cell_top = 0;
  int32_t leaf_top = 0;
  stack.emplace_back(root.get(), 0);

  while (!stack.empty()) {
    NumberedNode &curr = stack.back();
    if (curr.next == IsLeftChildType) {
      if (!curr.node->Split() || curr.node->Split()->type == IsLeaf) {
        tree->WriteToLeaf(curr.node, leaf_top++, curr.parent_id);
        stack.pop_back();
      } else {
        curr.id = cell_top++;
        tree->WriteToCell(curr.node, curr.id, curr.parent_id);
        curr.next = IsRightChildType;
        stack.emplace_back(curr.node->Left(), curr.id);
      }
    } else {
      TreeNode *right = curr.node->Right();
      int32_t id = curr.id;
      stack.pop_back();
      stack.emplace_back(right, id);
    }
  }

  tree->CleanUp();

  root.reset();
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

bool TreeBuilder::UpdateStatus(TreeNode *node) {
  std::unique_lock<std::mutex> lock(update_mut);
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