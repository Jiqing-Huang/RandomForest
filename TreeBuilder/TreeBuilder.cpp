
#include "TreeBuilder.h"

using std::accumulate;
using std::fill;
using std::sort;
using std::iota;
using std::make_unique;

#include <iostream>
using std::cout;
using std::endl;

void TreeBuilder::LoadDataSet(const Dataset &dataset) {
  this->dataset = &dataset;
  params.GetMaxNums(dataset);
  util.BindToDataset(dataset);
  feature_set.resize(dataset.num_features);
  iota(feature_set.begin(), feature_set.end(), 0);
}

void TreeBuilder::UpdateDataSet() {
  params.GetMaxNums(*dataset);
  util.UpdateDataset(*dataset);
}

void TreeBuilder::UnloadDataSet() {
  dataset = nullptr;
  params.ResetMaxNums();
  util.UnbindFromDataset();
  feature_set.clear();
  feature_set.shrink_to_fit();
}

void TreeBuilder::LoadPresortedIndices(const vector<vector<uint32_t>> &presorted_indices) {
  this->presorted_indices = (presorted_indices.empty())? nullptr : &presorted_indices;
}

void TreeBuilder::UnloadPresortedIndices() {
  presorted_indices = nullptr;
}

void TreeBuilder::Build(StoredTree &tree) {
  if (params.cost_function == GiniCost)
    splitter_ptr.gini_splitter->Init(1, *dataset, params, util);
  if (params.cost_function == EntropyCost)
    splitter_ptr.entropy_splitter->Init(1, *dataset, params, util);
  tree.Init(dataset->num_numerical_features, dataset->num_discrete_features, 1);
  SetupRoot(tree);
  BuildAllNodes(0, tree);
  CleanUp(tree);
}

void TreeBuilder::CleanUp(StoredTree &tree) {
  tree.NormalizeFeatureImportance(util, dataset->num_numerical_features, dataset->num_discrete_features);
  tree.ShrinkToFit();
  nodes.clear();
  nodes.shrink_to_fit();
  node_top = 0;
  finish = false;
}

void TreeBuilder::SetupRoot(StoredTree &tree) {
  tree.Resize(params.max_num_cell, params.max_num_leaf);
  nodes.resize(params.max_num_node);
  nodes[node_top++] = make_unique<TreeNode>(*dataset, util);
}

void TreeBuilder::BuildAllNodes(uint32_t node_id,
                                StoredTree &tree) {
  TreeNode *root = nodes[node_id].get();
  TreeNode *node = root;

  while (!root->right_child_processed) {

    if (tree.max_depth < node->depth) tree.max_depth = node->depth;

    if (!node->left_child_processed) {
      node->GetStats(*dataset, params, util);
      bool splittable = node->stats->cost > FloatError &&
                        node->stats->num_samples >= params.min_split_node &&
                        node->depth < params.max_depth;
      if (splittable) {
        GetBestSplit(node);
        if (node->split_info->type == IsLeaf)
          splittable = false;
      }
      if (!splittable) {
        PrepareLeaf(node, tree);
      } else {
        PrepareCell(node, tree);
        node = node->left;
      }
    } else if (!node->right_child_processed) {
      node = node->right;
    } else {
      node = node->parent;
    }
  }
}

void TreeBuilder::GetBestSplit(TreeNode *node) {
  node->GetEmptySplitInfo();
  util.SampleWithoutReplacement(dataset->num_features, params.num_features_for_split, feature_set);
  uint32_t feature_type, feature_idx;
  for (uint32_t idx = 0; idx != params.num_features_for_split; ++idx) {
    uint32_t feature = feature_set[idx];
    GetFeatureTypeAndIndex(feature, feature_type, feature_idx);
    if (params.cost_function == GiniCost) {
      GiniSplitter &splitter = *splitter_ptr.gini_splitter;
      SplitOnFeature(feature_type, feature_idx, node, splitter);
    }
    if (params.cost_function == EntropyCost) {
      EntropySplitter &splitter = *splitter_ptr.entropy_splitter;
      SplitOnFeature(feature_type, feature_idx, node, splitter);
    }
  }
  node->split_info->FinishUpdate();
}

void TreeBuilder::GetFeatureTypeAndIndex(uint32_t feature,
                                         uint32_t &feature_type,
                                         uint32_t &feature_idx) {
  if (feature < dataset->num_numerical_features) {
    feature_idx = feature;
    feature_type = IsContinuous;
  } else {
    feature_idx = feature - dataset->num_numerical_features;
    feature_type = (*dataset->discrete_feature_types)[feature_idx];
  }
}

template <typename Splitter_Type>
void TreeBuilder::SplitOnFeature(uint32_t feature_type,
                                 uint32_t feature_idx,
                                 TreeNode *node,
                                 Splitter_Type &splitter) {
  bool to_delelte_after_use = PrepareSubset(feature_type, feature_idx, node);
  if (feature_type == IsContinuous) {
    splitter.SplitNumerical(feature_idx, *dataset, params, util, node);
    if (to_delelte_after_use) node->DiscardSortedIdx(feature_idx);
  } else if (feature_type == IsOrdinal) {
    splitter.SplitOrdinal(feature_idx, *dataset, params, util, node);
  } else if (feature_type == IsOneVsAll) {
    splitter.SplitOneVsAll(feature_idx, *dataset, params, util, node);
  } else if (feature_type == IsManyVsMany) {
    splitter.SplitManyVsMany(feature_idx, *dataset, params, util, node);
  }
}

bool TreeBuilder::PrepareSubset(uint32_t feature_type,
                                uint32_t feature_idx,
                                TreeNode *node) {
  if (feature_type == IsContinuous) {
    SubsetNumericalFeature(feature_idx, node);
    TreeNode *ancestor = LookForAncestor(feature_idx, node);
    uint32_t size_ancestor = (ancestor)? ancestor->size : (presorted_indices)? dataset->num_samples : UINT32_MAX;
    auto max_size = static_cast<uint32_t>(node->size * log2(node->size) * SubsetToSortRatio);
    if (size_ancestor > max_size) {
      GetSortedIdxBySorting(feature_idx, node);
    } else {
      GetSortedIdxBySubsetting(feature_idx, node, ancestor);
    }
    return node->size * MemorySavingFactor >= size_ancestor;
  } else {
    SubsetDiscreteFeature(feature_idx, node);
    return false;
  }
}

TreeNode *TreeBuilder::LookForAncestor(uint32_t feature_idx,
                                       TreeNode *node) {
  TreeNode *ancestor = node->parent;
  while (ancestor) {
    if (ancestor->subset && !ancestor->subset->sorted_indices[feature_idx].empty())
      return ancestor;
    ancestor = ancestor->parent;
  }
  return nullptr;
}

void TreeBuilder::GetSortedIdxBySorting(uint32_t feature_idx,
                                        TreeNode *node) {
  node->subset->sorted_indices[feature_idx].resize(node->size);
  iota(node->subset->sorted_indices[feature_idx].begin(), node->subset->sorted_indices[feature_idx].end(), 0);
  const vector<float> &feature = node->subset->numerical_features[feature_idx];
  sort(node->subset->sorted_indices[feature_idx].begin(),
       node->subset->sorted_indices[feature_idx].end(),
       [&feature](uint32_t x, uint32_t y) {
         return feature[x] < feature[y];
       });
}

void TreeBuilder::GetSortedIdxBySubsetting(uint32_t feature_idx,
                                           TreeNode *node,
                                           TreeNode *ancestor) {

  node->subset->sorted_indices[feature_idx].reserve(node->size);
  vector<uint32_t> &target_idx = node->subset->sorted_indices[feature_idx];
  const vector<uint32_t> &source_idx = (ancestor)? ancestor->subset->sorted_indices[feature_idx] :
                                                   (*presorted_indices)[feature_idx];

  const vector<uint32_t> &superset = (ancestor)? ancestor->subset->sample_ids : nodes[0]->subset->sample_ids;
  const vector<uint32_t> &subset = node->subset->sample_ids;

  auto ancestor_num_samples = static_cast<uint32_t>(superset.size());
  vector<uint32_t> super_to_sub(ancestor_num_samples, UINT32_MAX);

  uint32_t super_idx = 0;
  for (uint32_t sub_idx = 0; sub_idx != node->size; ++sub_idx) {
    uint32_t sample_id = subset[sub_idx];
    while (superset[super_idx] != sample_id) ++super_idx;
    super_to_sub[super_idx++] = sub_idx;
  }

  for (const auto &sorted_idx: source_idx) {
    uint32_t sub_idx = super_to_sub[sorted_idx];
    if (sub_idx != UINT32_MAX) target_idx.push_back(sub_idx);
  }
}

void TreeBuilder::SubsetNumericalFeature(uint32_t feature_idx,
                                         TreeNode *node) {
  const vector<float> &source = (*dataset->numerical_features)[feature_idx];
  node->subset->numerical_features[feature_idx].reserve(node->size);
  vector<float> &target = node->subset->numerical_features[feature_idx];
  for (const auto &sample_id: node->subset->sample_ids)
    target.push_back(source[sample_id]);
}

void TreeBuilder::SubsetDiscreteFeature(uint32_t feature_idx,
                                        TreeNode *node) {
  const vector<uint32_t> &source = (*dataset->discrete_features)[feature_idx];
  node->subset->discrete_features[feature_idx].reserve(node->size);
  vector<uint32_t> &target = node->subset->discrete_features[feature_idx];
  for (const auto &sample_id: node->subset->sample_ids)
    target.push_back(source[sample_id]);
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
  node->left_child_processed = true;
  node->right_child_processed = true;

  uint32_t leaf_id = tree.num_leaf++;
  tree.leaf_probability[leaf_id].resize(dataset->num_classes);
  copy(node->stats->histogram.begin(), node->stats->histogram.end(), tree.leaf_probability[leaf_id].begin());
  util.Normalize(node->stats->weighted_num_samples, tree.leaf_probability[leaf_id]);
  tree.leaf_decision[leaf_id] = util.Argmax(tree.leaf_probability[leaf_id], dataset->num_classes);

  return leaf_id;
}

void TreeBuilder::LinkLeaf(const TreeNode *node,
                           uint32_t leaf_id,
                           StoredTree &tree) {
  if (node->type & IsLeftChild)
    tree.left[node->parent->cell_id] = -leaf_id;
  if (node->type & IsRightChild)
    tree.right[node->parent->cell_id] = -leaf_id;
};

bool TreeBuilder::UpdateStatus(TreeNode *node) {
  while (node) {
    node->DiscardSubset();
    if (node->type & IsRoot)
      return true;
    if (node->type & IsLeftChild) {
      node->parent->left_child_processed = true;
      if (!node->parent->right_child_processed)
        return false;
    }
    if (node->type & IsRightChild) {
      node->parent->right_child_processed = true;
      if (!node->parent->left_child_processed)
        return false;
    }
    node = node->parent;
  }
}

void TreeBuilder::PrepareCell(TreeNode *node,
                              StoredTree &tree) {
  tree.UpdateFeatureImportance(node);
  uint32_t cell_id = MakeCell(node, tree);
  LinkCell(node, cell_id, tree);
  vector<uint32_t> sample_ids_left, sample_ids_right;
  vector<uint32_t> labels_left, labels_right;
  vector<uint32_t> sample_weights_left, sample_weights_right;
  PartitionNode(node, sample_ids_left, sample_ids_right, labels_left, labels_right,
                sample_weights_left, sample_weights_right);
  InsertChildNodes(node, sample_ids_left, sample_ids_right, labels_left, labels_right,
                   sample_weights_left, sample_weights_right);
  node->DiscardTemporaryElements();
}

uint32_t TreeBuilder::MakeCell(TreeNode *node,
                               StoredTree &tree) {
  uint32_t cell_id = tree.num_cell++;

  uint32_t bitmask_id = 0;
  if (node->split_info->type == IsHighCardinality)
    bitmask_id = tree.num_bitmask++;

  node->cell_id = cell_id;
  uint32_t cell_type = node->split_info->type | node->split_info->feature_idx;

  StoredTree::Info info;
  if (node->split_info->type == IsContinuous) info.float_point = node->split_info->info.float_type;
  if (node->split_info->type == IsOrdinal || node->split_info->type == IsOneVsAll ||
      node->split_info->type == IsLowCardinality)
    info.integer = node->split_info->info.uint32_type;
  if (node->split_info->type == IsHighCardinality) {
    tree.bitmasks[bitmask_id] = *node->split_info->info.ptr_type;
    info.integer = bitmask_id;
  }
  tree.cell_type[cell_id] = cell_type;
  tree.cell_info[cell_id] = info;

  return cell_id;
};

void TreeBuilder::LinkCell(const TreeNode *node,
                           uint32_t treenode_id,
                           StoredTree &tree) {
  if (node->type & IsLeftChild) tree.left[node->parent->cell_id] = treenode_id;
  if (node->type & IsRightChild) tree.right[node->parent->cell_id] = treenode_id;
}

void TreeBuilder::InsertChildNodes(TreeNode *node,
                                   vector<uint32_t> &sample_ids_left,
                                   vector<uint32_t> &sample_ids_right,
                                   vector<uint32_t> &labels_left,
                                   vector<uint32_t> &labels_right,
                                   vector<uint32_t> &sample_weights_left,
                                   vector<uint32_t> &sample_weights_right) {
  uint32_t left_child_id = node_top++;
  uint32_t right_child_id = node_top++;
  nodes[left_child_id] = make_unique<TreeNode>(left_child_id, IsLeftChild, node->depth + 1, node, *dataset,
                                               move(sample_ids_left), move(labels_left), move(sample_weights_left));
  nodes[right_child_id] = make_unique<TreeNode>(right_child_id, IsRightChild, node->depth + 1, node, *dataset,
                                                move(sample_ids_right), move(labels_right), move(sample_weights_right));
  node->left = nodes[left_child_id].get();
  node->right = nodes[right_child_id].get();
}

void TreeBuilder::PartitionNode(const TreeNode *node,
                                vector<uint32_t> &sample_ids_left,
                                vector<uint32_t> &sample_ids_right,
                                vector<uint32_t> &labels_left,
                                vector<uint32_t> &labels_right,
                                vector<uint32_t> &sample_weights_left,
                                vector<uint32_t> &sample_weights_right) {
  const vector<uint32_t> &sample_ids = node->subset->sample_ids;
  sample_ids_left.reserve(node->size);
  sample_ids_right.reserve(node->size);
  labels_left.reserve(node->size);
  labels_right.reserve(node->size);
  sample_weights_left.reserve(node->size);
  sample_weights_right.reserve(node->size);
  uint32_t type = node->split_info->type;

  if (type == IsContinuous) {
    const vector<float> &feature = (*dataset->numerical_features)[node->split_info->feature_idx];
    const float threshold = node->split_info->info.float_type;
    auto comparator = [&feature, &threshold](uint32_t sample_id) {
      return feature[sample_id] < threshold;
    };
    BatchPartition(sample_ids, comparator, sample_ids_left, sample_ids_right,
                   labels_left, labels_right, sample_weights_left, sample_weights_right);
  } else if (type == IsOrdinal || type == IsOneVsAll || type == IsLowCardinality) {
    const vector<uint32_t> &feature = (*dataset->discrete_features)[node->split_info->feature_idx];
    const uint32_t threshold = node->split_info->info.uint32_type;
    if (type == IsOrdinal) {
      auto comparator = [&feature, &threshold](uint32_t sample_id) {
        return feature[sample_id] <= threshold;
      };
      BatchPartition(sample_ids, comparator, sample_ids_left, sample_ids_right,
                     labels_left, labels_right, sample_weights_left, sample_weights_right);
    } else if (type == IsOneVsAll) {
      auto comparator = [&feature, &threshold](uint32_t sample_id) {
        return feature[sample_id] == threshold;
      };
      BatchPartition(sample_ids, comparator, sample_ids_left, sample_ids_right,
                     labels_left, labels_right, sample_weights_left, sample_weights_right);
    } else {
      auto comparator = [&feature, &threshold](uint32_t sample_id) {
        return (1 << feature[sample_id]) & threshold;
      };
      BatchPartition(sample_ids, comparator, sample_ids_left, sample_ids_right,
                     labels_left, labels_right, sample_weights_left, sample_weights_right);
    }
  } else if (type == IsHighCardinality) {
    const vector<uint32_t> &feature = (*dataset->discrete_features)[node->split_info->feature_idx];
    const vector<uint32_t> &threshold = *(node->split_info->info.ptr_type);
    auto comparator = [&feature, &threshold](uint32_t sample_id) {
      uint32_t mask_idx = feature[sample_id] >> GetMaskIdx;
      uint32_t mask_shift = feature[sample_id] & GetMaskShift;
      return threshold[mask_idx] & (1 << mask_shift);
    };
    BatchPartition(sample_ids, comparator, sample_ids_left, sample_ids_right,
                   labels_left, labels_right, sample_weights_left, sample_weights_right);
  }

  sample_ids_left.shrink_to_fit();
  sample_ids_right.shrink_to_fit();
  labels_left.shrink_to_fit();
  labels_right.shrink_to_fit();
  sample_weights_left.shrink_to_fit();
  sample_weights_right.shrink_to_fit();
}

template <typename Comparator>
void TreeBuilder::BatchPartition(const vector<uint32_t> &sample_ids,
                                 Comparator &comparator,
                                 vector<uint32_t> &sample_ids_left,
                                 vector<uint32_t> &sample_ids_right,
                                 vector<uint32_t> &labels_left,
                                 vector<uint32_t> &labels_right,
                                 vector<uint32_t> &sample_weights_left,
                                 vector<uint32_t> &sample_weights_right) {
  for (const auto &sample_id: sample_ids)
    if (comparator(sample_id)) {
      sample_ids_left.push_back(sample_id);
      labels_left.push_back((*dataset->labels)[sample_id]);
      sample_weights_left.push_back((*dataset->sample_weights)[sample_id]);
    } else {
      sample_ids_right.push_back(sample_id);
      labels_right.push_back((*dataset->labels)[sample_id]);
      sample_weights_right.push_back((*dataset->sample_weights)[sample_id]);
    }
}