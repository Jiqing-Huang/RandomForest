
#include "ParallelTreeBuilder.h"

#include <thread>

using std::thread;
using std::ref;
using std::copy;
using std::make_unique;

#include <iostream>
using std::cout;
using std::endl;

void ParallelTreeBuilder::Build(StoredTree &tree) {
  if (params.cost_function == GiniCost)
    splitter_ptr.gini_splitter->Init(num_threads, *dataset, params, util);
  if (params.cost_function == EntropyCost)
    splitter_ptr.entropy_splitter->Init(num_threads, *dataset, params, util);
  tree.Init(dataset->num_numerical_features, dataset->num_discrete_features, num_threads);
  SetupRoot(tree);
  jobs.resize(params.max_num_node * (params.num_features_for_split + 2));
  jobs[job_end++] = Job(ToSplitRawNode, 0);

  vector<thread> thread_pool;
  thread_pool.reserve(num_threads);

  for (uint32_t thread_id = 0; thread_id != num_threads; ++thread_id)
    thread_pool.emplace_back(thread(&ParallelTreeBuilder::ParallelBuild, this, thread_id, ref(tree)));

  unique_lock<mutex> lock(main_thread);
  while (!finish)
    cv_finish.wait(lock);

  for (auto &thread: thread_pool)
    thread.join();

  CleanUp(tree);
}

void ParallelTreeBuilder::SetupRoot(StoredTree &tree) {
  tree.Resize(params.max_num_cell, params.max_num_leaf);
  nodes.resize(params.max_num_node);
  nodes[node_top++] = make_unique<ParallelTreeNode>(*dataset, util);
}

void ParallelTreeBuilder::CleanUp(StoredTree &tree) {
  TreeBuilder::CleanUp(tree);
  jobs.clear();
  jobs.shrink_to_fit();
  job_begin = 0;
  job_end = 0;
}

void ParallelTreeBuilder::ParallelBuild(uint32_t thread_id,
                                        StoredTree &tree) {
  Job job(Idle, 0);
  while (!finish) {
    if (job.type == Idle) GetJob(job);
    if (job.type == AllFinished) continue;
    auto *node = dynamic_cast<ParallelTreeNode *>(nodes[job.node_id].get());
    if (job.type == ToSplitRawNode) {
      node->SetToParallelBuilding(thread_id);
      SplitRawNode(job, tree);
    } else if (job.type == ToSplitProcessedNode) {
      node->SetToParallelBuilding(thread_id);
      SplitProcessedNode(job, tree);
    } else {
      node->SetToParallelSplitting(thread_id, job.type);
      ParallelSplitOnFeature(job);
    }
  }
  cv_get_job.notify_all();
  cv_finish.notify_one();
}

void ParallelTreeBuilder::SplitRawNode(Job &job,
                                       StoredTree &tree) {
  if (nodes[job.node_id]->size <= MaxNumSampleForSerialBuild) {
    BuildAllNodes(job.node_id, tree);
    job.SetToIdle();
  } else {
    BuildOneNode(job, tree);
  }
}

void ParallelTreeBuilder::BuildOneNode(Job &job,
                                       StoredTree &tree) {

  auto *node = dynamic_cast<ParallelTreeNode *>(nodes[job.node_id].get());
  if (tree.max_depth < node->depth) tree.max_depth = node->depth;

  node->GetStats(*dataset, params, util);
  bool splittable = node->stats->cost > FloatError &&
                    node->stats->num_samples >= params.min_split_node &&
                    node->depth < params.max_depth;

  if (splittable) {
    if (node->size > MaxNumSampleForSerialSplit) {
      AddSplitJobs(node);
      job.SetToIdle();
      return;
    }
    GetBestSplit(node);
  } else {
    node->GetEmptySplitInfo();
    node->split_info->type = IsLeaf;
  }
  SplitProcessedNode(job, tree);
}

void ParallelTreeBuilder::SplitProcessedNode(Job &job,
                                             StoredTree &tree) {
  TreeNode *node = nodes[job.node_id].get();
  if (node->split_info->type == IsLeaf) {
    PrepareLeaf(node, tree);
    job.SetToIdle();
  } else {
    PrepareCell(node, tree);
    if (node->left->size > node->right->size) {
      AddJob(Job(ToSplitRawNode, node->right->node_id));
      job = Job(ToSplitRawNode, node->left->node_id);
    } else {
      AddJob(Job(ToSplitRawNode, node->left->node_id));
      job = Job(ToSplitRawNode, node->right->node_id);
    }
  }
}

void ParallelTreeBuilder::AddSplitJobs(ParallelTreeNode *node) {
  node->thread_id_splitter.resize(dataset->num_features, UINT32_MAX);
  node->GetEmptySplitInfo();
  util.SampleWithoutReplacement(dataset->num_features, params.num_features_for_split, feature_set);
  vector<Job> jobs_to_add;
  jobs_to_add.reserve(params.num_features_for_split);
  for (uint32_t idx = 0; idx != params.num_features_for_split; ++idx)
    jobs_to_add.emplace_back(feature_set[idx], node->node_id);
  AddJobs(jobs_to_add, params.num_features_for_split);
}

void ParallelTreeBuilder::ParallelSplitOnFeature(Job &job) {
  TreeNode *node = nodes[job.node_id].get();
  uint32_t feature = job.type;
  uint32_t feature_type, feature_idx;
  GetFeatureTypeAndIndex(feature, feature_type, feature_idx);
  if (params.cost_function == GiniCost) {
    GiniSplitter &splitter = *splitter_ptr.gini_splitter;
    SplitOnFeature(feature_type, feature_idx, node, splitter);
  }
  if (params.cost_function == EntropyCost) {
    EntropySplitter &splitter = *splitter_ptr.entropy_splitter;
    SplitOnFeature(feature_type, feature_idx, node, splitter);
  }
  unique_lock<mutex> lock(add_processed_node);
  if (node->split_info && node->split_info->num_updates == params.num_features_for_split) {
    node->split_info->num_updates = 0;
    node->split_info->FinishUpdate();
    AddJob(Job(ToSplitProcessedNode, job.node_id));
  }
  lock.unlock();
  job.SetToIdle();
}

void ParallelTreeBuilder::InsertChildNodes(TreeNode *node,
                                           vector<uint32_t> &sample_ids_left,
                                           vector<uint32_t> &sample_ids_right,
                                           vector<uint32_t> &labels_left,
                                           vector<uint32_t> &labels_right,
                                           vector<uint32_t> &sample_weights_left,
                                           vector<uint32_t> &sample_weights_right) {
  uint32_t left_child_id = node_top++;
  uint32_t right_child_id = node_top++;
  auto *parallel_node = dynamic_cast<ParallelTreeNode *>(node);
  nodes[left_child_id] = make_unique<ParallelTreeNode>(left_child_id, IsLeftChild | IsParallelBuilding, node->depth + 1,
                                                       parallel_node, *dataset, move(sample_ids_left),
                                                       move(labels_left), move(sample_weights_left));
  nodes[right_child_id] = make_unique<ParallelTreeNode>(right_child_id, IsRightChild | IsParallelBuilding, node->depth + 1,
                                                        parallel_node, *dataset, move(sample_ids_right),
                                                        move(labels_right), move(sample_weights_right));
  node->left = nodes[left_child_id].get();
  node->right = nodes[right_child_id].get();
}

void ParallelTreeBuilder::GetJob(Job &job) {
  unique_lock<mutex> lock(get_job);
  while (job_begin == job_end && !finish)
    cv_get_job.wait(lock);
  job = (finish)? Job(AllFinished, 0) : jobs[job_begin++];
}

void ParallelTreeBuilder::AddJob(const Job &job_to_add) {
  jobs[job_end++] = job_to_add;
  cv_get_job.notify_one();
}

void ParallelTreeBuilder::AddJobs(const vector<Job> &jobs_to_add,
                                  uint32_t num_jobs) {
  uint32_t new_job_end = (job_end += num_jobs);
  copy(jobs_to_add.begin(), jobs_to_add.end(), jobs.begin() + new_job_end - num_jobs);
  cv_get_job.notify_all();
}

bool ParallelTreeBuilder::UpdateStatus(TreeNode *node) {
  unique_lock<mutex> lock(update_status);
  return TreeBuilder::UpdateStatus(node);
}