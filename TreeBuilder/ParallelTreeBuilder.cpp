
#include "ParallelTreeBuilder.h"
#include "../Tree/StoredTree.h"

#include <thread>

void ParallelTreeBuilder::Build(StoredTree &tree) {
  Init(num_threads, tree);
  nodes[node_top++] = std::make_unique<ParallelTreeNode>(dataset);

  jobs.resize(params.max_num_node * (params.num_features_for_split + 2));
  jobs[job_end++] = Job(ToSplitRawNode, 0);
  vector<std::thread> thread_pool;
  thread_pool.reserve(num_threads);
  for (uint32_t thread_id = 0; thread_id != num_threads; ++thread_id)
    thread_pool.emplace_back(std::thread(&ParallelTreeBuilder::ParallelBuild, this, thread_id, std::ref(tree)));
  std::unique_lock<std::mutex> lock(main_thread);
  while (!finish)
    cv_finish.wait(lock);
  for (auto &thread: thread_pool)
    thread.join();
  CleanUp(tree);
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
      node->SetParallelBuilding(thread_id);
      SplitRawNode(job, tree);
    } else if (job.type == ToSplitProcessedNode) {
      node->SetParallelBuilding(thread_id);
      SplitProcessedNode(job, tree);
    } else {
      node->SetParallelSplitting(thread_id, job.type);
      ParallelSplitOnFeature(job);
    }
  }
  cv_get_job.notify_all();
  cv_finish.notify_one();
}

void ParallelTreeBuilder::SplitRawNode(Job &job,
                                       StoredTree &tree) {
  if (nodes[job.node_id]->Size() <= MaxSizeForSerialBuild) {
    BuildAllNodes(job.node_id, tree);
    job.SetToIdle();
  } else {
    BuildOneNode(job, tree);
  }
}

void ParallelTreeBuilder::BuildOneNode(Job &job,
                                       StoredTree &tree) {

  auto *node = dynamic_cast<ParallelTreeNode *>(nodes[job.node_id].get());
  if (tree.max_depth < node->Depth())
    tree.max_depth = node->Depth();

  node->SetStats(dataset, params.cost_function);
  bool splittable = Splittable(node);
  if (splittable) {
    if (node->Size() > MaxSizeForSerialSplit) {
      AddSplitJobs(node);
      job.SetToIdle();
      return;
    }
    GetBestSplit(node);
  } else {
    node->InitSplitInfo();
    node->Split()->type = IsLeaf;
  }
  SplitProcessedNode(job, tree);
}

void ParallelTreeBuilder::SplitProcessedNode(Job &job,
                                             StoredTree &tree) {
  TreeNode *node = nodes[job.node_id].get();
  if (node->Split()->type == IsLeaf) {
    PrepareLeaf(node, tree);
    job.SetToIdle();
  } else {
    PrepareCell(node, tree);
    if (node->Left()->Size() > node->Right()->Size()) {
      AddJob(Job(ToSplitRawNode, node->Right()->NodeId()));
      job = Job(ToSplitRawNode, node->Left()->NodeId());
    } else {
      AddJob(Job(ToSplitRawNode, node->Left()->NodeId()));
      job = Job(ToSplitRawNode, node->Right()->NodeId());
    }
  }
}

void ParallelTreeBuilder::AddSplitJobs(ParallelTreeNode *node) {
  node->InitParallelSplitting(dataset->Meta().num_features);
  node->InitSplitInfo();
  const vec_uint32_t &feature_set = ShuffleFeatures(node);
  vector<Job> jobs_to_add;
  jobs_to_add.reserve(params.num_features_for_split);
  for (uint32_t idx = 0; idx != params.num_features_for_split; ++idx)
    jobs_to_add.emplace_back(feature_set[idx], node->NodeId());
  AddJobs(jobs_to_add, params.num_features_for_split);
}

void ParallelTreeBuilder::ParallelSplitOnFeature(Job &job) {
  TreeNode *node = nodes[job.node_id].get();
  uint32_t feature_idx = job.type;
  Split(dataset->FeatureType(feature_idx), feature_idx, node);
  unique_lock<mutex> lock(add_processed_node);
  if (node->Split() && node->Split()->num_updates == params.num_features_for_split) {
    node->Split()->num_updates = 0;
    node->Split()->FinishUpdate();
    AddJob(Job(ToSplitProcessedNode, job.node_id));
  }
  lock.unlock();
  job.SetToIdle();
}

void ParallelTreeBuilder::InsertChildNodes(TreeNode *node) {
  uint32_t left_child_id = node_top++;
  uint32_t right_child_id = node_top++;
  auto *parallel_node = dynamic_cast<ParallelTreeNode*>(node);
  assert(parallel_node);
  nodes[left_child_id] =
    std::make_unique<ParallelTreeNode>(left_child_id, IsLeftChildType | IsParallelBuildingType, parallel_node);
  nodes[right_child_id] =
    std::make_unique<ParallelTreeNode>(right_child_id, IsRightChildType | IsParallelBuildingType, parallel_node);
  auto *left = dynamic_cast<ParallelTreeNode*>(nodes[left_child_id].get());
  auto *right = dynamic_cast<ParallelTreeNode*>(nodes[right_child_id].get());
  node->LinkChildren(left, right);
  node->PartitionSubset(dataset, left, right);
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