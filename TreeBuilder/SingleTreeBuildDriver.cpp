
#include "SingleTreeBuildDriver.h"

SingleTreeBuildDriver::SingleTreeBuildDriver(uint32_t cost_function,
                                             uint32_t min_leaf_node,
                                             uint32_t min_split_node,
                                             uint32_t num_features_for_split,
                                             uint32_t random_state,
                                             uint32_t num_workers,
                                             uint32_t max_num_nodes,
                                             uint32_t max_depth):
  builder(cost_function, min_leaf_node, min_split_node, max_depth, max_num_nodes, num_features_for_split, random_state),
  num_workers(num_workers), dataset(nullptr), tree(nullptr), finish(false) {}

void SingleTreeBuildDriver::LoadDataset(const Dataset *dataset) {
  this->dataset = dataset;
}

void SingleTreeBuildDriver::LoadTree(StoredTree *tree) {
  this->tree = tree;
}

void SingleTreeBuildDriver::Build() {
  builder.LoadDataSet(dataset);
  JobQueue<Job> &jobs = JobQueue<Job>::GetInstance();
  jobs.Offer(Job::SetupRootJob(0));

  std::vector<std::thread> threads;
  for (uint32_t idx = 0; idx != num_workers; ++idx)
    threads.emplace_back(std::thread(&SingleTreeBuildDriver::Run, this));

  std::unique_lock<std::mutex> lock(finish_mut);
  while (!finish)
    cv_finish.wait(lock);

  for (uint32_t idx = 0; idx != num_workers; ++idx)
    threads[idx].join();
}

void SingleTreeBuildDriver::Run() {
  Job job = Job::IdleJob();
  JobQueue<Job> &jobs = JobQueue<Job>::GetInstance();
  while (!finish) {
    if (job.type == Job::SetupRoot) {
      jobs.Offer(Job::InitSplitJob(builder.SetupRoot()));
    } else if (job.type == Job::InitSplit) {
      switch (builder.InitSplit(job.node)) {
        case Job::MakeLeaf:
          MakeLeafAndCheck(job, jobs);
          break;
        case Job::FindSplitOnAllFeatures:
          builder.FindSplitOnAllFeatures(job.node);
          SplitOrMakeLeaf(job, jobs);
          break;
        case Job::FindSplitOnOneFeature: {
          auto feature_iters = builder.GetFeatureSet();
          for (auto iter = feature_iters.first; iter != feature_iters.second; ++iter)
            jobs.Offer(Job::FindSplitOnOneFeatureJob(job.node, *iter));
        }
      }
    } else if (job.type == Job::FindSplitOnOneFeature) {
      builder.FindSplitOnOneFeature(job.feature_idx, job.node);
      if (builder.FindSplitFinished(job.node)) {
        std::unique_lock<std::mutex> lock(update_mut);
        if (builder.FindSplitFinished(job.node)) {
          job.node->Split()->FinishUpdate();
          job.node->Split()->num_updates = 0;
          SplitOrMakeLeaf(job, jobs);
        }
      }
    } else if (job.type == Job::DoSplit) {
      if (builder.DoSplit(job.node)) {
        jobs.Offer(Job::InitSplitJob(job.node->Left()));
        jobs.Offer(Job::InitSplitJob(job.node->Right()));
      } else {
        MakeLeafAndCheck(job, jobs);
      }
    } else if (job.type == Job::WriteToTree) {
      builder.WriteToTree(tree);
      jobs.SetFinish();
    }
    finish = jobs.Poll(job);
  }
  cv_finish.notify_one();
}

void SingleTreeBuildDriver::MakeLeafAndCheck(const Job &job,
                                             JobQueue<Job> &jobs) {
  if (builder.MakeLeaf(job.node))
    jobs.Offer(Job::WriteToTreeJob(0));
}

void SingleTreeBuildDriver::SplitOrMakeLeaf(const Job &job,
                                            JobQueue<Job> &jobs) {
  if (job.node->Split()->type == IsLeaf) {
    MakeLeafAndCheck(job, jobs);
  } else {
    jobs.Offer(Job::DoSplitJob(job.node));
  }
}
