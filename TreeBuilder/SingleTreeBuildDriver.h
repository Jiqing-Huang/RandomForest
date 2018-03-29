
#ifndef DECISIONTREE_TREEBUILDDRIVER_H
#define DECISIONTREE_TREEBUILDDRIVER_H

#include <cstdint>
#include <thread>
#include <condition_variable>
#include "TreeBuilder.h"
#include "../Parallel/JobQueue.h"

class SingleTreeBuildDriver {
 public:
  SingleTreeBuildDriver(uint32_t cost_function,
                        uint32_t min_leaf_node,
                        uint32_t min_split_node,
                        uint32_t num_features_for_split,
                        uint32_t random_state,
                        uint32_t num_workers,
                        uint32_t max_num_nodes,
                        uint32_t max_depth);
  void LoadDataset(const Dataset *dataset);
  void LoadTree(StoredTree *tree);
  void Build();
  void Run();
 private:
  TreeBuilder builder;
  uint32_t num_workers;
  const Dataset *dataset;
  StoredTree *tree;

  bool finish;

  std::mutex update_mut;
  std::mutex finish_mut;
  std::condition_variable cv_finish;

  void MakeLeafAndCheck(const Job &job,
                        JobQueue<Job> &jobs);
  void SplitOrMakeLeaf(const Job &job, JobQueue<Job> &jobs);
};

#endif
