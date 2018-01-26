
#ifndef DECISIONTREE_PARALLELTREEBUILDER_H
#define DECISIONTREE_PARALLELTREEBUILDER_H

#include <cstdint>
#include <cstring>
#include <mutex>
#include <condition_variable>

#include "TreeBuilder.h"
#include "../Tree/ParallelTreeNode.h"

using std::mutex;
using std::unique_lock;
using std::condition_variable;

class ParallelTreeBuilder: public TreeBuilder {

 public:
  ParallelTreeBuilder(uint32_t cost_function,
                      uint32_t min_leaf_node,
                      uint32_t min_split_node,
                      uint32_t max_depth,
                      uint32_t num_features_for_split,
                      uint32_t random_state,
                      uint32_t num_threads):
    TreeBuilder::TreeBuilder(cost_function, min_leaf_node, min_split_node, max_depth,
                             num_features_for_split, random_state),
    num_threads(num_threads), jobs(), job_begin(0), job_end(0) {};

  void Build(StoredTree &tree) override;

 protected:

  struct Job {
    uint32_t type;
    uint32_t node_id;

    Job() {
      memset(this, 0, sizeof(Job));
    }

    Job(uint32_t type,
        uint32_t node_id):
            type(type), node_id(node_id) {}

    Job(const Job &job) {
      memcpy(this, &job, sizeof(Job));
    }

    Job &operator=(const Job &job) {
      memcpy(this, &job, sizeof(Job));
      return *this;
    }

    void SetToIdle() {
      type = Idle;
      node_id = 0;
    }
  };

  uint32_t num_threads;

  vector<Job> jobs;
  std::atomic<uint32_t> job_begin;
  std::atomic<uint32_t> job_end;

  mutex main_thread;
  mutex get_job;
  mutex add_processed_node;
  mutex update_status;

  condition_variable cv_get_job;
  condition_variable cv_finish;

  void ParallelBuild(uint32_t thread_id,
                     StoredTree &tree);
  void BuildOneNode(Job &job,
                    StoredTree &tree);
  void SplitRawNode(Job &job,
                    StoredTree &tree);
  void SplitProcessedNode(Job &job,
                          StoredTree &tree);
  void AddSplitJobs(ParallelTreeNode *node);
  void ParallelSplitOnFeature(Job &job);
  void GetJob(Job &job);
  void AddJob(const Job &job_to_add);
  void AddJobs(const vector<Job> &jobs_to_add,
               uint32_t num_jobs);
  void InsertChildNodes(TreeNode *node) override;
  bool UpdateStatus(TreeNode *node) override;
  void CleanUp(StoredTree &tree) override;
};

#endif