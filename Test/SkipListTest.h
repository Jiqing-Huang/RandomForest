
#ifndef DECISIONTREE_SKIPLISTTEST_H
#define DECISIONTREE_SKIPLISTTEST_H

#include <thread>
#include <iostream>
#include "../Parallel/Job.h"
#include "../Parallel/LockFreeSkipList.h"

class SkipListTest {
 public:
  SkipListTest():
    list() {}

  void Start(uint32_t num_threads) {
    std::vector<std::thread> threads;
    for (int i = 0; i != num_threads; ++i) {
      threads.emplace_back(&SkipListTest::Run, this, i);
    }
    for (int i = 0; i != num_threads; ++i) {
      threads[i].join();
    }
    assert(totat_take == total_give);
  }

  void Run(uint32_t thread_id) {
    uint32_t give = 0;
    uint32_t take = 0;
    uint32_t n = 1000000;
    for (uint32_t i = n - 1; i != UINT32_MAX; --i) {
      uint32_t tree_id = i + thread_id * n;
      give += tree_id;
      bool inserted = list.Insert(Job::WriteToTreeJob(tree_id));
      assert(inserted);
    }
    for (uint32_t i = 0; i != n; ++i) {
      Job job = Job::IdleJob();
      assert(list.Poll(job));
      take += job.tree_id;
    }
    total_give += give;
    totat_take += take;
  }

 private:
  LockFreeSkipList<Job> list;
  std::atomic<uint32_t> total_give;
  std::atomic<uint32_t> totat_take;
};

#endif
