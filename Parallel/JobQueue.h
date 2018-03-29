
#ifndef DECISIONTREE_SYNCJOBQUEUE_H
#define DECISIONTREE_SYNCJOBQUEUE_H

#include <cstdint>
#include "../Generics/TypeDefs.h"
#include "LockFreeSkipList.h"
#include "Job.h"

template <typename JobType>
class JobQueue {
 public:
  static JobQueue &GetInstance() {
    static JobQueue<JobType> instance;
    return instance;
  }

  void Offer(const JobType &job) {
    jobs.Insert(job);
  }

  bool Poll(JobType &output) {
    while (!jobs.Poll(output) && !finished)
      Backoff();
    return finished;
  }

  void SetFinish() {
    finished = true;
  }

 private:
  LockFreeSkipList<JobType> jobs;
  bool finished;

  explicit JobQueue():
    jobs(), finished(false) {}

  void Backoff() {
    // do nothing
  }
};

#endif
