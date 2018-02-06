
#ifndef DECISIONTREE_PARALLELTREEPREDICTOR_H
#define DECISIONTREE_PARALLELTREEPREDICTOR_H

#include <mutex>
#include <condition_variable>
#include <atomic>
#include "TreePredictor.h"

class ParallelTreePredictor: public TreePredictor {
 public:
  ParallelTreePredictor(uint32_t num_threads):
    TreePredictor::TreePredictor(), num_threads(num_threads), num_finished_job(0) {}
  vec_dbl_t PredictBatchByMean(const Dataset *dataset,
                               const uint32_t filter) override;
  vec_vec_dbl_t PredictBatchByProbability(const Dataset *dataset,
                                          const uint32_t filter) override;

 private:
  uint32_t num_threads;
  std::mutex main_thread;
  std::condition_variable cv_main_thread;
  std::atomic<uint32_t> num_finished_job;
  void ParallelPredictBatchByMean(const Dataset *dataset,
                                  const uint32_t filter,
                                  const uint32_t start_idx,
                                  const uint32_t end_idx,
                                  vec_dbl_t &output);
  void ParallelPredictBatchByProbability(const Dataset *dataset,
                                         const uint32_t filter,
                                         const uint32_t start_idx,
                                         const uint32_t end_idx,
                                         vec_vec_dbl_t &output);
};

#endif
