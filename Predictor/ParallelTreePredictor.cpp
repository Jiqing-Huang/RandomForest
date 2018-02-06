
#include <thread>
#include <vector>
#include <mutex>
#include "ParallelTreePredictor.h"
#include "../Dataset/Dataset.h"

vec_dbl_t ParallelTreePredictor::PredictBatchByMean(const Dataset *dataset,
                                                    const uint32_t filter) {
  uint32_t block_size = dataset->Meta().size / num_threads;
  vec_dbl_t output(dataset->Meta().size, 0.0);
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (uint32_t thread_id = 0; thread_id != num_threads; ++thread_id) {
    uint32_t start = thread_id * block_size;
    uint32_t end = (thread_id == num_threads - 1)? dataset->Meta().size : start + block_size;
    threads.emplace_back(std::thread(&ParallelTreePredictor::ParallelPredictBatchByMean,
                                     this, dataset, filter, start, end, std::ref(output)));
  }
  std::unique_lock<std::mutex> lock(main_thread);
  while (num_finished_job != num_threads)
    cv_main_thread.wait(lock);
  num_finished_job = 0;
  for (auto &thread: threads)
    thread.join();
  return output;
}

vec_vec_dbl_t ParallelTreePredictor::PredictBatchByProbability(const Dataset *dataset,
                                                               const uint32_t filter) {
  uint32_t block_size = dataset->Meta().size / num_threads;
  vec_vec_dbl_t output(dataset->Meta().size);
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (uint32_t thread_id = 0; thread_id != num_threads; ++thread_id) {
    uint32_t start = thread_id * block_size;
    uint32_t end = (thread_id == num_threads - 1)? dataset->Meta().size : start + block_size;
    threads.emplace_back(std::thread(&ParallelTreePredictor::ParallelPredictBatchByProbability,
                                     this, dataset, filter, start, end, std::ref(output)));
  }
  std::unique_lock<std::mutex> lock(main_thread);
  while (num_finished_job != num_threads)
    cv_main_thread.wait(lock);
  num_finished_job = 0;
  for (auto &thread: threads)
    thread.join();
  return output;
}

void ParallelTreePredictor::ParallelPredictBatchByMean(const Dataset *dataset,
                                                       const uint32_t filter,
                                                       const uint32_t start_idx,
                                                       const uint32_t end_idx,
                                                       vec_dbl_t &output) {
  const vec_uint32_t &sample_weights = dataset->SampleWeights();
  for (uint32_t idx = start_idx; idx != end_idx; ++idx)
    if (ToPredict(sample_weights, idx, filter))
      output[idx] = PredictOneByMean(dataset, idx);
  ++num_finished_job;
  cv_main_thread.notify_one();
}

void ParallelTreePredictor::ParallelPredictBatchByProbability(const Dataset *dataset,
                                                              const uint32_t filter,
                                                              const uint32_t start_idx,
                                                              const uint32_t end_idx,
                                                              vec_vec_dbl_t &output) {
  const vec_uint32_t &sample_weights = dataset->SampleWeights();
  for (uint32_t idx = start_idx; idx != end_idx; ++idx)
    if (ToPredict(sample_weights, idx, filter))
      output[idx] = PredictOneByProbability(dataset, idx);
  ++num_finished_job;
  cv_main_thread.notify_one();
}