
#include "Random.h"

namespace Random {

std::mt19937 random_generator;

void Init(uint32_t random_state) {
  random_generator.seed(random_state);
}

void SampleWithReplacement(uint32_t n,
                           uint32_t k,
                           vec_uint32_t &histogram) {
  std::uniform_int_distribution<uint32_t> distribution(0, n - 1);
  for (uint32_t i = 0; i != k; ++i) {
    uint32_t next_random = distribution(random_generator);
    ++histogram[next_random];
  }
}
} // namespace Random
