
#ifndef DECISIONTREE_RANDOM_H
#define DECISIONTREE_RANDOM_H

#include <random>
#include <vector>
#include "../Generics/Generics.h"

namespace Random {

using std::uniform_int_distribution;
using std::vector;

extern std::mt19937 random_generator;

void Init(uint32_t random_state);
void SampleWithReplacement(uint32_t n,
                           uint32_t k,
                           vec_uint32_t &histogram);

static void PartialShuffle(uint32_t n,
                           uint32_t k,
                           vec_uint32_t &target) {
  if (n == k) return;
  uniform_int_distribution<uint32_t> distribution(0, UINT32_MAX);
  for (uint32_t idx = 0; idx != k; ++idx) {
    uint32_t next_random = (distribution(random_generator) % n) + idx;
    uint32_t temp = target[idx];
    target[idx] = target[next_random];
    target[next_random] = temp;
    --n;
  }
}
} // namespace Random

#endif
