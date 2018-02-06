
#ifndef DECISIONTREE_COST_H
#define DECISIONTREE_COST_H

#include <algorithm>
#include "../Generics/TypeDefs.h"
#include "../Global/GlobalConsts.h"

namespace Cost {

extern uint32_t cost_function;
extern double multiplier;
extern vec_dbl_t nlogn_table;

void Init(uint32_t cost_function,
          const vec_dbl_t &class_weights,
          double wnum_samples);
void ConstructNLogNTable(const vec_dbl_t &class_weights,
                         double wnum_samples);
void ExtendNLogNTable(double wnum_samples);
void CleanUp();

static double NLogN(double x) {
  return (x > 0.0)? x * log2(x) : 0.0;
}

static double NLogN(uint32_t x) {
  return nlogn_table[x];
}

static double DeltaNLogN(double x2,
                         double x1) {
  return NLogN(x2) - NLogN(x1);
}

static double DeltaNLogN(uint32_t x2,
                         uint32_t x1) {
  return NLogN(x2) - NLogN(x1);
}

static double GiniCost(const vec_dbl_t &histogram) {
  double wnum_samples = std::accumulate(histogram.cbegin(), histogram.cend(), 0.0);
  double ret = 0.0;
  for (const auto &height: histogram)
    ret += height * (wnum_samples - height);
  ret /= wnum_samples;
  return ret;
}

static double EntropyCost(const vec_dbl_t &histogram) {
  double wnum_samples = std::accumulate(histogram.cbegin(), histogram.cend(), 0.0);
  double ret = NLogN(wnum_samples);
  for (const auto &height: histogram)
    ret -= NLogN(height);
  return ret;
}

static double Cost(const vec_dbl_t &histogram) {
  if (cost_function == GiniImpurity)
    return GiniCost(histogram);
  if (cost_function == Entropy)
    return EntropyCost(histogram);
  return 0.0;
}

static double Cost(double sum,
            double square_sum,
            double num_samples) {
  return square_sum - sum * sum / num_samples;
}

} // namespace Cost

#endif
