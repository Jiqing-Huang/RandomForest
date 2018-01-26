
#ifndef DECISIONTREE_COST_H
#define DECISIONTREE_COST_H

#include <cstdint>
#include <vector>
#include <algorithm>
#include <boost/variant/multivisitors.hpp>
#include "../Generics/Generics.h"
#include "../Global/GlobalConsts.h"

namespace Cost {

using std::vector;
using std::accumulate;

extern uint32_t cost_function;
extern double multiplier;
extern vector<double> nlogn_table;

void Init(uint32_t cost_function);
void ConstructNLogNTable(double multiplier,
                         uint32_t upper_bound);
void ExtendNLogNTable(uint32_t upper_bound);
void CleanUp();

static double NLogN(uint32_t x) {
  return nlogn_table[x];
}

static double DeltaNLogN(uint32_t x2,
                         uint32_t x1) {
  return NLogN(x2) - NLogN(x1);
}

template <typename class_weight_t>
static double GiniCost(const vector<class_weight_t> &histogram) {
  class_weight_t wnum_samples = accumulate(histogram.cbegin(), histogram.cend(), static_cast<class_weight_t>(0));
  double ret = 0.0;
  for (const auto &height: histogram)
    ret += height * (wnum_samples - height);
  ret /= wnum_samples;
  return ret;
}

template <typename class_weight_t>
static double EntropyCost(const vector<class_weight_t> &histogram) {
  class_weight_t wnum_samples = accumulate(histogram.cbegin(), histogram.cend(), static_cast<class_weight_t>(0));
  double ret = NLogN(wnum_samples);
  for (const auto &height: histogram)
    ret -= NLogN(height);
  return ret;
}

template <typename class_weight_t>
static double Cost(const vector<class_weight_t> &histogram) {
  if (cost_function == GiniImpurity)
    return GiniCost(histogram);
  if (cost_function == Entropy)
    return EntropyCost(histogram);
  assert(false);
  return 0.0;
}

struct CostVisitor: public boost::static_visitor<double> {
  template <typename class_weihgt_t>
  double operator()(const vector<class_weihgt_t> &histogram) {
    return Cost(histogram);
  }
};

static double Cost(double sum,
            double square_sum,
            double num_samples) {
  return square_sum - sum * sum / num_samples;
}

} // namespace Cost

#endif
