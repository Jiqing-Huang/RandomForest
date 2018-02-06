
#ifndef DECISIONTREE_NODESTATS_H
#define DECISIONTREE_NODESTATS_H

#include <vector>
#include <cstdint>
#include "../Generics/TypeDefs.h"
#include "../Global/GlobalConsts.h"

class Dataset;
class Subdataset;

class NodeStats {
 public:
  NodeStats():
    cost(0.0), wnum_samples(), histogram(), num_samples(0), sum(0.0), square_sum(0.0) {}

  double Cost() const {
    return cost;
  }

  double WNumSamples() const {
    return wnum_samples;
  }

  const vec_dbl_t &Histogram() const {
    return histogram;
  }

  uint32_t NumSamples() const {
    return num_samples;
  }

  double Sum() const {
    return sum;
  }

  double SquareSum() const {
    return square_sum;
  }

  void SetStats(const Subdataset *subset,
                const Dataset *dataset,
                const uint32_t cost_function) {
    if (cost_function == GiniImpurity || cost_function == Entropy)
      SetClassificationStats(subset, dataset);
    if (cost_function == Variance)
      SetRegressionStats(subset, dataset);
  }

 private:
  // common
  double cost;

  // specific to classification
  double wnum_samples;
  vec_dbl_t histogram;

  // specific to regression
  uint32_t num_samples;
  double sum;
  double square_sum;

  void SetClassificationStats(const Subdataset *subset,
                              const Dataset *dataset);
  void SetRegressionStats(const Subdataset *subset,
                          const Dataset *dataset);
};

#endif
