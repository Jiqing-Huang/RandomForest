
#ifndef DECISIONTREE_NODESTATS_H
#define DECISIONTREE_NODESTATS_H

#include <vector>
#include <cstdint>
#include "../Util/Maths.h"
#include "../Dataset/Dataset.h"
#include "TreeParams.h"

using std::vector;

class NodeStats {

 public:
  uint32_t num_samples;
  double weighted_num_samples;
  uint32_t integral_num_samples;
  double cost;
  vector<double> histogram;
  vector<uint32_t> integral_histogram;

  NodeStats(uint32_t num_classes, uint32_t cost_function):
          num_samples(0), weighted_num_samples(0.0), integral_num_samples(0), cost(0.0),
          histogram(num_classes, 0.0), integral_histogram(num_classes, 0) {};

  void GetStats(const TreeParams &param,
                const Dataset &dataset,
                const vector<uint32_t> &sample_ids,
                const Maths &util) {
    num_samples = util.GetNumSamples(sample_ids, *dataset.sample_weights);
    util.GetHistogramAndSum(sample_ids, *dataset.sample_weights, *dataset.labels, *dataset.class_weights,
                            histogram, weighted_num_samples);
    if (param.cost_function == GiniCost) {
      cost = util.GetCost(histogram, weighted_num_samples);
    }
    if (param.cost_function == EntropyCost) {
      util.GetIntegralHistogramAndSum(sample_ids, *dataset.sample_weights, *dataset.labels,
                                      dataset.integral_class_weights, integral_histogram, integral_num_samples);
      cost = util.GetCost(integral_histogram, integral_num_samples);
    }
  };
};

#endif
