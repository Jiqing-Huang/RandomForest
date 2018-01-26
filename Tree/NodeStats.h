
#ifndef DECISIONTREE_NODESTATS_H
#define DECISIONTREE_NODESTATS_H

#include <vector>
#include <cstdint>
#include "../Global/GlobalConsts.h"
#include "../Dataset/SubDataset.h"
#include "../Util/Cost.h"

using std::vector;
using std::accumulate;

class NodeStats {
 public:
  NodeStats():
    cost(0.0), wnum_samples(), histogram(), num_samples(0), sum(0.0), square_sum(0.0) {}

  double Cost() const {
    return cost;
  }

  generic_t WNumSamples() const {
    return wnum_samples;
  }

  const generic_vec_t &Histogram() const {
    return *histogram;
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

  uint32_t EffectiveNumSamples(uint32_t cost_function) const {
    return (cost_function == Variance)? num_samples : Generics::Round<uint32_t>(wnum_samples);
  }

  void SetStats(const SubDataset *subset,
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
  generic_t wnum_samples;
  unique_ptr<generic_vec_t> histogram;

  // specific to regression
  uint32_t num_samples;
  double sum;
  double square_sum;

  void SetClassificationStats(const SubDataset *subset,
                              const Dataset *dataset) {
    Maths::BuildHistogramVisitor bh_visitor(subset->SampleWeights());
    histogram = make_unique<generic_vec_t>(boost::apply_visitor(bh_visitor, subset->Labels(), dataset->ClassWeights()));
    Maths::GenericAccumulateVisitor ga_visitor;
    wnum_samples = boost::apply_visitor(ga_visitor, *histogram);
    Cost::CostVisitor c_visitor;
    cost = boost::apply_visitor(c_visitor, *histogram);
  }

  void SetRegressionStats(const SubDataset *subset,
                          const Dataset *dataset) {
    num_samples = accumulate(subset->SampleWeights().cbegin(), subset->SampleWeights().cend(), 0u);
    Maths::SumVisitor s_visitor(subset->SampleWeights());
    sum = boost::apply_visitor(s_visitor, subset->Labels());
    Maths::SquareSumVisitor ss_visitor(subset->SampleWeights());
    square_sum = boost::apply_visitor(ss_visitor, subset->Labels());
    cost = Cost::Cost(sum, square_sum, num_samples);
  }
};

#endif
