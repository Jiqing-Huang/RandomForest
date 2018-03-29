
#include <boost/variant.hpp>

#include "NodeStats.h"
#include "../Dataset/Dataset.h"
#include "../Dataset/Subdataset.h"
#include "../Util/Maths.h"
#include "../Util/Cost.h"

void NodeStats::SetClassificationStats(const Subdataset *subset,
                                       const Dataset *dataset) {
  histogram = boost::apply_visitor([&dataset, &subset] (const auto &labels) {
    return Maths::BuildHistogram(labels, subset->SampleWeights(), dataset->ClassWeights());
  }, subset->Labels());
  wnum_samples = accumulate(histogram.cbegin(), histogram.cend(), 0.0);
  cost = Cost::Cost(histogram);
}

void NodeStats::SetRegressionStats(const Subdataset *subset,
                                   const Dataset *dataset) {
  num_samples = accumulate(subset->SampleWeights().cbegin(), subset->SampleWeights().cend(), 0u);
  sum = boost::apply_visitor([&subset] (const auto &labels) {
    return Maths::Sum(labels, subset->SampleWeights());
  }, subset->Labels());
  square_sum = boost::apply_visitor([&subset] (const auto &labels) {
    return Maths::SquareSum(labels, subset->SampleWeights());
  }, subset->Labels());
  cost = Cost::Cost(sum, square_sum, num_samples);
}