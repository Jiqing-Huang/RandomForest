
#ifndef DECISIONTREE_MATHS_H
#define DECISIONTREE_MATHS_H

#include <vector>
#include <cstdint>
#include <cmath>
#include <boost/variant.hpp>
#include "../Generics/TypeDefs.h"
#include "../Generics/Generics.h"

namespace Maths {

template<typename source_t, typename target_t>
static std::vector<target_t> CastAndCopy(const std::vector<source_t> &source) {
  std::vector<target_t> target;
  target.reserve(source.size());
  for (const auto &value: source)
    target.push_back(static_cast<target_t>(value));
  return target;
};

template<typename target_t>
struct CastAndCopyVisitor : public boost::static_visitor<std::vector<target_t>> {
  template<typename source_t>
  std::vector<target_t> operator()(const std::vector<source_t> &source) {
    return CastAndCopy<source_t, target_t>(source);
  }
};

template<typename target_t>
static std::vector<target_t> ScaleAndCast(const vec_dbl_t &histogram,
                                          const double scaler) {
  std::vector<target_t> scaled_histogram;
  scaled_histogram.reserve(histogram.size());
  for (const auto &height: histogram)
    scaled_histogram.push_back(Generics::Round<target_t, double>(height * scaler));
  return scaled_histogram;
}

template<typename label_t>
static vec_dbl_t BuildHistogram(const std::vector<label_t> &labels,
                                const vec_uint32_t &sample_weights,
                                const vec_dbl_t &class_weights) {
  vec_dbl_t histogram(class_weights.size(), 0.0);
  for (uint32_t idx = 0; idx != labels.size(); ++idx) {
    label_t label = labels[idx];
    histogram[label] += sample_weights[idx] * class_weights[label];
  }
  return histogram;
}

template<typename label_t>
static double Sum(const std::vector<label_t> &labels,
                  const vec_uint32_t &sample_weights) {
  double sum = 0.0;
  for (uint32_t idx = 0; idx != sample_weights.size(); ++idx)
    sum += sample_weights[idx] * labels[idx];
  return sum;
}

template<typename label_t>
static double SquareSum(const std::vector<label_t> &labels,
                        const vec_uint32_t &sample_weights) {
  double sum = 0.0;
  for (uint32_t idx = 0; idx != sample_weights.size(); ++idx)
    sum += sample_weights[idx] * labels[idx] * labels[idx];
  return sum;
}

static void Normalize(vec_dbl_t &histogram) {
  double sum = accumulate(histogram.begin(), histogram.end(), 0.0);
  if (sum < FloatError) return;
  for (auto &height: histogram)
    height /= sum;
}

static uint32_t Argmax(const vec_dbl_t &histogram) {
  uint32_t ret = 0;
  double max = 0.0;
  for (uint32_t idx = 0; idx != histogram.size(); ++idx)
    if (histogram[idx] > max) {
      max = histogram[idx];
      ret = idx;
    }
  return ret;
}

}
#endif
