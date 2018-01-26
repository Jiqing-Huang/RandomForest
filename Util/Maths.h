
#ifndef DECISIONTREE_MATHS_H
#define DECISIONTREE_MATHS_H

#include <vector>
#include <cstdint>
#include <cmath>
#include <cstring>
#include "../Global/GlobalConsts.h"
#include "../Generics/Generics.h"

using std::vector;
using std::accumulate;
using std::fill;

class Maths {
 public:
  template <typename source_t, typename target_t>
  static vector<target_t> CastAndCopy(const vector<source_t> &source) {
    vector<target_t> target;
    target.reserve(source.size());
    for (const auto &value: source)
      target.push_back(static_cast<target_t>(value));
    return target;
  };

  template <typename target_t>
  struct CastAndCopyVisitor: public boost::static_visitor<vector<target_t>> {
    template <typename source_t>
    vector<target_t> operator()(const vector<source_t> &source) {
      return CastAndCopy<source_t, target_t>(source);
    }
  };

  template <typename label_t, typename class_weight_t>
  static vector<class_weight_t> BuildHistogram(const vector<label_t> &labels,
                                               const vec_uint32_t &sample_weights,
                                               const vector<class_weight_t> &class_weights) {
    vector<class_weight_t> histogram(class_weights.size(), 0);
    for (uint32_t idx = 0; idx != labels.size(); ++idx) {
      label_t label = labels[idx];
      histogram[label] += sample_weights[idx] * class_weights[label];
    }
    return histogram;
  }

  struct BuildHistogramVisitor: public boost::static_visitor<generic_vec_t> {
    const vec_uint32_t &sample_weights;
    explicit BuildHistogramVisitor(const vec_uint32_t &sample_weights):
      sample_weights(sample_weights) {}

    template <typename label_t, typename class_weight_t>
    vector<class_weight_t> operator()(const vector<label_t> &labels,
                                      const vector<class_weight_t> &class_weights) {
      return BuildHistogram(labels, sample_weights, class_weights);
    }
  };

  struct GenericAccumulateVisitor: public boost::static_visitor<generic_t> {
    template <typename data_t>
    data_t operator()(const vector<data_t> &vec) {
      return accumulate(vec.cbegin(), vec.cend(), static_cast<data_t>(0));
    }
  };

  struct DoubleAccumulateVisitor: public boost::static_visitor<double> {
    template <typename data_t>
    double operator()(const vector<data_t> &vec) {
      return static_cast<double>(accumulate(vec.cbegin(), vec.cend(), static_cast<data_t>(0)));
    }
  };

  template <typename label_t>
  static double Sum(const vector<label_t> &labels,
                    const vec_uint32_t &sample_weights) {
    double sum = 0.0;
    for (uint32_t idx = 0; idx != sample_weights.size(); ++idx)
      sum += sample_weights[idx] * labels[idx];
    return sum;
  }

  struct SumVisitor: public boost::static_visitor<double> {
    const vec_uint32_t &sample_weights;
    explicit SumVisitor(const vec_uint32_t &sample_weights):
      sample_weights(sample_weights) {}

    template <typename label_t>
    double operator()(const vector<label_t> &labels) {
      return Sum(labels, sample_weights);
    }
  };

  template <typename label_t>
  static double SquareSum(const vector<label_t> &labels,
                          const vec_uint32_t &sample_weights) {
    double sum = 0.0;
    for (uint32_t idx = 0; idx != sample_weights.size(); ++idx)
      sum += sample_weights[idx] * labels[idx] * labels[idx];
    return sum;
  }

  struct SquareSumVisitor: public boost::static_visitor<double> {
    const vec_uint32_t &sample_weights;
    explicit SquareSumVisitor(const vec_uint32_t &sample_weights):
      sample_weights(sample_weights) {}

    template <typename label_t>
    double operator()(const vector<label_t> &labels) {
      return SquareSum(labels, sample_weights);
    }
  };

  static void Normalize(vector<float> &histogram) {
    float sum = accumulate(histogram.begin(), histogram.end(), 0.0f);
    for (auto &height: histogram)
      height /= sum;
  }

  static uint32_t Argmax(const vector<float> &histogram) {
    uint32_t ret = 0;
    float max = 0.0;
    for (uint32_t idx = 0; idx != histogram.size(); ++idx)
      if (histogram[idx] > max) {
        max = histogram[idx];
        ret = idx;
      }
    return ret;
  }

  template<typename data_t>
  static void VectorAddInPlace(uint32_t num_classes,
                               const vector<data_t> &source,
                               uint32_t offset,
                               vector<data_t> &target) {
    for (uint32_t idx = 0; idx != num_classes; ++idx)
      target[idx] += source[offset + idx];
  }

  template<typename data_t>
  static void VectorMinusInPlace(uint32_t num_classes,
                                 const vector<data_t> &source,
                                 uint32_t offset,
                                 vector<data_t> &target) {
    for (uint32_t idx = 0; idx != num_classes; ++idx)
      target[idx] -= source[offset + idx];
  }

  template<typename data_t>
  static void VectorAddOutOfPlace(uint32_t num_classes,
                                  const vector<data_t> &lhs,
                                  const vector<data_t> &rhs,
                                  uint32_t offset,
                                  vector<data_t> &target) {
    for (uint32_t idx = 0; idx != num_classes; ++idx)
      target[idx] = lhs[idx] + rhs[offset + idx];
  }

  template<typename data_t>
  static void VectorMinusOutOfPlace(uint32_t num_classes,
                                    const vector<data_t> &lhs,
                                    const vector<data_t> &rhs,
                                    uint32_t offset,
                                    vector<data_t> &target) {
    for (uint32_t idx = 0; idx != num_classes; ++idx)
      target[idx] = lhs[idx] - rhs[offset + idx];
  }

  template<typename data_t>
  static void VectorMove(uint32_t num_classes,
                         const vector<data_t> &amount,
                         const uint32_t offset,
                         vector<data_t> &move_from,
                         vector<data_t> &move_to) {
    for (uint32_t idx = 0; idx != num_classes; ++idx) {
      move_from[idx] -= amount[offset + idx];
      move_to[idx] += amount[offset + idx];
    }
  }

  static uint32_t NextToFlip(uint32_t n) {
    return static_cast<uint32_t>(ffs(n) - 1);
  }
};
#endif
