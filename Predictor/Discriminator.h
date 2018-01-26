
#ifndef DECISIONTREE_DISCRIMINATOR_H
#define DECISIONTREE_DISCRIMINATOR_H

#include <cstdint>
#include <vector>
#include "../Global/GlobalConsts.h"

using std::vector;

template <typename feature_t>
class ContinuousDiscriminator {
 public:
  const float &threshold;
  const vector<feature_t> &feature;
  ContinuousDiscriminator(const float &threshold,
                          const vector<feature_t> &feature):
          threshold(threshold), feature(feature) {}
  bool operator()(uint32_t sample_id) const {
    return feature[sample_id] < threshold;
  }
};

template <typename feature_t>
class OrdinalDiscriminator {
 public:
  const uint32_t &ordinal_ceiling;
  const vector<feature_t> &feature;
  OrdinalDiscriminator(const uint32_t &ordinal_ceiling,
                       const vector<feature_t> &feature):
          ordinal_ceiling(ordinal_ceiling), feature(feature) {}
  bool operator()(uint32_t sample_id) const {
    return feature[sample_id] <= ordinal_ceiling;
  }
};

template <typename feature_t>
class OneVsAllDiscriminator {
 public:
  const uint32_t &one_vs_all_feature;
  const vector<feature_t> &feature;
  OneVsAllDiscriminator(const uint32_t &one_vs_all_feature,
                        const vector<feature_t> &feature):
          one_vs_all_feature(one_vs_all_feature), feature(feature) {}
  bool operator()(uint32_t sample_id) const {
    return feature[sample_id] == one_vs_all_feature;
  }
};

template <typename feature_t>
class LowCardDiscriminator {
 public:
  const uint32_t &bitmask;
  const vector<feature_t> &feature;
  LowCardDiscriminator(const uint32_t &bitmask,
                       const vector<feature_t> &feature):
          bitmask(bitmask), feature(feature) {}
  bool operator()(uint32_t sample_id) const {
    return ((1u << feature[sample_id]) & bitmask) != 0;
  }
};

template<>
class LowCardDiscriminator<float> {
 public:
  const uint32_t &bitmask;
  const vec_flt_t &feature;
  LowCardDiscriminator(const uint32_t &bitmask,
                       const vec_flt_t &feature):
    bitmask(bitmask), feature(feature) {}
  bool operator()(uint32_t sample_id) const {
    assert(false);
    return false;
  }
};

template<>
class LowCardDiscriminator<double> {
 public:
  const uint32_t &bitmask;
  const vec_dbl_t &feature;
  LowCardDiscriminator(const uint32_t &bitmask,
                       const vec_dbl_t &feature):
    bitmask(bitmask), feature(feature) {}
  bool operator()(uint32_t sample_id) const {
    assert(false);
    return false;
  }
};

template <typename feature_t>
class HighCardDiscriminator {
 public:
  const vec_uint32_t &bitmask;
  const vector<feature_t> &feature;
  HighCardDiscriminator(const vec_uint32_t &bitmask,
                        const vector<feature_t> &feature):
          bitmask(bitmask), feature(feature) {}
  bool operator()(uint32_t sample_id) const {
    uint32_t mask_idx = feature[sample_id] >> GetMaskIdx;
    uint32_t mask_shift = feature[sample_id] & GetMaskShift;
    return ((1u << mask_shift) & bitmask[mask_idx]) != 0;
  }
};

template<>
class HighCardDiscriminator<float> {
 public:
  const vec_uint32_t &bitmask;
  const vec_flt_t &feature;
  HighCardDiscriminator(const vec_uint32_t &bitmask,
                        const vec_flt_t &feature):
    bitmask(bitmask), feature(feature) {}
  bool operator()(uint32_t sample_id) const {
    assert(false);
    return false;
  }
};

template<>
class HighCardDiscriminator<double> {
 public:
  const vec_uint32_t &bitmask;
  const vec_dbl_t &feature;
  HighCardDiscriminator(const vec_uint32_t &bitmask,
                        const vec_dbl_t &feature):
    bitmask(bitmask), feature(feature) {}
  bool operator()(uint32_t sample_id) const {
    assert(false);
    return false;
  }
};

#endif
