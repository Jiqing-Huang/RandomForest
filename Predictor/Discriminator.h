
#ifndef DECISIONTREE_DISCRIMINATOR_H
#define DECISIONTREE_DISCRIMINATOR_H

#include "../Global/GlobalConsts.h"
#include "../Generics/TypeDefs.h"

/// Discriminators that decide which path a sample should go through a tree node
/// One of these is selected at runtime based on split type

template <typename feature_t, std::enable_if_t<!std::is_integral<feature_t>::value, void*> = nullptr>
class ContinuousDiscriminator {
 public:
  const float &threshold;
  const std::vector<feature_t> &feature;
  ContinuousDiscriminator(const float &threshold,
                          const std::vector<feature_t> &feature):
          threshold(threshold), feature(feature) {}
  bool operator()(uint32_t sample_id) const {
    return feature[sample_id] < threshold;
  }
};

template <typename feature_t, std::enable_if_t<std::is_integral<feature_t>::value, void*> = nullptr>
class OrdinalDiscriminator {
 public:
  const uint32_t &ordinal_ceiling;
  const std::vector<feature_t> &feature;
  OrdinalDiscriminator(const uint32_t &ordinal_ceiling,
                       const std::vector<feature_t> &feature):
          ordinal_ceiling(ordinal_ceiling), feature(feature) {}
  bool operator()(uint32_t sample_id) const {
    return feature[sample_id] <= ordinal_ceiling;
  }
};

template <typename feature_t, std::enable_if_t<std::is_integral<feature_t>::value, void*> = nullptr>
class OneVsAllDiscriminator {
 public:
  const uint32_t &one_vs_all_feature;
  const std::vector<feature_t> &feature;
  OneVsAllDiscriminator(const uint32_t &one_vs_all_feature,
                        const std::vector<feature_t> &feature):
          one_vs_all_feature(one_vs_all_feature), feature(feature) {}
  bool operator()(uint32_t sample_id) const {
    return feature[sample_id] == one_vs_all_feature;
  }
};

template <typename feature_t, std::enable_if_t<std::is_integral<feature_t>::value, void*> = nullptr>
class LowCardDiscriminator {
 public:
  const uint32_t &bitmask;
  const std::vector<feature_t> &feature;
  LowCardDiscriminator(const uint32_t &bitmask,
                       const std::vector<feature_t> &feature):
          bitmask(bitmask), feature(feature) {}
  bool operator()(uint32_t sample_id) const {
    return ((1u << feature[sample_id]) & bitmask) != 0;
  }
};

template <typename feature_t, std::enable_if_t<std::is_integral<feature_t>::value, void*> = nullptr>
class HighCardDiscriminator {
 public:
  const vec_uint32_t &bitmask;
  const std::vector<feature_t> &feature;
  HighCardDiscriminator(const vec_uint32_t &bitmask,
                        const std::vector<feature_t> &feature):
          bitmask(bitmask), feature(feature) {}
  bool operator()(uint32_t sample_id) const {
    uint32_t mask_idx = feature[sample_id] >> GetMaskIdx;
    uint32_t mask_shift = feature[sample_id] & GetMaskShift;
    return ((1u << mask_shift) & bitmask[mask_idx]) != 0;
  }
};

#endif
