
#ifndef DECISIONTREE_SPLITINFO_H
#define DECISIONTREE_SPLITINFO_H

#include <cstdint>
#include <vector>
#include <mutex>

#include "../Global/GlobalConsts.h"
#include "../Generics/TypeDefs.h"

class SplitInfo {

  union Info {
    float float_type;
    uint32_t uint32_type;
    vec_uint32_t *ptr_type;
  };

 public:

  uint32_t type;
  uint32_t feature_idx;
  double gain;
  Info info;
  uint32_t num_updates;

  SplitInfo():
          type(IsUnused), feature_idx(0), gain(0.0), info(), num_updates(0) {};

  ~SplitInfo() {
    if (type == IsHighCardinality) {
      delete info.ptr_type;
      info.ptr_type = nullptr;
    }
  };

  void UpdateFloat(double gain,
                   uint32_t type,
                   uint32_t feature_idx,
                   float value) {
    std::unique_lock<std::mutex> lock(updating);
    ++num_updates;
    if (UpdateGeneral(gain, type, feature_idx))
      this->info.float_type = value;
  }

  void UpdateUInt(double gain,
                  uint32_t type,
                  uint32_t feature_idx,
                  uint32_t uint32_info) {
    std::unique_lock<std::mutex> lock(updating);
    ++num_updates;
    if (UpdateGeneral(gain, type, feature_idx))
      this->info.uint32_type = uint32_info;
  }

  void UpdatePtr(double gain,
                 uint32_t type,
                 uint32_t feature_idx,
                 const vec_uint32_t &categorical_bitmask) {
    std::unique_lock<std::mutex> lock(updating);
    ++num_updates;
    if (UpdateGeneral(gain, type, feature_idx))
      this->info.ptr_type = new vec_uint32_t(categorical_bitmask);
  }

  void FinishUpdate() {
    if (gain < FloatError) type = IsLeaf;
  }

  void Clear() {
    if (type == IsHighCardinality) {
      delete info.ptr_type;
      info.ptr_type = nullptr;
    }
    this->type = IsUnused;
    this->feature_idx = 0;
    this->gain = 0.0;
  }

 private:
  std::mutex updating;

  bool UpdateGeneral(double gain,
                     uint32_t type,
                     uint32_t feature_idx) {
    if (gain - this->gain < FloatError) return false;
    if (this->type == IsHighCardinality) {
      delete info.ptr_type;
      info.ptr_type = nullptr;
    }
    this->gain = gain;
    this->type = type;
    this->feature_idx = feature_idx;
    return true;
  }
};

#endif
