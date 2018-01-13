
#ifndef DECISIONTREE_SPLITINFO_H
#define DECISIONTREE_SPLITINFO_H

#include <cstdint>
#include <vector>
#include <mutex>

#include "../Global/GlobalConsts.h"

using std::vector;
using std::mutex;
using std::unique_lock;

class SplitInfo {

  union Info {
    float float_type;
    uint32_t uint32_type;
    vector<uint32_t> *ptr_type;
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

  void UpdateFloat(bool is_synchronized,
                   double gain,
                   uint32_t type,
                   uint32_t feature_idx,
                   float value) {
    if (is_synchronized) {
      SafeUpdateFloat(gain, type, feature_idx, value);
    } else {
      UnsafeUpdateFloat(gain, type, feature_idx, value);
    }
  }

  void UpdateUInt(bool is_synchronized,
                  double gain,
                  uint32_t type,
                  uint32_t feature_idx,
                  uint32_t value) {
    if (is_synchronized) {
      SafeUpdateUInt(gain, type, feature_idx, value);
    } else {
      UnsafeUpdateUInt(gain, type, feature_idx, value);
    }
  }

  void UpdatePtr(bool is_synchronized,
                 double gain,
                 uint32_t type,
                 uint32_t feature_idx,
                 const vector<uint32_t> &value) {
    if (is_synchronized) {
      SafeUpdatePtr(gain, type, feature_idx, value);
    } else {
      UnsafeUpdatePtr(gain, type, feature_idx, value);
    }
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

  mutex updating;

  void UnsafeUpdateFloat(double gain,
                         uint32_t type,
                         uint32_t feature_idx,
                         float value) {
    if (UpdateGeneral(gain, type, feature_idx))
      this->info.float_type = value;
  }

  void SafeUpdateFloat(double gain,
                       uint32_t type,
                       uint32_t feature_idx,
                       float value) {
    unique_lock<mutex> lock(updating);
    ++num_updates;
    UnsafeUpdateFloat(gain, type, feature_idx, value);
  }


  void UnsafeUpdateUInt(double gain,
                        uint32_t type,
                        uint32_t feature_idx,
                        uint32_t uint32_info) {
    if (UpdateGeneral(gain, type, feature_idx))
      this->info.uint32_type = uint32_info;
  }

  void SafeUpdateUInt(double gain,
                      uint32_t type,
                      uint32_t feature_idx,
                      uint32_t uint32_info) {
    unique_lock<mutex> lock(updating);
    ++num_updates;
    UnsafeUpdateUInt(gain, type, feature_idx, uint32_info);
  }

  void UnsafeUpdatePtr(double gain,
                       uint32_t type,
                       uint32_t feature_idx,
                       const vector<uint32_t> &categorical_bitmask) {
    if (UpdateGeneral(gain, type, feature_idx))
      this->info.ptr_type = new vector<uint32_t>(categorical_bitmask);
  }

  void SafeUpdatePtr(double gain,
                     uint32_t type,
                     uint32_t feature_idx,
                     const vector<uint32_t> &categorical_bitmask) {
    unique_lock<mutex> lock(updating);
    ++num_updates;
    UnsafeUpdatePtr(gain, type, feature_idx, categorical_bitmask);
  }

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
