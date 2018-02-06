
#ifndef DECISIONTREE_INDEXEDFEATURE_H
#define DECISIONTREE_INDEXEDFEATURE_H

#include <cstdint>
#include <memory>
#include <cstring>

/// A compact struct of indexed feature used for sorting

template <typename feature_t>
struct IndexedFeature {
  feature_t feature;
  uint32_t idx;

  IndexedFeature() = default;

  IndexedFeature(feature_t feature,
                 uint32_t idx):
    feature(feature), idx(idx) {}

  IndexedFeature(const IndexedFeature &indexed_feature) {
    std::memcpy(this, &indexed_feature, sizeof(IndexedFeature));
  }

  IndexedFeature(IndexedFeature &&indexed_feature) {
    std::memcpy(this, &indexed_feature, sizeof(IndexedFeature));
  }

  ~IndexedFeature() = default;

  IndexedFeature &operator=(const IndexedFeature &indexed_feature) {
    std::memcpy(this, &indexed_feature, sizeof(IndexedFeature));
    return *this;
  }

  IndexedFeature &operator=(IndexedFeature &&indexed_feature) {
    std::memcpy(this, &indexed_feature, sizeof(IndexedFeature));
    return *this;
  }

  bool operator<(const IndexedFeature &indexed_feature) {
    return this->feature < indexed_feature.feature;
  }
};

#endif
