
#ifndef DECISIONTREE_METADATA_H
#define DECISIONTREE_METADATA_H

#include <cstdint>
#include "../Generics/TypeDefs.h"

/// metadata for dataset
struct MetaData {
  MetaData():
    size(0), num_samples(0), num_features(0), num_bins(), max_num_bins(0), num_classes(0), wnum_samples(0) {};

  ///////////////////////////////////////////////////////
  /// common to both regression and classification dataset

  /// length of the input data, not necessarily the number of training / testing samples
  uint32_t size;

  /// actual number of samples weighted on sample_weights
  uint32_t num_samples;

  /// number of features
  uint32_t num_features;

  /// ith element of this vector represents the cardinality of the ith feature if it is discrete, 0 if numerical
  vec_uint32_t num_bins;

  /// max cardinality of all discrete features
  uint32_t max_num_bins;
  ///////////////////////////////////////////////////////


  /////////////////////////////////////
  /// specific to classification dataset

  /// number of output labels
  uint32_t num_classes;

  /// number of samples weighted on both sample_weights and class_weights
  double wnum_samples;
  /////////////////////////////////////
};


#endif
