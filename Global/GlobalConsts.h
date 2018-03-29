
#ifndef DECISIONTREE_GLOBALCONSTS_H
#define DECISIONTREE_GLOBALCONSTS_H

#include <cstdint>

/// Compile-time Constants

/// Float point error
static const double FloatError = 1e-10;

/// Performance tuning paramters
/// Preference of subsetting numerical feature over sorting
static const float SubsetToSortRatio = 4.0;

/// Preference of memory saving at the expense of speed
static const float MemorySavingFactor = 3.0;

/// Threshold of switching from parallel split finding to serial split finding
static const uint32_t MaxSizeForSerialSplit = 50000;

/// Threshold of switching from brute force to heuristic to find split in many-vs-many discrete feature
static const uint32_t MaxNumBinsForBruteSplitter = 8;

/// Max number of bins to test in each step in the move-one-bin-at-a-time heuristic split finding algorithm
static const uint32_t MaxNumBinsForSampling = 16;

/// SplitInfo bit indicators
static const uint32_t IsContinuous = 0x80000000;
static const uint32_t IsOrdinal = 0x40000000;
static const uint32_t IsOneVsAll = 0x20000000;
static const uint32_t IsManyVsMany = 0x10000000;
static const uint32_t IsLowCardinality = 0x10000000;
static const uint32_t IsHighCardinality = 0x08000000;
static const uint32_t IsUnused = 0x04000000;
static const uint32_t IsLeaf = 0x02000000;

/// Bitmask to get split info in stored tree
static const uint32_t GetFeatureIdx = 0x00ffffff;
static const uint32_t GetFeatureType = 0xff000000;
static const uint32_t GetMaskIdx = 5;
static const uint32_t GetMaskShift = 31;
static const uint32_t NumBitsPerWord = 32;

/// tree node type bit indicators
static const uint32_t IsRootType = 0x80000000;
static const uint32_t IsLeftChildType = 0x40000000;
static const uint32_t IsRightChildType = 0x20000000;

/// Cost Function ID
static const uint32_t Entropy = 1;
static const uint32_t GiniImpurity = 2;
static const uint32_t Variance = 3;

/// Predict Options
static const uint32_t PredictAll = 0;
static const uint32_t PredictPresent = 1;
static const uint32_t PredictAbsent = 2;

#endif
