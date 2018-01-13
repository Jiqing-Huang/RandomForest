
#ifndef DECISIONTREE_GLOBALCONSTS_H
#define DECISIONTREE_GLOBALCONSTS_H

#include <cstdint>

// Float point error
static const double FloatError = 1e-10;

// Performance tuning paramters
static const float SubsetToSortRatio = 4.0;
static const float MemorySavingFactor = 3;
static const uint32_t MaxNumSampleForSerialBuild = 10000;
static const uint32_t MaxNumSampleForSerialSplit = 30000;
static const uint32_t MaxNumBinsForBruteSplitter = 8;
static const uint32_t MaxNumBinsForSampling = 16;

// SplitInfo bitmask
static const uint32_t IsContinuous = 0x80000000;
static const uint32_t IsOrdinal = 0x40000000;
static const uint32_t IsOneVsAll = 0x20000000;
static const uint32_t IsManyVsMany = 0x10000000;
static const uint32_t IsLowCardinality = 0x10000000;
static const uint32_t IsHighCardinality = 0x08000000;
static const uint32_t IsUnused = 0x04000000;
static const uint32_t IsLeaf = 0x02000000;
static const uint32_t GetFeatureIdx = 0x00ffffff;
static const uint32_t GetFeatureType = 0xff000000;
static const uint32_t GetMaskIdx = 5;
static const uint32_t GetMaskShift = 31;
static const uint32_t NumBitsPerWord = 32;

// tree node bitmask
static const uint32_t IsRoot = 0x80000000;
static const uint32_t IsLeftChild = 0x40000000;
static const uint32_t IsRightChild = 0x20000000;
static const uint32_t IsParallelBuilding = 0x10000000;
static const uint32_t IsParallelSplitting = 0x08000000;

// Cost Function ID
static const uint32_t UndefinedCost = 0x0;
static const uint32_t EntropyCost = 0x1;
static const uint32_t GiniCost = 0x2;

// job type flag
static const uint32_t AllFinished = UINT32_MAX;
static const uint32_t Idle = UINT32_MAX - 1;
static const uint32_t ToSplitRawNode = UINT32_MAX - 2;
static const uint32_t ToSplitProcessedNode = UINT32_MAX - 3;

#endif
