
#include "Splitter.h"
#include "SplitterImpl.h"

/// Implementation of class Splitter, delegate calls to SplitterImpl

Splitter::Splitter(uint32_t cost_function) {
  if (cost_function == GiniImpurity) {
    spliiter = std::make_unique<GiniSplitter>();
  } else if (cost_function == Entropy) {
    spliiter = std::make_unique<EntropySplitter>();
  } else if (cost_function == Variance) {
    spliiter = std::make_unique<VarianceSplitter>();
  }
}

Splitter::~Splitter() = default;

void Splitter::Init(uint32_t num_threads,
                    const Dataset *dataset,
                    const TreeParams &params) {
  spliiter->Init(num_threads, dataset, params);
}

void Splitter::CleanUp() {
  spliiter->CleanUp();
}

void Splitter::Split(uint32_t feature_idx,
                     uint32_t feature_type,
                     const Dataset *dataset,
                     TreeNode *node) {
  spliiter->Split(feature_idx, feature_type, dataset, node);
}