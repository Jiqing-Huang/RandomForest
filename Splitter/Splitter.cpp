
#include "Splitter.h"
#include "SplitterImpl.h"

/// Implementation of class Splitter, delegate all calls to SplitterImpl

Splitter::Splitter(const Dataset *dataset,
                   const TreeParams &params) {
  if (params.cost_function == GiniImpurity) {
    spliiter = std::make_unique<GiniSplitter>(dataset, params);
  } else if (params.cost_function == Entropy) {
    spliiter = std::make_unique<EntropySplitter>(dataset, params);
  } else if (params.cost_function == Variance) {
    spliiter = std::make_unique<VarianceSplitter>(dataset, params);
  }
}

Splitter::~Splitter() = default;

void Splitter::Split(uint32_t feature_idx,
                     uint32_t feature_type,
                     const Dataset *dataset,
                     TreeNode *node) {
  spliiter->Split(feature_idx, feature_type, dataset, node);
}

Splitter &Splitter::GetInstance(const Dataset *dataset,
                                const TreeParams &params) {
  static Splitter instance(dataset, params);
  return instance;
}