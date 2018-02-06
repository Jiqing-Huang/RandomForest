
#ifndef DECISIONTREE_BASESPLITTER_H
#define DECISIONTREE_BASESPLITTER_H

#include <memory>

class BaseSplitterImpl;
class Dataset;
class TreeParams;
class TreeNode;

/// Pimpl class of SplitterImpl

class Splitter {
 public:
  /// Construct the right SplitterImplementor based on the cost function
  explicit Splitter(uint32_t cost_function);
  ~Splitter();
  Splitter(const Splitter &splitter) = delete;
  Splitter(Splitter &&splitter) = delete;
  Splitter &operator=(const Splitter &splitter) = delete;
  Splitter &operator=(Splitter &&splitter) = delete;
  void Init(uint32_t num_threads,
            const Dataset *dataset,
            const TreeParams &params);
  void CleanUp();
  void Split(uint32_t feature_idx,
             uint32_t feature_type,
             const Dataset *dataset,
             TreeNode *node);
 private:
  std::unique_ptr<BaseSplitterImpl> spliiter;
};

#endif
