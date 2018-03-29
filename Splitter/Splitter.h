
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
  static Splitter &GetInstance(const Dataset *dataset,
                               const TreeParams &params);
  ~Splitter();
  Splitter(const Splitter &splitter) = delete;
  Splitter(Splitter &&splitter) = delete;
  Splitter &operator=(const Splitter &splitter) = delete;
  Splitter &operator=(Splitter &&splitter) = delete;
  void Split(uint32_t feature_idx,
             uint32_t feature_type,
             const Dataset *dataset,
             TreeNode *node);
 private:
  std::unique_ptr<BaseSplitterImpl> spliiter;

  Splitter(const Dataset *datatset,
           const TreeParams &params);
};

#endif
