
#ifndef DECISIONTREE_SKIPLISTNODE_H
#define DECISIONTREE_SKIPLISTNODE_H

#include <cstdint>
#include <atomic>
#include <memory>
#include <deque>
#include <vector>
#include "AtomicMarkablePtr.h"

template <typename ItemType>
struct SkipListNode {
 public:
  static SkipListNode *GetHead() {
    static thread_local std::deque<std::unique_ptr<SkipListNode>> nodes;
    nodes.emplace_back(new SkipListNode(ItemType::Min(), MaxLevel));
    return nodes.back().get();
  }

  static SkipListNode *GetTail() {
    static thread_local std::deque<std::unique_ptr<SkipListNode>> nodes;
    nodes.emplace_back(new SkipListNode(ItemType::Max(), MaxLevel));
    return nodes.back().get();
  }

  static SkipListNode *GetNode(const ItemType &item,
                               uint32_t level) {
    static thread_local std::deque<std::unique_ptr<SkipListNode>> nodes;
    nodes.emplace_back(new SkipListNode(item, level));
    return nodes.back().get();
  }

  static const uint32_t MaxLevel = 7;
  static const uint32_t InverseProb = 4;

  const uint32_t top_level;
  const ItemType item;
  std::atomic<bool> marked;
  std::vector<AtomicMarkablePtr<SkipListNode>> next;

 private:
  SkipListNode(const ItemType &item,
               const uint32_t level):
    top_level(level), item(item), marked(false), next(level + 1) {}
};


#endif
