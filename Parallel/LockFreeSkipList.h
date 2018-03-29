
#ifndef DECISIONTREE_LOCKFREESKIPLIST_H
#define DECISIONTREE_LOCKFREESKIPLIST_H

#include <vector>
#include <memory>
#include <random>
#include "AtomicMarkablePtr.h"
#include "Job.h"
#include "SkipListNode.h"

template <typename ItemType>
class LockFreeSkipList {

using Node = SkipListNode<ItemType>;

 public:
  LockFreeSkipList():
    head(Node::GetHead()), tail(Node::GetTail()), rand_gen(), dist(0, Node::InverseProb - 1) {
    for (uint32_t level = 0; level <= Node::MaxLevel; ++level)
      head->next[level] = tail;
  }

  bool Insert(const ItemType &item) {
    uint32_t top_level = RandomLevel();
    std::vector<Node *> preds(Node::MaxLevel + 1, nullptr);
    std::vector<Node *> succs(Node::MaxLevel + 1, nullptr);
    while (!Find(item, preds, succs)) {
      auto *node = Node::GetNode(item, top_level);
      for (uint32_t level = 0; level <= top_level; ++level)
         node->next[level] = succs[level];
      if (!preds[0]->next[0].CompareAndSet(succs[0], node, false, false))
        continue;
      for (uint32_t level = 1; level <= top_level; ++level)
          while (!preds[level]->next[level].CompareAndSet(succs[level], node, false, false))
            Find(item, preds, succs);
      return true;
    }
    return false;
  }

  bool Erase(const ItemType &item) {
    std::vector<Node *> preds(Node::MaxLevel + 1, nullptr);
    std::vector<Node *> succs(Node::MaxLevel + 1, nullptr);
    while (Find(item, preds, succs)) {
      Node *node = succs[0];
      for (uint32_t level = node->top_level; level != 0; --level) {
        bool mark = false;
        auto *succ = node->next[level].Get(mark);
        while (!mark) {
          node->next[level].CompareAndSet(succ, succ, false, true);
          succ = node->next[level].Get(mark);
        }
      }
      bool mark = false;
      auto *succ = node->next[0].Get(mark);
      while (true) {
        bool i_am_the_marker = node->next[0].CompareAndSet(succ, succ, false, true);
        succ = node->next[0].Get(mark);
        if (i_am_the_marker) {
          Find(item, preds, succs);
          return true;
        } else if (mark) {
          return false;
        }
      }
    }
    return false;
  }

  bool Poll(ItemType &item) {
    Node *node = GetAndMarkFirst();
    if (node) {
      item = node->item;
      Erase(node->item);
      return true;
    } else {
      return false;
    }
  }

  bool Contains(const ItemType &item) {
    bool mark = false;
    Node *pred = head, *curr = nullptr, *succ = nullptr;
    for (uint32_t level = Node::MaxLevel; level != UINT32_MAX; --level) {
      curr = pred->next[level].Get();
      while (true) {
        succ = curr->next[level].Get(mark);
        while (mark) {
          curr = curr->next[level].Get();
          succ = curr->next[level].Get(mark);
        }
        if (curr->item < item) {
          pred = curr;
          curr = succ;
        } else {
          break;
        }
      }
    }
    return curr->item == item;
  }

 private:
  Node *head;
  Node *tail;
  std::mt19937 rand_gen;
  std::uniform_int_distribution<uint32_t> dist;

  bool Find(const ItemType &item,
            std::vector<Node *> &preds,
            std::vector<Node *> &succs) {
    bool mark = false;
    Node *pred = nullptr, *curr = nullptr, *succ = nullptr;
    while (true) {
      bool retry = false;
      pred = head;
      for (uint32_t level = Node::MaxLevel; level != UINT32_MAX; --level) {
        curr = pred->next[level].Get();
        while (true) {
          succ = curr->next[level].Get(mark);
          while (mark) {
            bool snip = pred->next[level].CompareAndSet(curr, succ, false, false);
            if (!snip) {
              retry = true;
              break;
            }
            curr = pred->next[level].Get();
            succ = curr->next[level].Get(mark);
          }
          if (!retry && curr->item < item) {
            pred = curr;
            curr = succ;
          } else {
            break;
          }
        }
        if (retry) break;
        preds[level] = pred;
        succs[level] = curr;
      }
      if (!retry)
        return curr->item == item;
    }
  }

  uint32_t RandomLevel() {
    uint32_t level = 0;
    while (level < Node::MaxLevel) {
      if (dist(rand_gen) != 0) return level;
      ++level;
    }
    return level;
  }

  Node *GetAndMarkFirst() {
    Node *curr = head->next[0].Get();
    while (curr != tail)
      if (!curr->marked) {
        bool mark = false;
        if (curr->marked.compare_exchange_strong(mark, true, std::memory_order_seq_cst, std::memory_order_seq_cst))
          return curr;
      } else {
        curr = curr->next[0].Get();
      }
    return nullptr;
  }
};

#endif
