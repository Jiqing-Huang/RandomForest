
#ifndef DECISIONTREE_ATOMICMARKABLEPTR_H
#define DECISIONTREE_ATOMICMARKABLEPTR_H

#include <cstdint>
#include <atomic>

template <typename value_t>
class AtomicMarkablePtr {
 public:
  AtomicMarkablePtr():
    ptr(0) {}

  value_t *Get(bool &mark) const {
    uint64_t local_copy_ptr = ptr;
    mark = (local_copy_ptr & MarkMask) == TrueMark;
    return reinterpret_cast<value_t *>(local_copy_ptr & AddressMask);
  }

  value_t *Get() const {
    return reinterpret_cast<value_t *>(ptr & AddressMask);
  }

  bool GetMark() const {
    return (ptr & MarkMask) == TrueMark;
  }

  value_t &operator*() const {
    return *Get();
  }

  void Set(const value_t *desired_ptr,
           const bool mark) {
    ptr = reinterpret_cast<uint64_t>(desired_ptr) | (mark? TrueMark : FalseMark);
  }

  AtomicMarkablePtr &operator=(value_t *rhs) {
    Set(rhs, false);
    return *this;
  }

  bool CompareAndSet(const value_t *expected_ptr,
                     const value_t *desired_ptr,
                     const bool expected_mark,
                     const bool desired_mark) {
    uint64_t expected = reinterpret_cast<uint64_t>(expected_ptr) | (expected_mark? TrueMark : FalseMark);
    uint64_t desired = reinterpret_cast<uint64_t>(desired_ptr) | (desired_mark? TrueMark : FalseMark);
    return ptr.compare_exchange_strong(expected, desired, std::memory_order_seq_cst, std::memory_order_seq_cst);
  }

 private:
  static const uint64_t AddressMask = 0x0000ffffffffffff;
  static const uint64_t MarkMask = 0xffff000000000000;
  static const uint64_t TrueMark = 0x0001000000000000;
  static const uint64_t FalseMark = 0x0;
  std::atomic<uint64_t> ptr;
};


#endif
