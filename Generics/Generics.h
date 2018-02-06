
#ifndef DECISIONTREE_GENERICS_H
#define DECISIONTREE_GENERICS_H

#include <cstdint>
#include <vector>
#include <boost/variant.hpp>

#include "TypeDefs.h"

/// Definition of generic types and a few functions to perform rounding

using generic_vec_t = boost::variant<vec_uint8_t, vec_uint16_t, vec_uint32_t, vec_flt_t, vec_dbl_t>;
using generic_t = boost::variant<uint8_t, uint16_t, uint32_t, float, double>;

using integral_vec_t = boost::variant<vec_uint8_t, vec_uint16_t, vec_uint32_t>;
using integral_t = boost::variant<uint8_t, uint16_t, uint32_t>;

using floatpoint_vec_t = boost::variant<vec_flt_t, vec_dbl_t>;
using floatpoint_t = boost::variant<float, double>;

namespace Generics {

/// type ids that match exactly the underlying types in boost::variant when which() is called
static const uint32_t UInt8Type = 0;
static const uint32_t UInt16Type = 1;
static const uint32_t UInt32Type = 2;
static const uint32_t FltType = 3;
static const uint32_t DblType = 4;

static const uint32_t VecUInt8Type = 0;
static const uint32_t VecUInt16Type = 1;
static const uint32_t VecUInt32Type = 2;
static const uint32_t VecFltType = 3;
static const uint32_t VecDblType = 4;

/// cast between two numerical types, rounding to the nearest value representable as target_t
/// casting to integral type and to float point type are handled differently

/// cast to integral type
template <typename target_t, typename source_t>
inline std::enable_if_t<std::is_integral<target_t>::value, target_t>
Round(const source_t &value) {
  return static_cast<target_t>(0.5 + value);
}

/// cast to float point type
template <typename target_t, typename source_t>
inline std::enable_if_t<!std::is_integral<target_t>::value, target_t>
Round(const source_t &value) {
  return static_cast<target_t>(value);
}

/// cast a generic variable to data_t, rounding to the nearest value representable as data_t
/// casting to integral type and to float point type are handled differently

/// cast to integral type
template <typename data_t>
inline std::enable_if_t<std::is_integral<data_t>::value, data_t>
Round(const generic_t &value) {
  switch (value.which()) {
    case UInt8Type:
      return static_cast<data_t>(boost::get<uint8_t>(value));
    case UInt16Type:
      return static_cast<data_t>(boost::get<uint16_t>(value));
    case UInt32Type:
      return static_cast<data_t>(boost::get<uint32_t>(value));
    case FltType:
      return static_cast<data_t>(0.5f + boost::get<float>(value));
    case DblType:
      return static_cast<data_t>(0.5 + boost::get<double>(value));
    default:
      /// shouldn't reach here
      assert(false);
      return 0;
  }
}

/// cast to float point type
template <typename data_t>
inline std::enable_if_t<!std::is_integral<data_t>::value, data_t>
Round(const generic_t &value) {
  switch (value.which()) {
    case UInt8Type:
      return static_cast<data_t>(boost::get<uint8_t>(value));
    case UInt16Type:
      return static_cast<data_t>(boost::get<uint16_t>(value));
    case UInt32Type:
      return static_cast<data_t>(boost::get<uint32_t>(value));
    case FltType:
      return static_cast<data_t>(boost::get<float>(value));
    case DblType:
      return static_cast<data_t>(boost::get<double>(value));
    default:
      /// shouldn't reach here
      assert(false);
      return 0;
  }
}

/// get an elemnt in a generic vector by index, and cast it to data_t
/// by rounding to the nearest value representable as data_t
/// casting to integral type and to float point type are handled differently

/// cast to integral type
template <typename data_t>
inline std::enable_if_t<std::is_integral<data_t>::value, data_t>
RoundAt(const generic_vec_t &generic_vector,
        uint32_t idx) {
  switch (generic_vector.which()) {
    case VecUInt8Type:
      return static_cast<data_t>(boost::get<vec_uint8_t>(generic_vector)[idx]);
    case VecUInt16Type:
      return static_cast<data_t>(boost::get<vec_uint16_t>(generic_vector)[idx]);
    case VecUInt32Type:
      return static_cast<data_t>(boost::get<vec_uint32_t>(generic_vector)[idx]);
    case VecFltType:
      return static_cast<data_t>(0.5f + boost::get<vec_flt_t>(generic_vector)[idx]);
    case VecDblType:
      return static_cast<data_t>(0.5 + boost::get<vec_dbl_t>(generic_vector)[idx]);
    default:
      /// shouldn't reach here
      assert(false);
      return 0;
  }
}

/// cast to float point type
template <typename data_t>
inline std::enable_if_t<!std::is_integral<data_t>::value, data_t>
RoundAt(const generic_vec_t &generic_vector,
        uint32_t idx) {
  switch (generic_vector.which()) {
    case VecUInt8Type:
      return static_cast<data_t>(boost::get<vec_uint8_t>(generic_vector)[idx]);
    case VecUInt16Type:
      return static_cast<data_t>(boost::get<vec_uint16_t>(generic_vector)[idx]);
    case VecUInt32Type:
      return static_cast<data_t>(boost::get<vec_uint32_t>(generic_vector)[idx]);
    case VecFltType:
      return static_cast<data_t>(boost::get<vec_flt_t>(generic_vector)[idx]);
    case VecDblType:
      return static_cast<data_t>(boost::get<vec_dbl_t>(generic_vector)[idx]);
    default:
      /// shouldn't reach here
      assert(false);
      return 0;
  }
}

} // namespace Generics

#endif
