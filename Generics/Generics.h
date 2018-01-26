
#ifndef DECISIONTREE_GENERICS_H
#define DECISIONTREE_GENERICS_H

#include <cstdint>
#include <vector>
#include <boost/variant.hpp>

using std::vector;

using vec_uint8_t = vector<uint8_t>;
using vec_uint16_t = vector<uint16_t>;
using vec_uint32_t = vector<uint32_t>;
using vec_uint64_t = vector<uint64_t>;
using vec_flt_t = vector<float>;
using vec_dbl_t = vector<double>;

using generic_vec_t = boost::variant<vec_uint8_t, vec_uint16_t, vec_uint32_t, vec_uint64_t, vec_flt_t, vec_dbl_t>;
using generic_t = boost::variant<uint8_t, uint16_t, uint32_t, uint64_t, float, double>;

namespace Generics {

static const uint32_t UInt8Type = 0;
static const uint32_t UInt16Type = 1;
static const uint32_t UInt32Type = 2;
static const uint32_t UInt64Type = 3;
static const uint32_t FltType = 4;
static const uint32_t DblType = 5;

static const uint32_t VecUInt8Type = 0;
static const uint32_t VecUInt16Type = 1;
static const uint32_t VecUInt32Type = 2;
static const uint32_t VecUInt64Type = 3;
static const uint32_t VecFltType = 4;
static const uint32_t VecDblType = 5;

template <typename data_t>
inline typename std::enable_if<std::is_integral<data_t>::value, data_t>::type At(const generic_vec_t &generic_vector,
                                                                                 uint32_t idx) {
  switch (generic_vector.which()) {
    case VecUInt8Type:
      return static_cast<data_t>(boost::get<vec_uint8_t>(generic_vector)[idx]);
    case VecUInt16Type:
      return static_cast<data_t>(boost::get<vec_uint16_t>(generic_vector)[idx]);
    case VecUInt32Type:
      return static_cast<data_t>(boost::get<vec_uint32_t>(generic_vector)[idx]);
    case VecUInt64Type:
      return static_cast<data_t>(boost::get<vec_uint64_t>(generic_vector)[idx]);
    case VecFltType:
      return static_cast<data_t>(0.5f + boost::get<vec_flt_t>(generic_vector)[idx]);
    case VecDblType:
      return static_cast<data_t>(0.5 + boost::get<vec_dbl_t>(generic_vector)[idx]);
    default:
      assert(false);
      return 0;
  }
}

template <typename data_t>
inline typename std::enable_if<!std::is_integral<data_t>::value, data_t>::type At(const generic_vec_t &generic_vector,
                                                                                  uint32_t idx) {
  switch (generic_vector.which()) {
    case VecUInt8Type:
      return static_cast<data_t>(boost::get<vec_uint8_t>(generic_vector)[idx]);
    case VecUInt16Type:
      return static_cast<data_t>(boost::get<vec_uint16_t>(generic_vector)[idx]);
    case VecUInt32Type:
      return static_cast<data_t>(boost::get<vec_uint32_t>(generic_vector)[idx]);
    case VecUInt64Type:
      return static_cast<data_t>(boost::get<vec_uint64_t>(generic_vector)[idx]);
    case VecFltType:
      return static_cast<data_t>(boost::get<vec_flt_t>(generic_vector)[idx]);
    case VecDblType:
      return static_cast<data_t>(boost::get<vec_dbl_t>(generic_vector)[idx]);
    default:
      assert(false);
      return 0;
  }
}

template <typename data_t>
inline typename std::enable_if<std::is_integral<data_t>::value, data_t>::type Round(const generic_t &value) {
  switch (value.which()) {
    case UInt8Type:
      return static_cast<data_t>(boost::get<uint8_t>(value));
    case UInt16Type:
      return static_cast<data_t>(boost::get<uint16_t>(value));
    case UInt32Type:
      return static_cast<data_t>(boost::get<uint32_t>(value));
    case UInt64Type:
      return static_cast<data_t>(boost::get<uint64_t>(value));
    case FltType:
      return static_cast<data_t>(0.5f + boost::get<float>(value));
    case DblType:
      return static_cast<data_t>(0.5 + boost::get<double>(value));
    default:
      assert(false);
      return 0;
  }
}

template <typename data_t>
inline typename std::enable_if<!std::is_integral<data_t>::value, data_t>::type Round(const generic_t &value) {
  switch (value.which()) {
    case UInt8Type:
      return static_cast<data_t>(boost::get<uint8_t>(value));
    case UInt16Type:
      return static_cast<data_t>(boost::get<uint16_t>(value));
    case UInt32Type:
      return static_cast<data_t>(boost::get<uint32_t>(value));
    case UInt64Type:
      return static_cast<data_t>(boost::get<uint64_t>(value));
    case FltType:
      return static_cast<data_t>(boost::get<float>(value));
    case DblType:
      return static_cast<data_t>(boost::get<double>(value));
    default:
      assert(false);
      return 0;
  }
}

template <typename return_t, typename class_t, typename method_t, typename ...typed_t>
struct GenericVisitor: public boost::static_visitor<return_t> {

  class_t &object;
  method_t method;
  std::tuple<typed_t &&...> typed_params;

  GenericVisitor(class_t &object,
                 method_t method,
                 typed_t &&...typed_params):
    object(object), method(method), typed_params(std::forward<typed_t &&...>(typed_params...)) {};

  template <typename generic_t, size_t ...S>
  return_t Invoker(const generic_t &generic_param, std::index_sequence<S...>) {
    return (object.*method)(generic_param, std::get<S>(typed_params)...);
  };

  template <typename generic_t>
  return_t operator()(const generic_t &vec) {
    return Invoker(vec, std::make_index_sequence<sizeof...(typed_t)> {});
  }
};

}

#endif //DECISIONTREE_GENERICS_H
