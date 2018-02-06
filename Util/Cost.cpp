
#include "Cost.h"

namespace Cost {

uint32_t cost_function;
double multiplier = 0.0;
vec_dbl_t nlogn_table;

void Init(const uint32_t cost_function,
          const vec_dbl_t &class_weights,
          const double wnum_samples) {
  Cost::cost_function = cost_function;
  if (cost_function == Entropy) {
    if (multiplier == 0.0) {
      ConstructNLogNTable(class_weights, wnum_samples);
    } else {
      ExtendNLogNTable(wnum_samples);
    }
  }
}

void ConstructNLogNTable(const vec_dbl_t &class_weights,
                         const double wnum_samples) {
  multiplier = 0.0;
  bool valid_multiplier = false;
  while (!valid_multiplier) {
    multiplier += 1.0;
    valid_multiplier = true;
    for (const auto &class_weight: class_weights) {
      double approximated_class_weight = round(class_weight * multiplier) / multiplier;
      double error = fabs(approximated_class_weight - class_weight);
      if (error > FloatError) {
        valid_multiplier = false;
        break;
      }
    }
  }
  uint32_t upper_bound = static_cast<uint32_t>(0.5 + wnum_samples * multiplier) + 1;
  nlogn_table.reserve(upper_bound);
  nlogn_table.push_back(0.0);
  for (uint32_t idx = 1; idx != upper_bound; ++idx) {
    const double x = static_cast<double>(idx) / multiplier;
    nlogn_table.push_back(x * log2(x));
  }
}

void ExtendNLogNTable(const double wnum_samples) {
  uint32_t upper_bound = static_cast<uint32_t>(0.5 + wnum_samples * multiplier) + 1;
  if (upper_bound <= nlogn_table.size()) return;
  nlogn_table.reserve(upper_bound);
  for (uint32_t idx = nlogn_table.size(); idx != upper_bound; ++idx) {
    const double x = static_cast<double>(idx) / multiplier;
    nlogn_table.push_back(x * log2(x));
  }
}

void CleanUp() {
  nlogn_table.clear();
  nlogn_table.shrink_to_fit();
}

} // namespace Cost

