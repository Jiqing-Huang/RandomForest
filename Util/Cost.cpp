
#include "Cost.h"

namespace Cost {

uint32_t cost_function;
double multiplier;
vector<double> nlogn_table;

void Init(uint32_t cost_function) {
  Cost::cost_function = cost_function;
}

void ConstructNLogNTable(const double multiplier,
                         uint32_t upper_bound) {
  Cost::multiplier = multiplier;
  nlogn_table.reserve(upper_bound);
  nlogn_table.push_back(0.0);
  for (uint32_t idx = 1; idx != upper_bound; ++idx) {
    const double x = static_cast<double>(idx) / multiplier;
    nlogn_table.push_back(x * log2(x));
  }
}

void ExtendNLogNTable(uint32_t upper_bound) {
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

