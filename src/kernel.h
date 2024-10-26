#pragma once

#include <vector>

void cpu_add(const float* a, const float* b, int size, float* out);
void cpu_mul(const float* a, const float* b, int size, float* out);

void cpu_compact(const float* in_ptr,
const std::vector<int>& shape, const std::vector<int>& strides, int offset, float* out_ptr);