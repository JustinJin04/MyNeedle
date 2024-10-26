#include"kernel.h"
#include<iostream>
#include<functional>

void cpu_add(const float* a, const float* b,int size, float* out){
  for(int i=0;i<size;++i){
    out[i] = a[i] + b[i];
  }
}

void cpu_mul(const float* a, const float* b,int size, float* out){
  for(int i=0;i<size;++i){
    out[i] = a[i] * b[i];
  }
}

void cpu_compact(const float* in_ptr, const std::vector<int>& shape, const std::vector<int>& strides, int offset, float* out_ptr){
  
  //TODO: 复习lambda表达式的scope
  std::function<void(const float* in_ptr, int in_idx, float* out_ptr, int& out_idx, const std::vector<int>& shape,
  const std::vector<int>& strides, int ndims)>dfs_compact;
  dfs_compact = [&](const float* in_ptr, int in_idx, float* out_ptr, int& out_idx, const std::vector<int>& shape,
  const std::vector<int>& strides, int ndims){
    if(ndims >= shape.size()){
      out_ptr[out_idx] = in_ptr[in_idx];
      out_idx ++;
      return;
    }
    else{
      for(int i=0;i<shape[ndims];++i){
        dfs_compact(in_ptr, in_idx + i*strides[ndims], out_ptr, out_idx, shape, strides, ndims+1);
      }
    }
  };

  int out_idx = 0;
  dfs_compact(in_ptr, offset, out_ptr, out_idx, shape, strides, 0);

}