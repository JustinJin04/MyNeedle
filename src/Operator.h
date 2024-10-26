#pragma once
#include "shared_ptr_wrapper.h"
#include "Tensor.h"
#include "NDArray.h"
#include "Tuple.h"

class Tensor;

class Operator{
public:
  virtual NDArray compute(const Tuple<NDArray>& inputs) = 0;

  virtual Tuple<Tensor> gradient(Tensor& out_grad, Tensor& node) = 0;

  virtual Tensor operator()(const Tuple<Tensor>& inputs) = 0;
};