#pragma once

#include "../Operator.h"

class EWiseAdd: public Operator{
public:
  NDArray compute(const Tuple<NDArray>& inputs){
    assert(inputs.size() == 2);
    return inputs[0] + inputs[1];
  }

  Tuple<Tensor> gradient(Tensor& out_grad, Tensor& node){
    return Tuple<Tensor>({out_grad, out_grad}, true);
  }

  Tensor operator()(const Tuple<Tensor>& inputs)override{
    return Tensor((*this), inputs);
  }

};

class EWiseMul: public Operator{
public:
  NDArray compute(const Tuple<NDArray>& inputs){
    assert(inputs.size() == 2);
    return inputs[0] * inputs[1];
  }

  Tuple<Tensor> gradient(Tensor& out_grad, Tensor& node){
    auto input_a = node.inputs()[0];
    auto input_b = node.inputs()[1];
    auto out_grad_a = out_grad * input_b;
    auto out_grad_b = out_grad * input_a;
    return Tuple<Tensor>({out_grad_a, out_grad_b}, true);
  }

  Tensor operator()(const Tuple<Tensor>& inputs)override{
    return Tensor((*this), inputs);
  }

};
