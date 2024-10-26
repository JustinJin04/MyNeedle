#include "Tensor.h"
#include <vector>
#include <unordered_set>
#include <algorithm>
#include "ops/ops_header.h"


/********** class TensorDef **********/

TensorDef::TensorDef(const NDArray& array, bool require_grad){
  _cached_data.copy_(array);
  _requires_grad = require_grad;
}

template<typename DerivedOp>
TensorDef::TensorDef(const DerivedOp& op, const Tuple<Tensor>& inputs){
  _op_ptr = std::make_shared<DerivedOp>();
  // BUG: tuple的copy_函数包括深度拷贝Tensor。这里只需要浅拷贝？？？？
  // _inputs.copy_(inputs);
  _inputs.shallow_copy_(inputs);
  _requires_grad = true;
}




/********** class Tensor ************/

Tensor::Tensor(const NDArray& array, bool require_grad){
  this->_ptr = std::make_shared<TensorDef>(array, require_grad);
}

template<typename DerivedOp>
Tensor::Tensor(const DerivedOp& op, const Tuple<Tensor>& inputs){
  static_assert(std::is_base_of<Operator, DerivedOp>::value && !std::is_same<Operator, DerivedOp>::value);
  this->_ptr = std::make_shared<TensorDef>(op, inputs);
}

bool Tensor::operator==(const Tensor& t)const{
  return this->_ptr == t._ptr;
}

Tuple<Tensor> Tensor::inputs()const{
  if(this->_ptr->_inputs.is_defined())
    return this->_ptr->_inputs;
  else{
    return {};
  }
}

Tensor& Tensor::gradient(){
  assert(this->_ptr->_grad_ptr);
  return (*(this->_ptr->_grad_ptr));
}

void Tensor::add_to_gradient(const Tensor& t){
  if(this->_ptr->_grad_ptr == nullptr){
    this->_ptr->_grad_ptr = std::make_unique<Tensor>(t);
  }
  else{
    Tensor& current_grad = *this->_ptr->_grad_ptr;
    auto new_grad = current_grad + t;
    this->_ptr->_grad_ptr = std::make_unique<Tensor>(new_grad);
  }
}

void Tensor::reset_gradient(){
  this->_ptr->_grad_ptr = nullptr;
}

bool Tensor::require_grad()const{
  return this->_ptr->_requires_grad;
}

NDArray Tensor::realize_cached_data(){
  assert(this->_ptr);
  if(this->_ptr->_cached_data.is_defined()){
    return this->_ptr->_cached_data;
  }
  else{
    std::vector<NDArray> inputs_vec;
    for(auto input: this->_ptr->_inputs){
      inputs_vec.push_back(input.realize_cached_data());
    }
    Tuple<NDArray>inputs(inputs_vec);
    
    this->_ptr->_cached_data = this->_ptr->_op_ptr->compute(inputs);
    return this->_ptr->_cached_data;
  }
}

void Tensor::set_data(const NDArray& data){
  this->_ptr->_cached_data = data;
}

void Tensor::backward(Tensor& out_grad){
  std::cout<<"start backward: begin topo sort\n";
  // first, figure out reverse topo order of graph and reset each grad
  std::unordered_set<Tensor> visited;
  std::vector<Tensor> reverse_topo_order;
  topo_sort_dfs_reset_grad((*this), visited, reverse_topo_order);
  std::reverse(reverse_topo_order.begin(), reverse_topo_order.end());
  std::cout<<"finish topo sort. start update grad\n";
  
  // next, accumulate and update grad according to reverse_topo_order
  this->add_to_gradient(out_grad);
  for(auto node: reverse_topo_order){
    // std::cout<<"node: "<<node;
    if(node._ptr->_op_ptr == nullptr){
      continue;
    }
    assert(node._ptr->_grad_ptr);
    auto grad = node.gradient();
    // std::cout<<"grad: "<<grad;
    auto partial_adjoints = node._ptr->_op_ptr->gradient(grad, node);
    int input_size = node._ptr->_inputs.size();
    assert(partial_adjoints.size() == input_size);
    for(int i=0;i<input_size;++i){
      auto input_tensor = node._ptr->_inputs[i];
      auto partial_adjoint = partial_adjoints[i];
      // std::cout<<"input tensor: "<<input_tensor<<"partial_adjoint: "<<partial_adjoint;
      input_tensor.add_to_gradient(partial_adjoint);
      // std::cout<<"input grad: "<<input_tensor.gradient();
    }
  }
}


/*********** none member func *************/

std::ostream& operator<<(std::ostream& o, Tensor& t){
  assert(t.is_defined());
  o<<t.realize_cached_data();
  return o;
}

Tensor operator+(const Tensor& a, const Tensor& b){
  if(!a.is_defined()){
    assert(b.is_defined());
    return b;
  }
  else if (!b.is_defined()){
    return a;
  }
  return EWiseAdd()({a,b});
}

Tensor operator*(const Tensor& a, const Tensor& b){
  assert(a.is_defined() && b.is_defined());
  return EWiseMul()({a,b});
}