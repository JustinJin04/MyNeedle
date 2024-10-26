#pragma once

#include"shared_ptr_wrapper.h"
#include"NDArray.h"
#include"Operator.h"
#include"Tuple.h"
#include"iostream"
#include <unordered_set>

class Operator;

class TensorDef;
class Tensor;

class TensorDef: public shared_ptr_target{
public:
  TensorDef(const NDArray& array, bool require_grad=false);

  template<typename DerivedOp>
  TensorDef(const DerivedOp& op, const Tuple<Tensor>& inputs);

protected:
  // TODO: 后续再考虑split等返回值为tuple的情况
  NDArray _cached_data;
  std::shared_ptr<Operator> _op_ptr{nullptr};
  Tuple<Tensor> _inputs;
  bool _requires_grad=false;
  std::unique_ptr<Tensor> _grad_ptr{nullptr};
  
  friend class Tensor;
};

class Tensor: public shared_ptr_wrapper<TensorDef>{
public:
  Tensor(){}

  Tensor(const NDArray& array, bool require_grad=false);

  template<typename DerivedOp>
  Tensor(const DerivedOp& op, const Tuple<Tensor>& inputs);

  NDArray realize_cached_data();

  void set_data(const NDArray& data);

  void backward(Tensor& out_grad);

  bool operator==(const Tensor& t)const;

  Tuple<Tensor> inputs()const;

  Tensor& gradient();

  void add_to_gradient(const Tensor& t);

  void reset_gradient();

  bool require_grad()const;

  friend std::ostream& operator<<(std::ostream& o, Tensor& t);

protected:
  friend struct std::hash<Tensor>;
};

template<>
struct std::hash<Tensor>{
  std::size_t operator()(const Tensor& t)const{
    return std::hash<std::size_t>()(std::size_t(t._ptr.get()));
  }
};

static void 
topo_sort_dfs_reset_grad(Tensor& node, std::unordered_set<Tensor>& visited, std::vector<Tensor>& topo_order){
  visited.insert(node);
  for(auto parent: node.inputs()){
    if(visited.find(parent)!=visited.end()){
      continue;
    }
    topo_sort_dfs_reset_grad(parent, visited, topo_order);
  }
  node.reset_gradient();
  topo_order.push_back(node);
}


/********** ops declaration ***********/

Tensor operator+(const Tensor& a, const Tensor& b);

Tensor operator*(const Tensor& a, const Tensor& b);