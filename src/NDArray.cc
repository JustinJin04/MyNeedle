#include"NDArray.h"
#include<cassert>
#include"kernel.h"
#include<functional>
#include<stack>
#include<iomanip>

NDArray::NDArray(const float* data, int size, const Tuple<int>& shape){
  this->_ptr = std::make_shared<NDArrayDef>(data,size, shape);
}

NDArray::NDArray(NDArrayStorage&& data, const Tuple<int>& shape){
  this->_ptr = std::make_shared<NDArrayDef>(std::move(data), shape);
}

NDArray::NDArray(NDArrayStorage& data, const Tuple<int>& shape){
  this->_ptr = std::make_shared<NDArrayDef>(data, shape);
}

NDArray::NDArray(NDArrayStorage& data, const Tuple<int>& shape, const Tuple<int>& strides){
  this->_ptr = std::make_shared<NDArrayDef>(data, shape, strides);
}

Tuple<int> NDArray::shape()const{
  return this->_ptr->_meta._shape;
}

Tuple<int> NDArray::strides()const{
  return this->_ptr->_meta._strides;
}

int NDArray::offset()const{
  return this->_ptr->_offset;
}

int NDArray::compact_data_size()const{
  int size=1;
  for(auto i: this->shape()){
      size *= i;
  }
  return size;
}

/**
 * BUG: maybe return a temporary object
 * when referred to its resource, one must 
 * keep the temporary object until the reference is done
 */
NDArray NDArray::compact()const{
  if(this->_ptr->is_compact()){
      return (*this);
  }
  else{
    int size = this->compact_data_size();
    auto out_storage = NDArrayStorage(nullptr, size);    
    cpu_compact(this->raw_data_ptr(), this->shape().to_vec(), this->strides().to_vec(), this->offset(), out_storage.data_ptr());
    auto out = NDArray(std::move(out_storage), this->shape());
    return out;
  }
}

bool NDArray::is_compact()const{
  return this->_ptr->is_compact();
}

const float* NDArray::raw_data_ptr()const{
  return this->_ptr->raw_data_ptr();
}

NDArray NDArray::reshape(const Tuple<int>& shape)const{
  assert(is_compact());
  return std::move(NDArray(this->_ptr->_handle, shape));
}

NDArray NDArray::permute(const Tuple<int>& new_axes)const{
  
  auto permute_func = [](const Tuple<int>& input_list, const Tuple<int>& permute_list) {
    std::vector<int>result_vec;
    auto input_vec = std::vector<int>(input_list.begin(), input_list.end());  // Convert list to vector for random access
    for (int index : permute_list) {
        result_vec.push_back(input_list[index]);
    }
    Tuple<int> out_list(result_vec);
    return std::move(out_list);
  };
  
  auto new_shape = permute_func(this->shape(), new_axes);
  auto new_strides = permute_func(this->strides(), new_axes);
  return std::move(NDArray(this->_ptr->_handle, new_shape, new_strides));
}

std::ostream& operator<<(std::ostream& os, const NDArray& obj){
  int size = obj.compact_data_size();
  auto compact_obj = obj.compact();
  const float* ptr = compact_obj.raw_data_ptr();
  
  std::function<void(const float* in_ptr, int in_idx, const Tuple<int>& shape,
  const Tuple<int>& strides, int ndims, bool is_last)>dfs;
  dfs = [&](const float* in_ptr, int in_idx, const Tuple<int>& shape,
  const Tuple<int>& strides, int ndims, bool is_last){
    if(ndims >= shape.size()){
      if(is_last){
        os<<in_ptr[in_idx];
      }
      else{
        os<<in_ptr[in_idx]<<", ";
      }
      return;
    }
    else{
      os<<"[";
      for(int i=0;i<shape[ndims];++i){
        bool is_last = (i==(shape[ndims]-1));
        dfs(in_ptr, in_idx + i*strides[ndims], shape, strides, ndims+1,is_last);
      }
      os<<"]";
      if(!is_last){
        os<<"\n";
      }
    }
  };
  dfs(ptr, 0, compact_obj.shape(), compact_obj.strides(), 0, 0);
  os << std::endl;
  return os;
}

/**
 * BUG: 获取临时对象的资源指针是不合法的。临时对象的资源随时可能被释放。
 * 如果需要访问临时对象的资源，必须要确保资源使用结束前临时对象不会被销毁
 * const float* lhs_ptr = lhs.compact().raw_data_ptr();
 * 是错误的写法。由于lhs.compact可能返回一个临时对象，获取其资源lhs_ptr是UDF
 */
NDArray operator+(const NDArray& lhs, const NDArray& rhs){
  assert(lhs.shape() == rhs.shape());
  // const float* lhs_ptr = lhs.compact().raw_data_ptr();
  // const float* rhs_ptr = rhs.compact().raw_data_ptr();
  auto lhs_compact = lhs.compact();
  auto rhs_compact = rhs.compact();
  const float* lhs_ptr = lhs_compact.raw_data_ptr();
  const float* rhs_ptr = rhs_compact.raw_data_ptr();
  int size = lhs.compact_data_size();
  auto out_storage = NDArrayStorage(nullptr, size);
  cpu_add(lhs_ptr, rhs_ptr,size, out_storage.data_ptr());
  return NDArray (out_storage, lhs.shape());
}

NDArray operator*(const NDArray& lhs, const NDArray& rhs){
  assert(lhs.shape() == rhs.shape());
  auto lhs_compact = lhs.compact();
  auto rhs_compact = rhs.compact();
  const float* lhs_ptr = lhs_compact.raw_data_ptr();
  const float* rhs_ptr = rhs_compact.raw_data_ptr();
  int size = lhs.compact_data_size();
  auto out_storage = NDArrayStorage(nullptr, size);
  cpu_mul(lhs_ptr, rhs_ptr,size, out_storage.data_ptr());
  return NDArray (out_storage, lhs.shape());
}


