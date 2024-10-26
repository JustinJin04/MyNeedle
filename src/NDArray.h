#pragma once

#include<memory>
#include<iostream>
#include<cassert>
#include<string.h>
#include<algorithm>
#include<vector>
#include"shared_ptr_wrapper.h"
#include"Tuple.h"

class NDArrayStorageDef;
class NDArrayStorage;
class NDArrayMeta;
class NDArrayDef;
class NDArray;

static Tuple<int> compact_strides(const Tuple<int>& shape){
  int strides = 1;
  std::vector<int> shape_vec = shape.to_vec();
  std::vector<int> result;
  for(auto it=shape_vec.rbegin(); it!=shape_vec.rend(); it++){
    result.push_back(strides);
    strides *= *it;
  }
  std::reverse(result.begin(), result.end());
  return std::move(Tuple<int>(std::move(result)));
}


class NDArrayMeta{
public:
  // NDArrayMeta(const Tuple<int>& shape):
  // _shape(shape), _strides(compact_strides(shape)){}
  // NDArrayMeta(const Tuple<int>& shape, const Tuple<int>& strides):
  // _shape(shape), _strides(strides){}
  NDArrayMeta(){}
  NDArrayMeta(const Tuple<int>& shape){
    _shape.copy_(shape);
    _strides.copy_(compact_strides(shape));
  }
  NDArrayMeta(const Tuple<int>& shape, const Tuple<int>& strides){
    _shape.copy_(shape);
    _strides.copy_(strides);
  }
  // 非wrapper类复制构造函数均为深拷贝
  NDArrayMeta(const NDArrayMeta& m){
    _shape.copy_(m._shape);
    _strides.copy_(m._strides);
  }

protected:
  Tuple<int> _shape;
  Tuple<int> _strides;
  friend class NDArrayDef;
  friend class NDArray;

};


class NDArrayStorageDef: public shared_ptr_target{
public:
  NDArrayStorageDef(const float* ptr=nullptr, int size=0){
    _data_ptr = nullptr;
    _size = size;
    if(size!=0){
      _data_ptr = new float[size];
    }
    if(ptr!=nullptr){
      memcpy(_data_ptr, ptr, size*sizeof(float));
    }
  }

  NDArrayStorageDef(const NDArrayStorageDef& s){
    _data_ptr = nullptr;
    _size = s._size;
    if(_size!=0){
      _data_ptr = new float[_size];
    }
    if(s._data_ptr!=nullptr){
      memcpy(_data_ptr, s._data_ptr, _size*sizeof(float));
    }
  }
  NDArrayStorageDef(NDArrayStorageDef&& s)=delete;
  NDArrayStorageDef& operator=(const NDArrayStorageDef& s)=delete;
  NDArrayStorageDef& operator=(NDArrayStorageDef&& s)=delete;

  ~NDArrayStorageDef(){
    if(_data_ptr!=nullptr){
      delete[] _data_ptr;
      _data_ptr = nullptr;
    }
  }

  float* raw_data_ptr()const{
    return _data_ptr;
  }

protected:
  float* _data_ptr;
  int _size;
  friend class NDArrayStorage;
};

class NDArrayStorage: public shared_ptr_wrapper<NDArrayStorageDef>{
public:
  NDArrayStorage(){}
  NDArrayStorage(const float* data, int size){
    this->_ptr = std::make_shared<NDArrayStorageDef>(data, size);
  }

  NDArrayStorage copy_(const NDArrayStorage& s){
    if(s._ptr){
      this->_ptr = std::make_shared<NDArrayStorageDef>(*s._ptr);
      return (*this);
    }
    else{
      this->_ptr = nullptr;
      return (*this);
    }
  }

  float* data_ptr()const{
    if(this->_ptr)
      return this->_ptr->raw_data_ptr();
    assert(0);
    return nullptr;
  }

  int size()const{
    if(this->_ptr)
      return this->_ptr->_size;
    else{
      return 0;
    }
  }
protected:
};


class NDArrayDef: public shared_ptr_target{
public:
  // BUG: shape可能是一个函数内部的临时变量，传引用会导致资源释放后的undefined behavior.需要单独定义
  NDArrayDef(const float* data, int size, const Tuple<int>& shape):
  _meta(shape), _handle(data, size), _offset(0){}

  // constructor function must make a new copy of the underlying data
  NDArrayDef(NDArrayStorage&& data, const Tuple<int>& shape):
  _meta(shape), _handle(data.data_ptr(),data.size()), _offset(0){}

  NDArrayDef(NDArrayStorage& data, const Tuple<int>& shape):
  _meta(shape), _handle(data.data_ptr(),data.size()), _offset(0){}
  
  NDArrayDef(NDArrayStorage& data, const Tuple<int>& shape, const Tuple<int>& strides):
  _meta(shape, strides), _handle(data.data_ptr(),data.size()), _offset(0){}

  // NDArrayDef(const NDArrayDef& s):_meta(s._meta), _handle(s._handle), _offset(0){}
  NDArrayDef(const NDArrayDef& s):_meta(s._meta), _offset(s._offset){
    // 由于是深拷贝，然而NDArrayStorage的复制构造函数均为浅拷贝，故必须调用其copy_
    _handle.copy_(s._handle);
  }
  
  NDArrayDef(NDArrayDef&& s)=delete;
  NDArrayDef& operator=(const NDArrayDef& s)=delete;
  NDArrayDef& operator=(NDArrayDef&& s)=delete;

  
  Tuple<int> shape()const{
    return _meta._shape;
  }

  Tuple<int> strides()const{
    return _meta._strides;
  }

  bool is_compact()const{
    auto strides = compact_strides(_meta._shape);
    return _meta._strides == strides;
  }

  // maybe not compact
  const float* raw_data_ptr()const{
    return _handle.data_ptr();
  }

protected:
  NDArrayMeta _meta;
  NDArrayStorage _handle;
  int _offset;
  friend class NDArray;
};

/**
 * Abstract: a reference(like python) to underlying object
 * Thus, we can treat it just like python does
 * i.e., if we already have a = A(), then b = a just increase a 
 * reference count on the created object
 * and if we define a function that return an object that's created in this function,
 * we simply use rvalue construction and = function to assign this newly created object to
 * a new owner (with shared_ptr referenc count -1 because of destruction of function and +1 because newly assi
 * gned owner)
 */
class NDArray: public shared_ptr_wrapper<NDArrayDef>{
public:
  NDArray(){}
  // 以下均为深度拷贝
  NDArray(const float* data, int size, const Tuple<int>& shape);
  NDArray(NDArrayStorage&& data, const Tuple<int>& shape);
  NDArray(NDArrayStorage& data, const Tuple<int>& shape);
  NDArray(NDArrayStorage& data, const Tuple<int>& shape, const Tuple<int>& strides);

  NDArray copy_(const NDArray& array){
    if(array._ptr){
      this->_ptr = std::make_shared<NDArrayDef>(*array._ptr);
    }
    else{
      this->_ptr = nullptr;
    }
  return (*this);
  }

  Tuple<int> shape()const;
  Tuple<int> strides()const;
  int offset()const;
  
  // contiguous data size if compacted
  int compact_data_size()const;
  
  // return self if compact
  // otherwise create a new ndarray
  // because ndarray is indeed a shared_ptr, if return (*this), it will call
  // copy construct function which increase a reference count of the shared_ptr
  NDArray compact() const;

  bool is_compact()const;

  // maybe none compact
  const float* raw_data_ptr()const;

  float* debug_data_ptr()const{
    return this->_ptr->_handle.data_ptr();
  }

  friend std::ostream& operator<<(std::ostream& os, const NDArray& obj);


  //
  NDArray reshape(const Tuple<int>& shape)const;
  NDArray permute(const Tuple<int>& new_axes)const;

protected:
};


NDArray operator+(const NDArray& lhs, const NDArray& rhs);

NDArray operator*(const NDArray& lhs, const NDArray& rhs);

