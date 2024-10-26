#pragma once

#include"shared_ptr_wrapper.h"
#include<vector>
#include"algorithm"

template<typename T>
class TupleDef;

template<typename T>
class Tuple;

/**
 * TODO: GPT生成，不知道是否有bug？？？？
 */
template <typename T>
struct has_copy_ {
  template <typename U>
  static auto test(U*) -> decltype(std::declval<U>().copy_(std::declval<const U&>()), std::true_type{});

  template <typename>
  static std::false_type test(...);

  static constexpr bool value = decltype(test<T>(nullptr))::value;
};


  template<typename T>
  class TupleDef: public shared_ptr_target{
  public:


  TupleDef(const std::vector<T>& vec, bool is_shallow){
    _size = vec.size();
    _data = nullptr;
    if(_size > 0){
      _data = new T[_size];
      if constexpr(has_copy_<T>::value){
        if(is_shallow){
          for(int i=0;i<_size;++i){
            _data[i] = vec[i];
          }
        }
        else{
          for(int i=0;i<_size;++i){
            _data[i].copy_(vec[i]);
          }
        }
      }
      else{
        for(int i=0;i<_size;++i){
          _data[i] = vec[i];
        }
      }
    }
  }

  TupleDef(const std::initializer_list<T>& init_list, bool is_shallow){
    _size = init_list.size();
    _data = nullptr;
    if(_size > 0){
      _data = new T[_size];
      std::vector<T> vec(init_list.begin(), init_list.end());
      if constexpr(has_copy_<T>::value){
        if(is_shallow){
          for(int i=0;i<_size;++i){
            _data[i] = vec[i];
          }
        }
        else{
          for(int i=0;i<_size;++i){
            _data[i].copy_(vec[i]);
          }
        }
      }
      else{
        for(int i=0;i<_size;++i){
          _data[i] = vec[i];
        }
      }
    }
  }
  
  TupleDef(const TupleDef<T>& t, bool is_shallow){
    _size = t._size;
    _data = nullptr;
    if(_size > 0){
      _data = new T[_size];
      // due to deep copy, we must call copy_ func on every slots
      if constexpr(has_copy_<T>::value){
        if(is_shallow){
          for(int i=0;i<_size;++i){
            _data[i] = t._data[i];
          }
        }
        else{
          for(int i=0;i<_size;++i){
            _data[i].copy_(t._data[i]);
          }
        }
      }
      else{
        for(int i=0;i<_size;++i){
          _data[i] = t._data[i];
        }
      }
    }
  }

  ~TupleDef(){
    if(_data != nullptr){
      delete[] _data;
      _data = nullptr;
      _size = 0;
    }
  }
  
  TupleDef(TupleDef<T>&& t)=delete;
  TupleDef<T>&operator=(const TupleDef<T>& t)=delete;
  TupleDef<T>&operator=(TupleDef<T>&& t)=delete;

protected:
  T* _data;
  int _size;
  // friend cannot succeed
  friend class Tuple<T>;
};

template<typename T>
class Tuple: public shared_ptr_wrapper<TupleDef<T>>{
public:
  Tuple(){}
  Tuple(const std::vector<T>& vec, bool is_shallow=false){
    this->_ptr = std::make_shared<TupleDef<T>>(vec, is_shallow);
  }

  Tuple(const std::initializer_list<T>& init_list, bool is_shallow=false) {
    this->_ptr = std::make_shared<TupleDef<T>>(init_list, is_shallow);
  }

  /**
   * copy_ 包括T在内的深度靠别
   * shallow_copy_ 创建新的tuple，内容还是原来的reference
   */
  Tuple<T> copy_ (const Tuple<T>& t, bool is_shallow=false){
    if(t._ptr){
      this->_ptr = std::make_shared<TupleDef<T>>(*t._ptr, is_shallow);
    }
    else{
      this->_ptr = nullptr;
    }
    return (*this);
  }

  Tuple<T> shallow_copy_(const Tuple<T>& t){
    if(t._ptr){
      this->_ptr = std::make_shared<TupleDef<T>>(*t._ptr, true);
    }
    else{
      this->_ptr = nullptr;
    }
    return (*this);
  }

  bool operator==(const Tuple<T>& t)const{
    if(this->_ptr->_size != t._ptr->_size){
      return false;
    }
    for(int i=0;i<this->_ptr->_size;++i){
      if((*this)[i]!=t[i]){
        return false;
      }
    }
    return true;
  }

  std::vector<T> to_vec()const{
    std::vector<T>out(this->_ptr->_data, this->_ptr->_data + this->_ptr->_size);
    return out;
  }

  int size()const{
    if(this->is_defined()){
      return this->_ptr->_size;
    }
    else{
      return 0;
    }
  }

  // TODO: iterator func
  // this suggest that the underlying data is contuguous
  // BUG: 如何处理_ptr为null的情况
  const T* begin() const {
    if(this->_ptr)
      return this->_ptr->_data;
    else{
      return nullptr;
    }
  }
  const T* end() const {
    if(this->_ptr)
      return this->_ptr->_data + this->_ptr->_size;
    else{
      return nullptr;
    }
  }

  T& operator[](int idx)const{
    assert(idx >= 0 && idx < this->_ptr->_size);
    return this->_ptr->_data[idx];
  }

  void debug_print(){
    for(int i=0;i<this->size();++i){
      std::cout<<(*this)[i]<<" ";
    }
    std::cout<<"\n";
  }

protected:

};


