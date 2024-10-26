#pragma once

#include<memory>
#include<ostream>

template <typename T>
class shared_ptr_wrapper;

class shared_ptr_target {
private:
  friend class shared_ptr_wrapper<shared_ptr_target>;
};

template <typename T>
class shared_ptr_wrapper {
 public:
  shared_ptr_wrapper() {
    static_assert(std::is_base_of<shared_ptr_target, T>::value,
                  "Template class is not derived from shared_ptr_target");
  }

  shared_ptr_wrapper& operator=(const shared_ptr_wrapper& other) {
    _ptr = other._ptr;
    return *this;
  }

  shared_ptr_wrapper& operator=(shared_ptr_wrapper&& other) {
    _ptr = other._ptr;
    return *this;
  }

  // template <typename U>
  // shared_ptr_wrapper& operator=(const shared_ptr_wrapper<U>& other) {
  //   static_assert(std::is_convertible<U*, T*>::value,
  //                 "Types are not convertible.");
  //   _ptr = other._ptr;
  //   return *this;
  // }

  // template <typename U>
  // shared_ptr_wrapper& operator=(shared_ptr_wrapper<U>&& other) {
  //   static_assert(std::is_convertible<U*, T*>::value,
  //                 "Types are not convertible.");
  //   _ptr = other._ptr;
  //   return *this;
  // }

  // BUG: 对所有的复制构造函数，一律单独创造一个副本？？？
  // 对于 = ，一律创造引用？？？
  shared_ptr_wrapper(const shared_ptr_wrapper<T>& other) {
    _ptr = other._ptr;
    // _ptr = std::make_shared<T>(*other._ptr)
  }

  shared_ptr_wrapper(shared_ptr_wrapper<T>&& other) {
    _ptr = other._ptr;
    // _ptr = std::make_shared<T>(*other._ptr);
  }

  // virtual shared_ptr_wrapper copy_(const shared_ptr_wrapper& t) = 0;

  // template <typename U>
  // shared_ptr_wrapper(const shared_ptr_wrapper<U>& other) {
  //   static_assert(std::is_convertible<U*, T*>::value,
  //                 "Types are not convertible.");
  //   _ptr = other._ptr;
  // }

  // template <typename U>
  // shared_ptr_wrapper(shared_ptr_wrapper<U>&& other) {
  //   static_assert(std::is_convertible<U*, T*>::value,
  //                 "Types are not convertible.");
  //   _ptr = other._ptr;
  // }

  bool is_defined() const {
    return _ptr != nullptr;
  }

  // methods from shared pointers
  T* get() const noexcept {
    return _ptr.get();
  }
  operator bool() const noexcept {
    return _ptr != nullptr;
  }
  T& operator*() const noexcept {
    return _ptr.get();
  }
  std::shared_ptr<T> operator->() const noexcept {
    return _ptr;
  }
  size_t use_count() const noexcept {
    return _ptr.use_count();
  }
  bool unique() const noexcept {
    return _ptr.unique();
  }

 protected:
  template <typename U>
  friend class shared_ptr_wrapper;

  template <class Derived, class... Args>
  static std::shared_ptr<T> make_ptr(Args&&... args) {
    static_assert(std::is_base_of<T, Derived>::value);
    return std::make_shared<Derived>(std::forward<Args>(args)...);
  }

  std::shared_ptr<T> _ptr{nullptr};
};