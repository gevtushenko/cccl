//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NASTY_CONTAINERS_H
#define NASTY_CONTAINERS_H

#include <cuda/std/cassert>

#include <list>
#include <vector>

#include "test_macros.h"

template <class T>
class nasty_vector
{
public:
  typedef typename std::vector<T> nested_container;
  typedef typename nested_container::value_type value_type;
  typedef typename nested_container::reference reference;
  typedef typename nested_container::const_reference const_reference;
  typedef typename nested_container::iterator iterator;
  typedef typename nested_container::const_iterator const_iterator;

  typedef typename nested_container::size_type size_type;
  typedef typename nested_container::difference_type difference_type;
  typedef typename nested_container::pointer pointer;
  typedef typename nested_container::const_pointer const_pointer;

  typedef typename nested_container::reverse_iterator reverse_iterator;
  typedef typename nested_container::const_reverse_iterator const_reverse_iterator;

  nasty_vector()
      : v_()
  {}
  explicit nasty_vector(size_type n)
      : v_(n)
  {}
  nasty_vector(size_type n, const value_type& value)
      : v_(n, value)
  {}
  template <class InputIterator>
  nasty_vector(InputIterator first, InputIterator last)
      : v_(first, last)
  {}
  nasty_vector(std::initializer_list<value_type> il)
      : v_(il)
  {}
  ~nasty_vector() {}

  template <class InputIterator>
  void assign(InputIterator first, InputIterator last)
  {
    v_.assign(first, last);
  }
  void assign(size_type n, const value_type& u)
  {
    v_.assign(n, u);
  }
  void assign(std::initializer_list<value_type> il)
  {
    v_.assign(il);
  }

  iterator begin() noexcept
  {
    return v_.begin();
  }
  const_iterator begin() const noexcept
  {
    return v_.begin();
  }
  iterator end() noexcept
  {
    return v_.end();
  }
  const_iterator end() const noexcept
  {
    return v_.end();
  }

  reverse_iterator rbegin() noexcept
  {
    return v_.rbegin();
  }
  const_reverse_iterator rbegin() const noexcept
  {
    return v_.rbegin();
  }
  reverse_iterator rend() noexcept
  {
    return v_.rend();
  }
  const_reverse_iterator rend() const noexcept
  {
    return v_.rend();
  }

  const_iterator cbegin() const noexcept
  {
    return v_.cbegin();
  }
  const_iterator cend() const noexcept
  {
    return v_.cend();
  }
  const_reverse_iterator crbegin() const noexcept
  {
    return v_.crbegin();
  }
  const_reverse_iterator crend() const noexcept
  {
    return v_.crend();
  }

  size_type size() const noexcept
  {
    return v_.size();
  }
  size_type max_size() const noexcept
  {
    return v_.max_size();
  }
  size_type capacity() const noexcept
  {
    return v_.capacity();
  }
  bool empty() const noexcept
  {
    return v_.empty();
  }
  void reserve(size_type n)
  {
    v_.reserve(n);
  };
  void shrink_to_fit() noexcept
  {
    v_.shrink_to_fit();
  }

  reference operator[](size_type n)
  {
    return v_[n];
  }
  const_reference operator[](size_type n) const
  {
    return v_[n];
  }
  reference at(size_type n)
  {
    return v_.at(n);
  }
  const_reference at(size_type n) const
  {
    return v_.at(n);
  }

  reference front()
  {
    return v_.front();
  }
  const_reference front() const
  {
    return v_.front();
  }
  reference back()
  {
    return v_.back();
  }
  const_reference back() const
  {
    return v_.back();
  }

  value_type* data() noexcept
  {
    return v_.data();
  }
  const value_type* data() const noexcept
  {
    return v_.data();
  }

  void push_back(const value_type& x)
  {
    v_.push_back(x);
  }
  void push_back(value_type&& x)
  {
    v_.push_back(std::forward<value_type&&>(x));
  }
  template <class... Args>
  void emplace_back(Args&&... args)
  {
    v_.emplace_back(std::forward<Args>(args)...);
  }
  void pop_back()
  {
    v_.pop_back();
  }

  template <class... Args>
  iterator emplace(const_iterator pos, Args&&... args)
  {
    return v_.emplace(pos, std::forward<Args>(args)...);
  }

  iterator insert(const_iterator pos, const value_type& x)
  {
    return v_.insert(pos, x);
  }
  iterator insert(const_iterator pos, value_type&& x)
  {
    return v_.insert(pos, std::forward<value_type>(x));
  }
  iterator insert(const_iterator pos, size_type n, const value_type& x)
  {
    return v_.insert(pos, n, x);
  }
  template <class InputIterator>
  iterator insert(const_iterator pos, InputIterator first, InputIterator last)
  {
    return v_.insert(pos, first, last);
  }

  iterator insert(const_iterator pos, std::initializer_list<value_type> il)
  {
    return v_.insert(pos, il);
  }

  iterator erase(const_iterator pos)
  {
    return v_.erase(pos);
  }
  iterator erase(const_iterator first, const_iterator last)
  {
    return v_.erase(first, last);
  }

  void clear() noexcept
  {
    v_.clear();
  }

  void resize(size_type sz)
  {
    v_.resize(sz);
  }
  void resize(size_type sz, const value_type& c)
  {
    v_.resize(sz, c);
  }

  void swap(nasty_vector& nv) noexcept(std::is_nothrow_swappable<nested_container>::value)
  {
    v_.swap(nv.v_);
  }

  nasty_vector* operator&()
  {
    assert(false);
    return nullptr;
  } // nasty
  const nasty_vector* operator&() const
  {
    assert(false);
    return nullptr;
  } // nasty

  nested_container v_;
};

template <class T>
bool operator==(const nasty_vector<T>& x, const nasty_vector<T>& y)
{
  return x.v_ == y.v_;
}

template <class T>
class nasty_list
{
public:
  typedef typename std::list<T> nested_container;
  typedef typename nested_container::value_type value_type;
  typedef typename nested_container::reference reference;
  typedef typename nested_container::const_reference const_reference;
  typedef typename nested_container::iterator iterator;
  typedef typename nested_container::const_iterator const_iterator;

  typedef typename nested_container::size_type size_type;
  typedef typename nested_container::difference_type difference_type;
  typedef typename nested_container::pointer pointer;
  typedef typename nested_container::const_pointer const_pointer;

  typedef typename nested_container::reverse_iterator reverse_iterator;
  typedef typename nested_container::const_reverse_iterator const_reverse_iterator;

  nasty_list()
      : l_()
  {}
  explicit nasty_list(size_type n)
      : l_(n)
  {}
  nasty_list(size_type n, const value_type& value)
      : l_(n, value)
  {}
  template <class Iter>
  nasty_list(Iter first, Iter last)
      : l_(first, last)
  {}
  nasty_list(std::initializer_list<value_type> il)
      : l_(il)
  {}

  ~nasty_list() {}

  nasty_list& operator=(std::initializer_list<value_type> il)
  {
    l_ = il;
    return *this;
  }
  template <class Iter>
  void assign(Iter first, Iter last)
  {
    l_.assign(first, last);
  }
  void assign(size_type n, const value_type& t)
  {
    l_.assign(n, t);
  }
  void assign(std::initializer_list<value_type> il)
  {
    l_.assign(il);
  }

  iterator begin() noexcept
  {
    return l_.begin();
  }
  const_iterator begin() const noexcept
  {
    return l_.begin();
  }
  iterator end() noexcept
  {
    return l_.end();
  }
  const_iterator end() const noexcept
  {
    return l_.end();
  }

  reverse_iterator rbegin() noexcept
  {
    return l_.rbegin();
  }
  const_reverse_iterator rbegin() const noexcept
  {
    return l_.rbegin();
  }
  reverse_iterator rend() noexcept
  {
    return l_.rend();
  }
  const_reverse_iterator rend() const noexcept
  {
    return l_.rend();
  }

  const_iterator cbegin() const noexcept
  {
    return l_.cbegin();
  }
  const_iterator cend() const noexcept
  {
    return l_.cend();
  }
  const_reverse_iterator crbegin() const noexcept
  {
    return l_.crbegin();
  }
  const_reverse_iterator crend() const noexcept
  {
    return l_.crend();
  }

  reference front()
  {
    return l_.front();
  }
  const_reference front() const
  {
    return l_.front();
  }
  reference back()
  {
    return l_.back();
  }
  const_reference back() const
  {
    return l_.back();
  }

  size_type size() const noexcept
  {
    return l_.size();
  }
  size_type max_size() const noexcept
  {
    return l_.max_size();
  }
  bool empty() const noexcept
  {
    return l_.empty();
  }

  void push_front(const value_type& x)
  {
    l_.push_front(x);
  }
  void push_back(const value_type& x)
  {
    l_.push_back(x);
  }
  void push_back(value_type&& x)
  {
    l_.push_back(std::forward<value_type&&>(x));
  }
  void push_front(value_type&& x)
  {
    l_.push_back(std::forward<value_type&&>(x));
  }
  template <class... Args>
  void emplace_back(Args&&... args)
  {
    l_.emplace_back(std::forward<Args>(args)...);
  }
  template <class... Args>
  void emplace_front(Args&&... args)
  {
    l_.emplace_front(std::forward<Args>(args)...);
  }
  void pop_front()
  {
    l_.pop_front();
  }
  void pop_back()
  {
    l_.pop_back();
  }

  template <class... Args>
  iterator emplace(const_iterator pos, Args&&... args)
  {
    return l_.emplace(pos, std::forward<Args>(args)...);
  }

  iterator insert(const_iterator pos, const value_type& x)
  {
    return l_.insert(pos, x);
  }
  iterator insert(const_iterator pos, value_type&& x)
  {
    return l_.insert(pos, std::forward<value_type>(x));
  }
  iterator insert(const_iterator pos, size_type n, const value_type& x)
  {
    return l_.insert(pos, n, x);
  }
  template <class InputIterator>
  iterator insert(const_iterator pos, InputIterator first, InputIterator last)
  {
    return l_.insert(pos, first, last);
  }

  iterator insert(const_iterator pos, std::initializer_list<value_type> il)
  {
    return l_.insert(pos, il);
  }

  iterator erase(const_iterator pos)
  {
    return l_.erase(pos);
  }
  iterator erase(const_iterator pos, const_iterator last)
  {
    return l_.erase(pos, last);
  }

  void resize(size_type)
  {
    l_.resize();
  }
  void resize(size_type, const value_type& c)
  {
    l_.resize(c);
  }

  void swap(nasty_list& nl) noexcept(std::is_nothrow_swappable<nested_container>::value)
  {
    l_.swap(nl.l_);
  }

  void clear() noexcept
  {
    l_.clear();
  }

  //     void splice(const_iterator position, list& x);
  //     void splice(const_iterator position, list&& x);
  //     void splice(const_iterator position, list& x, const_iterator i);
  //     void splice(const_iterator position, list&& x, const_iterator i);
  //     void splice(const_iterator position, list& x, const_iterator first,
  //                                                   const_iterator last);
  //     void splice(const_iterator position, list&& x, const_iterator first,
  //                                                   const_iterator last);
  //
  //     void remove(const value_type& value);
  //     template <class Pred> void remove_if(Pred pred);
  //     void unique();
  //     template <class BinaryPredicate>
  //         void unique(BinaryPredicate binary_pred);
  //     void merge(list& x);
  //     void merge(list&& x);
  //     template <class Compare>
  //         void merge(list& x, Compare comp);
  //     template <class Compare>
  //         void merge(list&& x, Compare comp);
  //     void sort();
  //     template <class Compare>
  //         void sort(Compare comp);
  //     void reverse() noexcept;

  nasty_list* operator&()
  {
    assert(false);
    return nullptr;
  } // nasty
  const nasty_list* operator&() const
  {
    assert(false);
    return nullptr;
  } // nasty

  nested_container l_;
};

template <class T>
bool operator==(const nasty_list<T>& x, const nasty_list<T>& y)
{
  return x.l_ == y.l_;
}

// Not really a mutex, but can play one in tests
class nasty_mutex
{
public:
  nasty_mutex() noexcept {}
  ~nasty_mutex() {}

  nasty_mutex* operator&()
  {
    assert(false);
    return nullptr;
  }
  template <typename T>
  void operator,(const T&)
  {
    assert(false);
  }

private:
  nasty_mutex(const nasty_mutex&)
  {
    assert(false);
  }
  nasty_mutex& operator=(const nasty_mutex&)
  {
    assert(false);
    return *this;
  }

public:
  void lock() {}
  bool try_lock() noexcept
  {
    return true;
  }
  void unlock() noexcept {}

  // Shared ownership
  void lock_shared() {}
  bool try_lock_shared()
  {
    return true;
  }
  void unlock_shared() {}
};

#endif
