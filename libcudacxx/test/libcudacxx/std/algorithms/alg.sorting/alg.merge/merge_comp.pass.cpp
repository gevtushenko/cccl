//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator InIter1, InputIterator InIter2, typename OutIter,
//          CopyConstructible Compare>
//   requires OutputIterator<OutIter, InIter1::reference>
//         && OutputIterator<OutIter, InIter2::reference>
//         && Predicate<Compare, InIter1::value_type, InIter2::value_type>
//         && Predicate<Compare, InIter2::value_type, InIter1::value_type>
//   constexpr OutIter       // constexpr after C++17
//   merge(InIter1 first1, InIter1 last1, InIter2 first2, InIter2 last2,
//         OutIter result, Compare comp);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>

#include "../sortable_helpers.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class T, class Iter1, class Iter2, class OutIter>
__host__ __device__ constexpr void test4()
{
  const T a[] = {11, 33, 31, 41};
  const T b[] = {22, 32, 43, 42, 52};
  {
    T result[20] = {};
    T expected[] = {11, 22, 33, 31, 32, 41, 43, 42, 52};
    OutIter end =
      cuda::std::merge(Iter1(a), Iter1(a + 4), Iter2(b), Iter2(b + 5), OutIter(result), typename T::Comparator());
    assert(cuda::std::lexicographical_compare(result, base(end), expected, expected + 9, T::less) == 0);
    for (const T* it = base(end); it != result + 20; ++it)
    {
      assert(it->value == 0);
    }
  }
  {
    T result[20] = {};
    T expected[] = {11, 22, 32, 33, 31, 43, 42, 41, 52};
    OutIter end =
      cuda::std::merge(Iter1(b), Iter1(b + 5), Iter2(a), Iter2(a + 4), OutIter(result), typename T::Comparator());
    assert(cuda::std::lexicographical_compare(result, base(end), expected, expected + 9, T::less) == 0);
    for (const T* it = base(end); it != result + 20; ++it)
    {
      assert(it->value == 0);
    }
  }
}

template <class T, class Iter1, class Iter2>
__host__ __device__ constexpr void test3()
{
  test4<T, Iter1, Iter2, cpp17_output_iterator<T*>>();
  // test4<T, Iter1, Iter2, forward_iterator<T*> >();
  test4<T, Iter1, Iter2, bidirectional_iterator<T*>>();
  test4<T, Iter1, Iter2, random_access_iterator<T*>>();
  test4<T, Iter1, Iter2, T*>();
}

template <class T, class Iter1>
__host__ __device__ constexpr void test2()
{
  test3<T, Iter1, cpp17_input_iterator<const T*>>();
  // test3<T, Iter1, forward_iterator<const T*> >();
  test3<T, Iter1, bidirectional_iterator<const T*>>();
  test3<T, Iter1, random_access_iterator<const T*>>();
  test3<T, Iter1, const T*>();
}

template <class T>
__host__ __device__ constexpr void test1()
{
  test2<T, cpp17_input_iterator<const T*>>();
  // test2<T, forward_iterator<const T*> >();
  test2<T, bidirectional_iterator<const T*>>();
  test2<T, random_access_iterator<const T*>>();
  test2<T, const T*>();
}

__host__ __device__ constexpr bool test()
{
  test1<TrivialSortableWithComp>();
  test1<NonTrivialSortableWithComp>();
  return true;
}

int main(int, char**)
{
  test();
#if defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
  static_assert(test(), "");
#endif // _CCCL_BUILTIN_IS_CONSTANT_EVALUATED

  return 0;
}
