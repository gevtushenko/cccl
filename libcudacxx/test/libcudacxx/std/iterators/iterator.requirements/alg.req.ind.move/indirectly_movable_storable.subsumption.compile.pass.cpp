//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++17

// template<class In, class Out>
// concept indirectly_movable_storable;

#include <cuda/std/iterator>

template <class I, class O>
  requires cuda::std::indirectly_movable<I, O>
__host__ __device__ constexpr bool indirectly_movable_storable_subsumption()
{
  return false;
}

template <class I, class O>
  requires cuda::std::indirectly_movable_storable<I, O>
__host__ __device__ constexpr bool indirectly_movable_storable_subsumption()
{
  return true;
}

static_assert(indirectly_movable_storable_subsumption<int*, int*>(), "");

int main(int, char**)
{
  return 0;
}
