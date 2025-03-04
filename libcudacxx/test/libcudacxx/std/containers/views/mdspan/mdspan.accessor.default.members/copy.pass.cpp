//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc && c++14, msvc && c++17

#include <cuda/std/cassert>
#include <cuda/std/mdspan>

int main(int, char**)
{
  {
    using element_t = int;
    cuda::std::array<element_t, 2> d{42, 43};
    cuda::std::default_accessor<element_t> a0;
    cuda::std::default_accessor<element_t> a(a0);

    assert(a.access(d.data(), 0) == 42);
    assert(a.access(d.data(), 1) == 43);
    assert(a.offset(d.data(), 0) == d.data());
    assert(a.offset(d.data(), 1) == d.data() + 1);
  }

  return 0;
}
