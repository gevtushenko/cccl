//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// duration

// template <class Rep1, class Period, class Rep2>
//   constexpr
//   duration<typename common_type<Rep1, Rep2>::type, Period>
//   operator*(const duration<Rep1, Period>& d, const Rep2& s);

// template <class Rep1, class Period, class Rep2>
//   constexpr
//   duration<typename common_type<Rep1, Rep2>::type, Period>
//   operator*(const Rep1& s, const duration<Rep2, Period>& d);

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "test_macros.h"

int main(int, char**)
{
  {
    cuda::std::chrono::nanoseconds ns(3);
    ns = ns * 5;
    assert(ns.count() == 15);
    ns = 6 * ns;
    assert(ns.count() == 90);
  }
  {
    constexpr cuda::std::chrono::nanoseconds ns(3);
    constexpr cuda::std::chrono::nanoseconds ns2 = ns * 5;
    static_assert(ns2.count() == 15, "");
    constexpr cuda::std::chrono::nanoseconds ns3 = 6 * ns;
    static_assert(ns3.count() == 18, "");
  }

  return 0;
}
