//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class year_month;

// constexpr year_month& operator+=(const months& d) noexcept;
// constexpr year_month& operator-=(const months& d) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <typename D, typename Ds>
__host__ __device__ constexpr bool testConstexpr(D d1)
{
  if (static_cast<unsigned>((d1).month()) != 1)
  {
    return false;
  }
  if (static_cast<unsigned>((d1 += Ds{1}).month()) != 2)
  {
    return false;
  }
  if (static_cast<unsigned>((d1 += Ds{2}).month()) != 4)
  {
    return false;
  }
  if (static_cast<unsigned>((d1 += Ds{12}).month()) != 4)
  {
    return false;
  }
  if (static_cast<unsigned>((d1 -= Ds{1}).month()) != 3)
  {
    return false;
  }
  if (static_cast<unsigned>((d1 -= Ds{2}).month()) != 1)
  {
    return false;
  }
  if (static_cast<unsigned>((d1 -= Ds{12}).month()) != 1)
  {
    return false;
  }
  return true;
}

int main(int, char**)
{
  using month      = cuda::std::chrono::month;
  using months     = cuda::std::chrono::months;
  using year       = cuda::std::chrono::year;
  using year_month = cuda::std::chrono::year_month;

  static_assert(noexcept(cuda::std::declval<year_month&>() += cuda::std::declval<months>()));
  static_assert(
    cuda::std::is_same_v<year_month&, decltype(cuda::std::declval<year_month&>() += cuda::std::declval<months>())>);

  static_assert(noexcept(cuda::std::declval<year_month&>() -= cuda::std::declval<months>()));
  static_assert(
    cuda::std::is_same_v<year_month&, decltype(cuda::std::declval<year_month&>() -= cuda::std::declval<months>())>);

  static_assert(testConstexpr<year_month, months>(year_month{year{1234}, month{1}}), "");

  for (unsigned i = 0; i <= 10; ++i)
  {
    year y{1234};
    year_month ym(y, month{i});
    assert(static_cast<unsigned>((ym += months{2}).month()) == i + 2);
    assert(ym.year() == y);
    assert(static_cast<unsigned>((ym).month()) == i + 2);
    assert(ym.year() == y);
    assert(static_cast<unsigned>((ym -= months{1}).month()) == i + 1);
    assert(ym.year() == y);
    assert(static_cast<unsigned>((ym).month()) == i + 1);
    assert(ym.year() == y);
  }

  return 0;
}
