//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>

// struct last_spec {
//   explicit last_spec() = default;
// };
//
// inline constexpr last_spec last{};

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using last_spec = cuda::std::chrono::last_spec;

  static_assert(cuda::std::is_same_v<const last_spec, decltype(cuda::std::chrono::last)>);

  static_assert(cuda::std::is_trivially_copyable_v<last_spec>, "");
  static_assert(cuda::std::is_standard_layout_v<last_spec>, "");

  return 0;
}
