//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>

// not1
//  deprecated in C++17

// UNSUPPORTED: clang-4.0
// REQUIRES: verify-support

#include <cuda/std/functional>

#include "test_macros.h"

struct Predicate
{
  typedef int argument_type;
  bool operator()(argument_type) const
  {
    return true;
  }
};

int main(int, char**)
{
  cuda::std::not1(Predicate()); // expected-error{{'not1<Predicate>' is deprecated}}

  return 0;
}
