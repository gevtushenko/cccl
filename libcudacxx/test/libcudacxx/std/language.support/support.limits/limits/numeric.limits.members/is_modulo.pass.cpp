//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// is_modulo

#include <cuda/std/limits>

#include "test_macros.h"

template <class T, bool expected>
__host__ __device__ void test()
{
  static_assert(cuda::std::numeric_limits<T>::is_modulo == expected, "is_modulo test 1");
  static_assert(cuda::std::numeric_limits<const T>::is_modulo == expected, "is_modulo test 2");
  static_assert(cuda::std::numeric_limits<volatile T>::is_modulo == expected, "is_modulo test 3");
  static_assert(cuda::std::numeric_limits<const volatile T>::is_modulo == expected, "is_modulo test 4");
}

int main(int, char**)
{
  test<bool, false>();
  //    test<char, false>(); // don't know
  test<signed char, false>();
  test<unsigned char, true>();
//    test<wchar_t, false>(); // don't know
#if TEST_STD_VER > 2017 && defined(__cpp_char8_t)
  test<char8_t, true>();
#endif
#ifndef _LIBCUDACXX_HAS_NO_UNICODE_CHARS
  test<char16_t, true>();
  test<char32_t, true>();
#endif // _LIBCUDACXX_HAS_NO_UNICODE_CHARS
  test<short, false>();
  test<unsigned short, true>();
  test<int, false>();
  test<unsigned int, true>();
  test<long, false>();
  test<unsigned long, true>();
  test<long long, false>();
  test<unsigned long long, true>();
#if _CCCL_HAS_INT128()
  test<__int128_t, false>();
  test<__uint128_t, true>();
#endif // _CCCL_HAS_INT128()
  test<float, false>();
  test<double, false>();
#ifndef _LIBCUDACXX_HAS_NO_LONG_DOUBLE
  test<long double, false>();
#endif
#if _CCCL_HAS_NVFP16()
  test<__half, false>();
#endif // _CCCL_HAS_NVFP16
#if _CCCL_HAS_NVBF16()
  test<__nv_bfloat16, false>();
#endif // _CCCL_HAS_NVBF16
#if _CCCL_HAS_NVFP8()
  test<__nv_fp8_e4m3, false>();
  test<__nv_fp8_e5m2, false>();
#endif // _CCCL_HAS_NVFP8()

  return 0;
}
