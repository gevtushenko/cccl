//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// epsilon()

#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cfloat>
#include <cuda/std/limits>

#include "common.h"
#include "test_macros.h"

template <class T>
__host__ __device__ void test(T expected)
{
  assert(float_eq(cuda::std::numeric_limits<T>::epsilon(), expected));
  assert(float_eq(cuda::std::numeric_limits<const T>::epsilon(), expected));
  assert(float_eq(cuda::std::numeric_limits<volatile T>::epsilon(), expected));
  assert(float_eq(cuda::std::numeric_limits<const volatile T>::epsilon(), expected));
}

int main(int, char**)
{
  test<bool>(false);
  test<char>(0);
  test<signed char>(0);
  test<unsigned char>(0);
  test<wchar_t>(0);
#if TEST_STD_VER > 2017 && defined(__cpp_char8_t)
  test<char8_t>(0);
#endif
  test<char16_t>(0);
  test<char32_t>(0);
  test<short>(0);
  test<unsigned short>(0);
  test<int>(0);
  test<unsigned int>(0);
  test<long>(0);
  test<unsigned long>(0);
  test<long long>(0);
  test<unsigned long long>(0);
#if _CCCL_HAS_INT128()
  test<__int128_t>(0);
  test<__uint128_t>(0);
#endif // _CCCL_HAS_INT128()
  test<float>(FLT_EPSILON);
  test<double>(DBL_EPSILON);
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>(LDBL_EPSILON);
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_NVFP16()
  test<__half>(__double2half(0.0009765625));
#endif // _CCCL_HAS_NVFP16
#if _CCCL_HAS_NVBF16()
  test<__nv_bfloat16>(__double2bfloat16(0.0078125));
#endif // _CCCL_HAS_NVBF16
#if _CCCL_HAS_NVFP8_E4M3()
  test<__nv_fp8_e4m3>(make_fp8_e4m3(0.125));
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
  test<__nv_fp8_e5m2>(make_fp8_e5m2(0.25));
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_E8M0()
  test<__nv_fp8_e8m0>(make_fp8_e8m0(1.0));
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
  test<__nv_fp6_e2m3>(make_fp6_e2m3(0.125));
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
  test<__nv_fp6_e3m2>(make_fp6_e3m2(0.25));
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
  test<__nv_fp4_e2m1>(make_fp4_e2m1(0.5));
#endif // _CCCL_HAS_NVFP4_E2M1()
#if _CCCL_HAS_FLOAT128()
  test<__float128>(cuda::std::bit_cast<__float128>(__uint128_t{0x3f8f000000000000} << 64));
#endif // _CCCL_HAS_FLOAT128()

  return 0;
}
