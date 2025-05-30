/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/adl/uninitialized_fill.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/uninitialized_fill.h>
#include <thrust/uninitialized_fill.h>

THRUST_NAMESPACE_BEGIN

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename ForwardIterator, typename T>
_CCCL_HOST_DEVICE void uninitialized_fill(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  ForwardIterator first,
  ForwardIterator last,
  const T& x)
{
  _CCCL_NVTX_RANGE_SCOPE("uninitialized_fill");
  using thrust::system::detail::generic::uninitialized_fill;
  return uninitialized_fill(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, x);
} // end uninitialized_fill()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename ForwardIterator, typename Size, typename T>
_CCCL_HOST_DEVICE ForwardIterator uninitialized_fill_n(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec, ForwardIterator first, Size n, const T& x)
{
  _CCCL_NVTX_RANGE_SCOPE("uninitialized_fill_n");
  using thrust::system::detail::generic::uninitialized_fill_n;
  return uninitialized_fill_n(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, n, x);
} // end uninitialized_fill_n()

template <typename ForwardIterator, typename T>
void uninitialized_fill(ForwardIterator first, ForwardIterator last, const T& x)
{
  _CCCL_NVTX_RANGE_SCOPE("uninitialized_fill");
  using thrust::system::detail::generic::select_system;

  using System = typename thrust::iterator_system<ForwardIterator>::type;

  System system;

  thrust::uninitialized_fill(select_system(system), first, last, x);
} // end uninitialized_fill()

template <typename ForwardIterator, typename Size, typename T>
ForwardIterator uninitialized_fill_n(ForwardIterator first, Size n, const T& x)
{
  _CCCL_NVTX_RANGE_SCOPE("uninitialized_fill_n");
  using thrust::system::detail::generic::select_system;

  using System = typename thrust::iterator_system<ForwardIterator>::type;

  System system;

  return thrust::uninitialized_fill_n(select_system(system), first, n, x);
} // end uninitialized_fill_n()

THRUST_NAMESPACE_END
