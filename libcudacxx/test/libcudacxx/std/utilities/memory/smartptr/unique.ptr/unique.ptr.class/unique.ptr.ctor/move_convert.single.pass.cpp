//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// Test unique_ptr converting move ctor

// NOTE: unique_ptr does not provide converting constructors in C++03

#include <cuda/std/__memory_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"
#include "unique_ptr_test_helper.h"

// test converting move ctor.  Should only require a MoveConstructible deleter, or if
//    deleter is a reference, not even that.
// Explicit version

template <class LHS, class RHS>
__host__ __device__ TEST_CONSTEXPR_CXX23 void checkReferenceDeleter(LHS& lhs, RHS& rhs)
{
  typedef typename LHS::deleter_type NewDel;
  static_assert(cuda::std::is_reference<NewDel>::value, "");
  rhs.get_deleter().set_state(42);
  assert(rhs.get_deleter().state() == 42);
  assert(lhs.get_deleter().state() == 42);
  lhs.get_deleter().set_state(99);
  assert(lhs.get_deleter().state() == 99);
  assert(rhs.get_deleter().state() == 99);
}

template <class LHS, class RHS>
__host__ __device__ TEST_CONSTEXPR_CXX23 void checkDeleter(LHS& lhs, RHS& rhs, int LHSVal, int RHSVal)
{
  assert(lhs.get_deleter().state() == LHSVal);
  assert(rhs.get_deleter().state() == RHSVal);
}

template <class LHS, class RHS>
__host__ __device__ TEST_CONSTEXPR_CXX23 void checkCtor(LHS& lhs, RHS& rhs, A* RHSVal)
{
  assert(lhs.get() == RHSVal);
  assert(rhs.get() == nullptr);
  if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
  {
    assert(A_count == 1);
    assert(B_count == 1);
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX23 void checkNoneAlive()
{
  if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
  {
    assert(A_count == 0);
    assert(B_count == 0);
  }
}

template <class T>
struct NCConvertingDeleter
{
  TEST_CONSTEXPR_CXX23 NCConvertingDeleter()                      = default;
  NCConvertingDeleter(NCConvertingDeleter const&)                 = delete;
  TEST_CONSTEXPR_CXX23 NCConvertingDeleter(NCConvertingDeleter&&) = default;

  template <class U>
  __host__ __device__ TEST_CONSTEXPR_CXX23 NCConvertingDeleter(NCConvertingDeleter<U>&&)
  {}

  __host__ __device__ TEST_CONSTEXPR_CXX23 void operator()(T*) const {}
};

template <class T>
struct NCConvertingDeleter<T[]>
{
  TEST_CONSTEXPR_CXX23 NCConvertingDeleter()                      = default;
  NCConvertingDeleter(NCConvertingDeleter const&)                 = delete;
  TEST_CONSTEXPR_CXX23 NCConvertingDeleter(NCConvertingDeleter&&) = default;

  template <class U>
  __host__ __device__ TEST_CONSTEXPR_CXX23 NCConvertingDeleter(NCConvertingDeleter<U>&&)
  {}

  __host__ __device__ TEST_CONSTEXPR_CXX23 void operator()(T*) const {}
};

struct NCGenericDeleter
{
  TEST_CONSTEXPR_CXX23 NCGenericDeleter()                   = default;
  NCGenericDeleter(NCGenericDeleter const&)                 = delete;
  TEST_CONSTEXPR_CXX23 NCGenericDeleter(NCGenericDeleter&&) = default;

  __host__ __device__ TEST_CONSTEXPR_CXX23 void operator()(void*) const {}
};

__host__ __device__ TEST_CONSTEXPR_CXX23 void test_sfinae()
{
  using DA  = NCConvertingDeleter<A>; // non-copyable deleters
  using DB  = NCConvertingDeleter<B>;
  using UA  = cuda::std::unique_ptr<A>;
  using UB  = cuda::std::unique_ptr<B>;
  using UAD = cuda::std::unique_ptr<A, DA>;
  using UBD = cuda::std::unique_ptr<B, DB>;
  { // cannot move from an lvalue
    static_assert(cuda::std::is_constructible<UA, UB&&>::value, "");
    static_assert(!cuda::std::is_constructible<UA, UB&>::value, "");
    static_assert(!cuda::std::is_constructible<UA, const UB&>::value, "");
  }
  { // cannot move if the deleter-types cannot convert
    static_assert(cuda::std::is_constructible<UAD, UBD&&>::value, "");
    static_assert(!cuda::std::is_constructible<UAD, UB&&>::value, "");
    static_assert(!cuda::std::is_constructible<UA, UBD&&>::value, "");
  }
  { // cannot move-convert with reference deleters of different types
    using UA1 = cuda::std::unique_ptr<A, DA&>;
    using UB1 = cuda::std::unique_ptr<B, DB&>;
    static_assert(!cuda::std::is_constructible<UA1, UB1&&>::value, "");
  }
  { // cannot move-convert with reference deleters of different types
    using UA1 = cuda::std::unique_ptr<A, const DA&>;
    using UB1 = cuda::std::unique_ptr<B, const DB&>;
    static_assert(!cuda::std::is_constructible<UA1, UB1&&>::value, "");
  }
  { // cannot move-convert from unique_ptr<Array[]>
    using UA1 = cuda::std::unique_ptr<A>;
    using UA2 = cuda::std::unique_ptr<A[]>;
    using UB1 = cuda::std::unique_ptr<B[]>;
    static_assert(!cuda::std::is_constructible<UA1, UA2&&>::value, "");
    static_assert(!cuda::std::is_constructible<UA1, UB1&&>::value, "");
  }
  { // cannot move-convert from unique_ptr<Array[]>
    using UA1 = cuda::std::unique_ptr<A, NCGenericDeleter>;
    using UA2 = cuda::std::unique_ptr<A[], NCGenericDeleter>;
    using UB1 = cuda::std::unique_ptr<B[], NCGenericDeleter>;
    static_assert(!cuda::std::is_constructible<UA1, UA2&&>::value, "");
    static_assert(!cuda::std::is_constructible<UA1, UB1&&>::value, "");
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX23 void test_noexcept()
{
  {
    typedef cuda::std::unique_ptr<A> APtr;
    typedef cuda::std::unique_ptr<B> BPtr;
    static_assert(cuda::std::is_nothrow_constructible<APtr, BPtr>::value, "");
  }
  {
    typedef cuda::std::unique_ptr<A, Deleter<A>> APtr;
    typedef cuda::std::unique_ptr<B, Deleter<B>> BPtr;
    static_assert(cuda::std::is_nothrow_constructible<APtr, BPtr>::value, "");
  }
  {
    typedef cuda::std::unique_ptr<A, NCDeleter<A>&> APtr;
    typedef cuda::std::unique_ptr<B, NCDeleter<A>&> BPtr;
    static_assert(cuda::std::is_nothrow_constructible<APtr, BPtr>::value, "");
  }
  {
    typedef cuda::std::unique_ptr<A, const NCConstDeleter<A>&> APtr;
    typedef cuda::std::unique_ptr<B, const NCConstDeleter<A>&> BPtr;
    static_assert(cuda::std::is_nothrow_constructible<APtr, BPtr>::value, "");
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX23 bool test()
{
  {
    test_sfinae();
    test_noexcept();
  }
  {
    typedef cuda::std::unique_ptr<A> APtr;
    typedef cuda::std::unique_ptr<B> BPtr;
    { // explicit
      BPtr b(new B);
      A* p = b.get();
      APtr a(cuda::std::move(b));
      checkCtor(a, b, p);
    }
    checkNoneAlive();
    { // implicit
      BPtr b(new B);
      A* p   = b.get();
      APtr a = cuda::std::move(b);
      checkCtor(a, b, p);
    }
    checkNoneAlive();
  }
  { // test with moveable deleters
    typedef cuda::std::unique_ptr<A, Deleter<A>> APtr;
    typedef cuda::std::unique_ptr<B, Deleter<B>> BPtr;
    {
      Deleter<B> del(5);
      BPtr b(new B, cuda::std::move(del));
      A* p = b.get();
      APtr a(cuda::std::move(b));
      checkCtor(a, b, p);
      checkDeleter(a, b, 5, 0);
    }
    checkNoneAlive();
    {
      Deleter<B> del(5);
      BPtr b(new B, cuda::std::move(del));
      A* p   = b.get();
      APtr a = cuda::std::move(b);
      checkCtor(a, b, p);
      checkDeleter(a, b, 5, 0);
    }
    checkNoneAlive();
  }
  { // test with reference deleters
    typedef cuda::std::unique_ptr<A, NCDeleter<A>&> APtr;
    typedef cuda::std::unique_ptr<B, NCDeleter<A>&> BPtr;
    NCDeleter<A> del(5);
    {
      BPtr b(new B, del);
      A* p = b.get();
      APtr a(cuda::std::move(b));
      checkCtor(a, b, p);
      checkReferenceDeleter(a, b);
    }
    checkNoneAlive();
    {
      BPtr b(new B, del);
      A* p   = b.get();
      APtr a = cuda::std::move(b);
      checkCtor(a, b, p);
      checkReferenceDeleter(a, b);
    }
    checkNoneAlive();
  }
  {
    typedef cuda::std::unique_ptr<A, CDeleter<A>> APtr;
    typedef cuda::std::unique_ptr<B, CDeleter<B>&> BPtr;
    CDeleter<B> del(5);
    {
      BPtr b(new B, del);
      A* p = b.get();
      APtr a(cuda::std::move(b));
      checkCtor(a, b, p);
      checkDeleter(a, b, 5, 5);
    }
    checkNoneAlive();
    {
      BPtr b(new B, del);
      A* p   = b.get();
      APtr a = cuda::std::move(b);
      checkCtor(a, b, p);
      checkDeleter(a, b, 5, 5);
    }
    checkNoneAlive();
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2023
  static_assert(test());
#endif // TEST_STD_VER >= 2023

  return 0;
}
