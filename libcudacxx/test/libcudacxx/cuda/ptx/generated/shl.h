// This file was automatically generated. Do not edit.

// We use a special strategy to force the generation of the PTX. This is mainly
// a fight against dead-code-elimination in the NVVM layer.
//
// The reason we need this strategy is because certain older versions of ptxas
// segfault when a non-sensical sequence of PTX is generated. So instead, we try
// to force the instantiation and compilation to PTX of all the overloads of the
// PTX wrapping functions.
//
// We do this by writing a function pointer of each overload to the kernel
// parameter `fn_ptr`.
//
// Because `fn_ptr` is possibly visible outside this translation unit, the
// compiler must compile all the functions which are stored.

__global__ void test_shl(void** fn_ptr)
{
#if __cccl_ptx_isa >= 100
  NV_IF_TARGET(NV_PROVIDES_SM_50,
               (
                   // shl.b16 dest, a_reg, b_reg;
                   * fn_ptr++ = reinterpret_cast<void*>(static_cast<int16_t (*)(int16_t, uint32_t)>(cuda::ptx::shl));));
#endif // __cccl_ptx_isa >= 100

#if __cccl_ptx_isa >= 100
  NV_IF_TARGET(NV_PROVIDES_SM_50,
               (
                   // shl.b32 dest, a_reg, b_reg;
                   * fn_ptr++ = reinterpret_cast<void*>(static_cast<int32_t (*)(int32_t, uint32_t)>(cuda::ptx::shl));));
#endif // __cccl_ptx_isa >= 100

#if __cccl_ptx_isa >= 100
  NV_IF_TARGET(NV_PROVIDES_SM_50,
               (
                   // shl.b64 dest, a_reg, b_reg;
                   * fn_ptr++ = reinterpret_cast<void*>(static_cast<int64_t (*)(int64_t, uint32_t)>(cuda::ptx::shl));));
#endif // __cccl_ptx_isa >= 100
}
