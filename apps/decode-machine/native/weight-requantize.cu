// Diagnostic conversion only. Uses the same RNE primitive as deployed E2M1 materialization.
#include <stdint.h>
#ifndef __NV_SILENCE_DEPRECATION_BEGIN
#define __NV_SILENCE_DEPRECATION_BEGIN
#define __NV_SILENCE_DEPRECATION_END
#endif
#include "l1-hardware/mma/silicon_l1_e4m3_row_reduce.cuh"

// The input pair is a desired weight in units of its unchanged block scale.
// The transient decoded value stays within this conversion; only narrow codes cross its ports.
__device__ __forceinline__ uint8_t weight_requantize(E4m3ScalarPair value) {
    return static_cast<uint8_t>(e4m3s_row_reduce_e2m1_encode_rne(e4m3_pair_decode(value)));
}

extern "C" __global__ void qgre_weight_requantize(
    const uint8_t* codes, const uint8_t* scales, uint8_t* output) {
    const PackedU8x4Word at = blockIdx.x * blockDim.x + threadIdx.x;
    output[at] = weight_requantize(E4m3ScalarPair{E4m3ScalarCode{codes[at]}, E8m0ScalarScale{scales[at]}});
}
