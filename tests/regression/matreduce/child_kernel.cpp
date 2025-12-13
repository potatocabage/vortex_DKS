// #include <vx_spawn.h>
#include <vx_intrinsics.h>
#include "common.h"

// Child kernel: sums all row sums from buffer B into final result C
// This kernel is launched dynamically and runs with a single thread
int main() {
    child_params_t* params = (child_params_t*)csr_read(VX_CSR_MSCRATCH);

    int warpId = static_cast<int>(csr_read(VX_CSR_CTA_ID));
    int threadId = vx_thread_id();

    TYPE* B = reinterpret_cast<TYPE*>(params->buf_B);
    TYPE* C = reinterpret_cast<TYPE*>(params->buf_C);
    uint32_t num_rows = params->num_rows;

    // Single thread sums all row results
    // Only thread 0 of warp 0 performs the computation
    if (warpId == 0 && threadId == 0) {
        TYPE sum = 0;
        for (uint32_t i = 0; i < num_rows; ++i) {
            sum += B[i];
        }
        C[0] = sum;
    }

    // Terminate all warps except warp 0
    vx_tmc(warpId == 0);

    return 0;
}
