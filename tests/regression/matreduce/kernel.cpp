// #include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include "common.h"

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);

    int warpId = static_cast<int>(csr_read(VX_CSR_CTA_ID));
    int warpSize = vx_num_threads();
    int threadId = vx_thread_id();

    // Calculate global thread ID (similar to vecadd)
    int row = warpId + threadId;

    TYPE* A = reinterpret_cast<TYPE*>(arg->buf_A);
    TYPE* B = reinterpret_cast<TYPE*>(arg->buf_B);
    uint32_t rows = arg->rows;
    uint32_t cols = arg->cols;

    // Each thread sums one row
    if ((uint32_t)row < rows) {
        TYPE sum = 0;
        for (uint32_t c = 0; c < cols; ++c) {
            sum += A[row * cols + c];
        }
        B[row] = sum;
    }

    // Spin wait to ensure all row sums are complete (placeholder for barrier)
    int spin = 0;
    for (int i = 0; i < 1000; i++) {
        spin++;
    }

    // Only warp 0, thread 0 launches the child kernel
    if (warpId == 0 && threadId == 0) {
        // Set up child kernel parameters
        child_params_t* cp = reinterpret_cast<child_params_t*>(arg->child_params);
        cp->buf_B = arg->buf_B;
        cp->buf_C = arg->buf_C;
        cp->num_rows = rows;

        // Child kernel dimensions: single thread
        uint32_t grid_dim[3] = {1, 1, 1};
        uint32_t block_dim[3] = {1, 1, 1};

        // Launch child kernel dynamically
        dynamic_kernel_launch(arg->child_pc, grid_dim, block_dim, arg->child_params);
    }

    // Terminate all warps except warp 0 (similar to vecadd)
    vx_tmc(warpId == 0);

    return 0;
}
