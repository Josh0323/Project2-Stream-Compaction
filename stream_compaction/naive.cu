#include <cuda.h>
#include <cuda_runtime.h>
#include <utility>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        int blockSize = 256;

        /**
         * One level of the naive scan (GPU Gems 3, Example 39-2, with the 2^(d-1) fix):
         * odata[k] = idata[k - offset] + idata[k] when k >= offset, otherwise a plain copy.
         * Reading and writing different buffers is what keeps the threads from racing.
         */
        __global__ void kernNaiveScanStep(int n, int offset, int *odata, const int *idata) {
            int k = blockIdx.x * blockDim.x + threadIdx.x;
            if (k >= n) {
                return;
            }
            odata[k] = k >= offset ? idata[k - offset] + idata[k] : idata[k];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            if (n <= 0) {
                return;
            }

            int *dev_in = nullptr;
            int *dev_out = nullptr;
            cudaMalloc(&dev_in, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_in");
            cudaMalloc(&dev_out, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_out");

            // This algorithm produces an inclusive scan. Uploading the input shifted right by
            // one, [0, x0, x1, ..., x(n-2)], turns that into the exclusive scan we want.
            cudaMemset(dev_in, 0, sizeof(int));
            if (n > 1) {
                cudaMemcpy(dev_in + 1, idata, (n - 1) * sizeof(int), cudaMemcpyHostToDevice);
            }
            checkCUDAError("upload idata");

            timer().startGpuTimer();
            dim3 blocksPerGrid((n + blockSize - 1) / blockSize);
            int levels = ilog2ceil(n);
            for (int d = 0; d < levels; d++) {
                kernNaiveScanStep<<<blocksPerGrid, blockSize>>>(n, 1 << d, dev_out, dev_in);
                std::swap(dev_in, dev_out);
            }
            timer().endGpuTimer();
            checkCUDAError("kernNaiveScanStep");

            // The last swap left the result in dev_in.
            cudaMemcpy(odata, dev_in, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("download result");

            cudaFree(dev_in);
            cudaFree(dev_out);
        }
    }
}
