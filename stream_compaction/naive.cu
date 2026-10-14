#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <utility>

#define blockSize 128

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        // this is inclusive scan
        __global__ void kernNaiveScanPass(int n, int offset, int *odata, const int *idata) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (k >= n) return;
            if (k >= offset) {
                odata[k] = idata[k - offset] + idata[k];
            } else {
                odata[k] = idata[k];
            }
        }

        //shift right to make it inclusive scan to exclusive scan
        __global__ void kernShiftRight(int n, int *odata, const int *idata) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (k >= n) return;
            odata[k] = (k > 0) ? idata[k - 1] : 0;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            if (n <= 0) return;

            int *dev_bufA;
            int *dev_bufB;
            cudaMalloc((void**)&dev_bufA, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_bufA failed");
            cudaMalloc((void**)&dev_bufB, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_bufB failed!");
            cudaMemcpy(dev_bufA, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to device failed");

            const int blocks = (n + blockSize - 1) / blockSize;

            timer().startGpuTimer();
            // TODO
            for (int d = 0; d <ilog2ceil(n); d++) {
                const int offset = 1 << d;
                kernNaiveScanPass<<<blocks, blockSize>>>(n, offset, dev_bufB, dev_bufA);
                checkCUDAError("kernNaiveScanPass failed");
                std::swap(dev_bufA, dev_bufB);
            }
            kernShiftRight<<<blocks, blockSize>>>(n, dev_bufB, dev_bufA);
            checkCUDAError("kernShiftRight failed");
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_bufB, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy to host failed");
            cudaFree(dev_bufA);
            cudaFree(dev_bufB);
        }
    }
}
