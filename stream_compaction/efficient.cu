#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 128

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int offset, int *data) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (k >= n) return;
            if (k % (2 * offset) == 0) {
                data[k + offset * 2 - 1] += data[k + offset - 1];
            }
        }

        __global__ void kernDownSweep(int n, int offset, int *data) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (k >= n) return;
            if (k % (2 * offset) == 0) {
                int t = data[k + offset - 1];
                data[k + offset - 1] = data[k + offset * 2 - 1];
                data[k + offset * 2 - 1] = t + data[k + offset * 2 - 1];
            }
        }

        void scanOnDevice(int paddedN, int *dev_data) {
            const int blocks = (paddedN + blockSize - 1) / blockSize;
            const int levels = ilog2(paddedN);

            for (int d = 0; d < levels; d++) {
                kernUpSweep<<<blocks, blockSize>>>(paddedN, 1 << d, dev_data);
                checkCUDAError("kernUpSweep failed");
            }

            // setting the last element to 0
            cudaMemset(dev_data + paddedN - 1, 0, sizeof(int));

            for (int d = levels - 1; d >= 0; d--) {
                kernDownSweep<<<blocks, blockSize>>>(paddedN, 1 << d, dev_data);
                checkCUDAError("kernDownSweep failed");
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            if (n <= 0) return;

            const int paddedN = 1 << ilog2ceil(n);

            int *dev_data;
            cudaMalloc((void**)&dev_data, paddedN * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed");
            cudaMemset(dev_data, 0, paddedN * sizeof(int));
            checkCUDAError("cudaMemset dev_data failed");
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to device failed");

            timer().startGpuTimer();
            scanOnDevice(paddedN, dev_data);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy to host failed");
            cudaFree(dev_data);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            if (n <= 0 ) return 0;

            const int paddedN = 1 << ilog2ceil(n);
            const int blocks = (n + blockSize - 1) / blockSize;

            int *dev_idata, *dev_bools, *dev_indices, *dev_odata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed");
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_bools failed");
            cudaMalloc((void**)&dev_indices, paddedN * sizeof(int));
            checkCUDAError("cudaMalloc dev_indices failed");
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_idata to device failed");
            cudaMemset(dev_indices, 0, paddedN * sizeof(int));
            checkCUDAError("cudaMemset dev_indices failed");

            timer().startGpuTimer();
            Common::kernMapToBoolean<<<blocks, blockSize>>>(n, dev_bools, dev_idata);
            checkCUDAError("kernMapToBoolean failed");

            cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
            checkCUDAError("cudaMemcpy from dev_bools to dev_indices failed");
            scanOnDevice(paddedN, dev_indices);

            Common::kernScatter<<<blocks, blockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);
            checkCUDAError("kernScatter failed");
            timer().endGpuTimer();

            int lastIndex = 0;
            int lastBool = 0;
            cudaMemcpy(&lastIndex, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_indices to lastIndex failed");
            cudaMemcpy(&lastBool, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_bools to lastBool failed");
            int count = lastIndex + lastBool;

            cudaMemcpy(odata, dev_odata, count * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("copy to host failed");
            cudaFree(dev_idata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_odata);
            return count;
        }
    }
}
