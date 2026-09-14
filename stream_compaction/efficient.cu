#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        int blockSize = 256;

        /**
         * One up-sweep level. Thread i owns the i-th node of the level, so only
         * n / stride threads get launched and none of them are idle:
         *   data[k + stride - 1] += data[k + stride/2 - 1], with k = i * stride.
         */
        __global__ void kernUpSweep(int numNodes, int stride, int *data) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= numNodes) {
                return;
            }
            int right = (i + 1) * stride - 1;
            int left = right - stride / 2;
            data[right] += data[left];
        }

        /**
         * One down-sweep level (GPU Gems 3, Example 39-4, with the 2^(d+1) fix):
         * the left child takes the parent's value, the right child gets parent + old left.
         */
        __global__ void kernDownSweep(int numNodes, int stride, int *data) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= numNodes) {
                return;
            }
            int right = (i + 1) * stride - 1;
            int left = right - stride / 2;
            int t = data[left];
            data[left] = data[right];
            data[right] += t;
        }

        // Straight-from-the-slides versions: one thread per element, most do nothing.
        __global__ void kernUpSweepAllThreads(int n, int stride, int *data) {
            int k = blockIdx.x * blockDim.x + threadIdx.x;
            if (k >= n || k % stride != 0) {
                return;
            }
            data[k + stride - 1] += data[k + stride / 2 - 1];
        }

        __global__ void kernDownSweepAllThreads(int n, int stride, int *data) {
            int k = blockIdx.x * blockDim.x + threadIdx.x;
            if (k >= n || k % stride != 0) {
                return;
            }
            int left = k + stride / 2 - 1;
            int right = k + stride - 1;
            int t = data[left];
            data[left] = data[right];
            data[right] += t;
        }

        void scanDevice(int n, int *dev_data) {
            int levels = ilog2(n);

            for (int d = 0; d < levels; d++) {
                int numNodes = n >> (d + 1);
                // Deep levels have only a handful of nodes; don't launch a full block for them.
                int threads = std::min(blockSize, numNodes);
                dim3 blocksPerGrid((numNodes + threads - 1) / threads);
                kernUpSweep<<<blocksPerGrid, threads>>>(numNodes, 1 << (d + 1), dev_data);
            }

            // The root now holds the total. Replace it with the identity and sweep back down.
            cudaMemset(dev_data + n - 1, 0, sizeof(int));

            for (int d = levels - 1; d >= 0; d--) {
                int numNodes = n >> (d + 1);
                int threads = std::min(blockSize, numNodes);
                dim3 blocksPerGrid((numNodes + threads - 1) / threads);
                kernDownSweep<<<blocksPerGrid, threads>>>(numNodes, 1 << (d + 1), dev_data);
            }
            checkCUDAError("work-efficient scan");
        }

        static void scanDeviceAllThreads(int n, int *dev_data) {
            int levels = ilog2(n);
            dim3 blocksPerGrid((n + blockSize - 1) / blockSize);

            for (int d = 0; d < levels; d++) {
                kernUpSweepAllThreads<<<blocksPerGrid, blockSize>>>(n, 1 << (d + 1), dev_data);
            }

            cudaMemset(dev_data + n - 1, 0, sizeof(int));

            for (int d = levels - 1; d >= 0; d--) {
                kernDownSweepAllThreads<<<blocksPerGrid, blockSize>>>(n, 1 << (d + 1), dev_data);
            }
            checkCUDAError("unoptimized work-efficient scan");
        }

        /**
         * Uploads idata into a zero-padded power-of-two buffer, times the device scan,
         * and copies the first n results back.
         */
        static void scanHostArray(int n, int *odata, const int *idata, void (*scanOnDevice)(int, int *)) {
            if (n <= 0) {
                return;
            }
            int pow2 = 1 << ilog2ceil(n);

            int *dev_data = nullptr;
            cudaMalloc(&dev_data, pow2 * sizeof(int));
            checkCUDAError("cudaMalloc dev_data");

            // cudaMalloc doesn't clear memory, and the padding has to be the identity (0).
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            if (pow2 > n) {
                cudaMemset(dev_data + n, 0, (pow2 - n) * sizeof(int));
            }
            checkCUDAError("upload idata");

            timer().startGpuTimer();
            scanOnDevice(pow2, dev_data);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("download result");
            cudaFree(dev_data);
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            scanHostArray(n, odata, idata, scanDevice);
        }

        void scanUnoptimized(int n, int *odata, const int *idata) {
            scanHostArray(n, odata, idata, scanDeviceAllThreads);
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
            if (n <= 0) {
                return 0;
            }
            int pow2 = 1 << ilog2ceil(n);

            int *dev_idata = nullptr;
            int *dev_bools = nullptr;
            int *dev_indices = nullptr;
            int *dev_odata = nullptr;
            cudaMalloc(&dev_idata, n * sizeof(int));
            cudaMalloc(&dev_bools, n * sizeof(int));
            cudaMalloc(&dev_indices, pow2 * sizeof(int));
            cudaMalloc(&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc compact buffers");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            if (pow2 > n) {
                cudaMemset(dev_indices + n, 0, (pow2 - n) * sizeof(int));
            }
            checkCUDAError("upload idata");

            timer().startGpuTimer();
            int threads = std::min(blockSize, n);
            dim3 blocksPerGrid((n + threads - 1) / threads);

            Common::kernMapToBoolean<<<blocksPerGrid, threads>>>(n, dev_bools, dev_idata);

            // The scan runs in place, so give it a copy and keep the bools for scatter.
            cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
            scanDevice(pow2, dev_indices);

            Common::kernScatter<<<blocksPerGrid, threads>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);

            // Everything before the last element is counted by its index; add the last one if kept.
            int lastBool = 0;
            int lastIndex = 0;
            cudaMemcpy(&lastBool, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastIndex, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int count = lastIndex + lastBool;
            timer().endGpuTimer();
            checkCUDAError("work-efficient compact");

            cudaMemcpy(odata, dev_odata, count * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("download result");

            cudaFree(dev_idata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_odata);
            return count;
        }
    }
}
