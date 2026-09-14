#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <utility>
#include "common.h"
#include "efficient.h"
#include "radix.h"

namespace StreamCompaction {
    namespace Radix {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        int blockSize = 256;

        // For a normal bit, elements with a 0 go first. For the sign bit it's flipped so
        // negatives end up in front of non-negatives.
        __host__ __device__ inline int goesFirst(int value, int bit) {
            int b = (value >> bit) & 1;
            return bit == 31 ? b : 1 - b;
        }

        /**
         * odata[i] = idata[i] ^ first, zero in the padding. OR-ing this over the whole array
         * leaves exactly the bits that differ somewhere; every other bit is the same in all
         * elements, so a pass over it wouldn't move anything.
         */
        __global__ void kernXorWithFirst(int paddedN, int n, int first, int *odata, const int *idata) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= paddedN) {
                return;
            }
            odata[i] = i < n ? idata[i] ^ first : 0;
        }

        // Same shape as the up-sweep, with | instead of +. The root ends up with the OR of everything.
        __global__ void kernReduceOr(int numNodes, int stride, int *data) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= numNodes) {
                return;
            }
            int right = (i + 1) * stride - 1;
            data[right] |= data[right - stride / 2];
        }

        // e array from the chapter: 1 where the element goes to the front half for this bit.
        __global__ void kernMapGoesFirst(int paddedN, int n, int bit, int *e, const int *idata) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= paddedN) {
                return;
            }
            e[i] = i < n ? goesFirst(idata[i], bit) : 0;
        }

        /**
         * With f = scan(e), front elements keep their relative order at f[i], and the rest
         * go to t[i] = i - f[i] + totalFalses. e is recomputed from the bit so it doesn't
         * need its own buffer.
         */
        __global__ void kernRadixScatter(int n, int bit, int totalFalses,
                int *odata, const int *idata, const int *f) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n) {
                return;
            }
            int dst = goesFirst(idata[i], bit) ? f[i] : i - f[i] + totalFalses;
            odata[dst] = idata[i];
        }

        static dim3 gridFor(int count, int threads) {
            return dim3((count + threads - 1) / threads);
        }

        void sort(int n, int *odata, const int *idata) {
            if (n <= 0) {
                return;
            }
            int pow2 = 1 << ilog2ceil(n);

            int *dev_in = nullptr;
            int *dev_out = nullptr;
            int *dev_f = nullptr;
            cudaMalloc(&dev_in, n * sizeof(int));
            cudaMalloc(&dev_out, n * sizeof(int));
            cudaMalloc(&dev_f, pow2 * sizeof(int));
            checkCUDAError("cudaMalloc radix buffers");

            cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("upload idata");

            timer().startGpuTimer();
            int threadsPadded = std::min(blockSize, pow2);
            int threadsN = std::min(blockSize, n);

            // Find the bits worth sorting on.
            kernXorWithFirst<<<gridFor(pow2, threadsPadded), threadsPadded>>>(pow2, n, idata[0], dev_f, dev_in);
            for (int d = 0; d < ilog2(pow2); d++) {
                int numNodes = pow2 >> (d + 1);
                int threads = std::min(blockSize, numNodes);
                kernReduceOr<<<gridFor(numNodes, threads), threads>>>(numNodes, 1 << (d + 1), dev_f);
            }
            unsigned int varyingBits = 0;
            cudaMemcpy(&varyingBits, dev_f + pow2 - 1, sizeof(int), cudaMemcpyDeviceToHost);

            // LSD order, so the sign bit (if it varies) is always the last pass.
            for (int bit = 0; bit < 32; bit++) {
                if (((varyingBits >> bit) & 1u) == 0) {
                    continue;
                }
                kernMapGoesFirst<<<gridFor(pow2, threadsPadded), threadsPadded>>>(pow2, n, bit, dev_f, dev_in);
                Efficient::scanDevice(pow2, dev_f);

                // totalFalses = f[n-1] + e[n-1]
                int lastF = 0;
                int lastValue = 0;
                cudaMemcpy(&lastF, dev_f + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&lastValue, dev_in + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                int totalFalses = lastF + goesFirst(lastValue, bit);

                kernRadixScatter<<<gridFor(n, threadsN), threadsN>>>(n, bit, totalFalses, dev_out, dev_in, dev_f);
                std::swap(dev_in, dev_out);
            }
            timer().endGpuTimer();
            checkCUDAError("radix sort");

            cudaMemcpy(odata, dev_in, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("download result");

            cudaFree(dev_in);
            cudaFree(dev_out);
            cudaFree(dev_f);
        }
    }
}
