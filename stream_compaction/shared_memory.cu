#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "common.h"
#include "shared_memory.h"

namespace StreamCompaction {
    namespace SharedMemory {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        int blockSize = 256;
        int logNumBanks = 4;

        /**
         * Shared memory slot for logical index i when avoiding bank conflicts: leaves a gap
         * every 2^logBanks slots so threads striding through the tree hit different banks.
         * This is Listing 39-3's CONFLICT_FREE_OFFSET; the online version shifts by NUM_BANKS
         * and has no parentheses, so + binds before >>. logBanks == 0 disables it.
         */
        __host__ __device__ static inline int bankSlot(int i, int logBanks) {
            return logBanks > 0 ? i + (i >> logBanks) + (i >> (2 * logBanks)) : i;
        }

        static int efficientSharedInts(int elemsPerBlock, int logBanks) {
            return bankSlot(elemsPerBlock - 1, logBanks) + 1;
        }

        /**
         * Example 39-1 for one block of B threads. Writes the block's exclusive scan to
         * g_odata and its total to g_sums[block].
         *
         * The online listing does temp[pout] += temp[pin - offset], which adds to a value
         * from two passes ago (uninitialized memory on the first pass). Both terms have to
         * come from the pin buffer.
         */
        __global__ void kernNaiveScanBlock(int B, int *g_odata, const int *g_idata, int *g_sums) {
            extern __shared__ int temp[];
            int thid = threadIdx.x;
            int base = blockIdx.x * B;
            int pout = 0;

            // Shifting the block right by one makes the inclusive algorithm exclusive.
            temp[thid] = thid > 0 ? g_idata[base + thid - 1] : 0;
            __syncthreads();

            for (int offset = 1; offset < B; offset *= 2) {
                pout = 1 - pout;
                int pin = 1 - pout;
                temp[pout * B + thid] = thid >= offset
                    ? temp[pin * B + thid] + temp[pin * B + thid - offset]
                    : temp[pin * B + thid];
                __syncthreads();
            }

            g_odata[base + thid] = temp[pout * B + thid];
            if (thid == B - 1) {
                g_sums[blockIdx.x] = temp[pout * B + thid] + g_idata[base + thid];
            }
        }

        /**
         * Example 39-2 for one block of B elements handled by B/2 threads. With logBanks > 0
         * it uses the 39.2.3 layout: load from the two halves instead of adjacent pairs, and
         * map every index through bankSlot.
         */
        __global__ void kernEfficientScanBlock(int B, int logBanks,
                int *g_odata, const int *g_idata, int *g_sums) {
            extern __shared__ int temp[];
            int thid = threadIdx.x;
            int half = B / 2;
            int base = blockIdx.x * B;

            int ai = logBanks > 0 ? thid : 2 * thid;
            int bi = logBanks > 0 ? thid + half : 2 * thid + 1;
            temp[bankSlot(ai, logBanks)] = g_idata[base + ai];
            temp[bankSlot(bi, logBanks)] = g_idata[base + bi];

            int offset = 1;
            for (int d = half; d > 0; d >>= 1) {
                __syncthreads();
                if (thid < d) {
                    int l = bankSlot(offset * (2 * thid + 1) - 1, logBanks);
                    int r = bankSlot(offset * (2 * thid + 2) - 1, logBanks);
                    temp[r] += temp[l];
                }
                offset *= 2;
            }

            if (thid == 0) {
                int root = bankSlot(B - 1, logBanks);
                g_sums[blockIdx.x] = temp[root];
                temp[root] = 0;
            }

            for (int d = 1; d < B; d *= 2) {
                offset >>= 1;
                __syncthreads();
                if (thid < d) {
                    int l = bankSlot(offset * (2 * thid + 1) - 1, logBanks);
                    int r = bankSlot(offset * (2 * thid + 2) - 1, logBanks);
                    int t = temp[l];
                    temp[l] = temp[r];
                    temp[r] += t;
                }
            }
            __syncthreads();

            g_odata[base + ai] = temp[bankSlot(ai, logBanks)];
            g_odata[base + bi] = temp[bankSlot(bi, logBanks)];
        }

        // data[i] += incr[i / B]: turns per-block scans into a scan of the whole array.
        __global__ void kernAddBlockOffsets(int n, int logB, int *data, const int *incr) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n) {
                return;
            }
            data[i] += incr[i >> logB];
        }

        static int roundUp(int x, int multiple) {
            return (x + multiple - 1) / multiple * multiple;
        }

        struct Level {
            int length;     // multiple of the block's element count
            int *dev_in;    // level 0: the input; level k+1: block sums of level k
            int *dev_out;   // scan of dev_in
        };

        /**
         * GPU Gems 3, 39.2.4. Every level scans its blocks independently and writes the block
         * totals into the next level's input. Once a level fits in one block, walk back down
         * adding each block's increment. All buffers are allocated before the timer starts.
         */
        static void blockScan(int n, int *odata, const int *idata, bool efficient) {
            if (n <= 0) {
                return;
            }
            int threads = blockSize;
            int B = efficient ? 2 * threads : threads;
            // Levels shrink by a factor of B, so B = 1 would never reach a single block.
            if (threads < 1 || (threads & (threads - 1)) != 0 || B < 2) {
                throw std::invalid_argument("shared memory scan needs a power-of-two block size (>= 2 for naive)");
            }
            int logBanks = efficient ? logNumBanks : 0;
            size_t sharedBytes = efficient
                ? efficientSharedInts(B, logBanks) * sizeof(int)
                : 2 * B * sizeof(int);

            std::vector<Level> levels;
            for (int length = roundUp(n, B); ; length = roundUp(length / B, B)) {
                Level level = { length, nullptr, nullptr };
                cudaMalloc(&level.dev_in, length * sizeof(int));
                cudaMalloc(&level.dev_out, length * sizeof(int));
                cudaMemset(level.dev_in, 0, length * sizeof(int));
                levels.push_back(level);
                if (length == B) {
                    break;
                }
            }
            // The top level is a single block; its total isn't needed, but the kernel writes it.
            int *dev_topSum = nullptr;
            cudaMalloc(&dev_topSum, sizeof(int));
            checkCUDAError("cudaMalloc block scan levels");

            cudaMemcpy(levels[0].dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("upload idata");

            timer().startGpuTimer();
            int numLevels = static_cast<int>(levels.size());
            for (int k = 0; k < numLevels; k++) {
                int numBlocks = levels[k].length / B;
                int *sums = k + 1 < numLevels ? levels[k + 1].dev_in : dev_topSum;
                if (efficient) {
                    kernEfficientScanBlock<<<numBlocks, threads, sharedBytes>>>(
                        B, logBanks, levels[k].dev_out, levels[k].dev_in, sums);
                } else {
                    kernNaiveScanBlock<<<numBlocks, threads, sharedBytes>>>(
                        B, levels[k].dev_out, levels[k].dev_in, sums);
                }
            }
            int logB = ilog2(B);
            for (int k = numLevels - 2; k >= 0; k--) {
                dim3 grid((levels[k].length + threads - 1) / threads);
                kernAddBlockOffsets<<<grid, threads>>>(levels[k].length, logB, levels[k].dev_out, levels[k + 1].dev_out);
            }
            timer().endGpuTimer();
            checkCUDAError("shared memory block scan");

            cudaMemcpy(odata, levels[0].dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("download result");

            for (Level &level : levels) {
                cudaFree(level.dev_in);
                cudaFree(level.dev_out);
            }
            cudaFree(dev_topSum);
        }

        void naiveScan(int n, int *odata, const int *idata) {
            blockScan(n, odata, idata, false);
        }

        void efficientScan(int n, int *odata, const int *idata) {
            blockScan(n, odata, idata, true);
        }

        int naiveOccupancy(int threadsPerBlock) {
            int blocks = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernNaiveScanBlock,
                threadsPerBlock, 2 * threadsPerBlock * sizeof(int));
            return blocks;
        }

        int efficientOccupancy(int threadsPerBlock, int logBanks) {
            int blocks = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernEfficientScanBlock,
                threadsPerBlock, efficientSharedInts(2 * threadsPerBlock, logBanks) * sizeof(int));
            return blocks;
        }
    }
}
