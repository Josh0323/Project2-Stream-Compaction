#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace SharedMemory {
        StreamCompaction::Common::PerformanceTimer& timer();

        // Threads per block. Must be a power of two, at most the device's max threads per block.
        extern int blockSize;

        // log2 of the bank count used for the conflict-free shared memory layout in
        // efficientScan (GPU Gems 3, 39.2.3). 0 uses the original Example 39-2 layout.
        extern int logNumBanks;

        /**
         * GPU Gems 3 Example 39-1: naive scan with each block's ping-pong buffers in
         * shared memory. Blocks are joined with the block-sum method from 39.2.4, so
         * any n works.
         */
        void naiveScan(int n, int *odata, const int *idata);

        /**
         * GPU Gems 3 Example 39-2: work-efficient scan in shared memory, two elements
         * per thread, joined across blocks with 39.2.4.
         */
        void efficientScan(int n, int *odata, const int *idata);

        // Max active blocks per streaming multiprocessor for each kernel at a block size.
        int naiveOccupancy(int threadsPerBlock);
        int efficientOccupancy(int threadsPerBlock, int logBanks);
    }
}
