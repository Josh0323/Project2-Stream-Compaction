#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Radix {
        StreamCompaction::Common::PerformanceTimer& timer();

        // Threads per block for kernel launches in this module.
        extern int blockSize;

        /**
         * Stable GPU LSD radix sort built on the work-efficient scan (GPU Gems 3, 39.3.3).
         * Handles the full int range, negatives included.
         */
        void sort(int n, int *odata, const int *idata);
    }
}
