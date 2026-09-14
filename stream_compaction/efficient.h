#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        // Threads per block for kernel launches in this module. Tunable so the
        // benchmark can sweep it without recompiling.
        extern int blockSize;

        /**
         * Untimed, in-place work-efficient scan of a device buffer. n must be a power
         * of two, with zeros past the real data. compact() and the radix sort use this
         * so they don't start the timer twice or round-trip through the host.
         */
        void scanDevice(int n, int *dev_data);

        void scan(int n, int *odata, const int *idata);

        /**
         * Part 5 baseline: the same up/down sweep written straight from the slides.
         * Every level launches one thread per element and most of them only fail the
         * "k % stride == 0" test. Kept around for the performance comparison.
         */
        void scanUnoptimized(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);
    }
}
