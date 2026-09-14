#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Naive {
        StreamCompaction::Common::PerformanceTimer& timer();

        // Threads per block for kernel launches in this module. Tunable so the
        // benchmark can sweep it without recompiling.
        extern int blockSize;

        void scan(int n, int *odata, const int *idata);
    }
}
