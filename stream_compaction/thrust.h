#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Thrust {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        // Not required: thrust::remove_if compaction, for comparing against Efficient::compact.
        int compact(int n, int *odata, const int *idata);
    }
}
