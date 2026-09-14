#include <cstdio>
#include <vector>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * Untimed exclusive prefix sum. compactWithScan needs a scan too, but it can't
         * call scan() because starting the CPU timer while it's running throws.
         * Reading idata[i] before writing odata[i] keeps this safe when odata == idata.
         */
        static void prefixSum(int n, int *odata, const int *idata) {
            int sum = 0;
            for (int i = 0; i < n; i++) {
                int value = idata[i];
                odata[i] = sum;
                sum += value;
            }
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            prefixSum(n, odata, idata);
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[count++] = idata[i];
                }
            }
            timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            // Allocate outside the timed region, same rule as cudaMalloc for the GPU versions.
            std::vector<int> bools(n > 0 ? n : 0);
            std::vector<int> indices(n > 0 ? n : 0);

            timer().startCpuTimer();
            // Map: 1 for elements we keep, 0 for elements we drop.
            for (int i = 0; i < n; i++) {
                bools[i] = idata[i] != 0 ? 1 : 0;
            }

            // Scan: indices[i] is where idata[i] lands in the output, if kept.
            prefixSum(n, indices.data(), bools.data());

            // Scatter.
            for (int i = 0; i < n; i++) {
                if (bools[i]) {
                    odata[indices[i]] = idata[i];
                }
            }
            int count = n > 0 ? indices[n - 1] + bools[n - 1] : 0;
            timer().endCpuTimer();
            return count;
        }
    }
}
