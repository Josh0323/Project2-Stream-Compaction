#include <cstdio>
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

        void scanNoTimer(int n, int *odata, const int * idata) {
            int pre = 0;
            for (int i = 0; i < n; i++) {
                int cur = idata[i];
                odata[i] = pre;
                pre += cur;
            }
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            scanNoTimer(n, odata, idata);
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int j = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] == 0) continue;
                odata[j] = idata[i];
                j++;
            }
            timer().endCpuTimer();
            return j;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            if (n == 0) return 0;

            int *flags = new int[n];
            int *scanResult = new int[n];
            timer().startCpuTimer();
            // TODO
            for (int i = 0; i < n; i++) {
                flags[i] = (int)(idata[i] != 0);
            }

            scanNoTimer(n, scanResult, flags);

            for (int i = 0; i < n; i++) {
                if(flags[i] == 1) odata[scanResult[i]] = idata[i];
            }
            int result = scanResult[n - 1] + flags[n - 1];
            timer().endCpuTimer();
            delete[] flags;
            delete[] scanResult;
            return result;
        }
    }
}
