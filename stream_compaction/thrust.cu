#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            if (n <= 0) {
                return;
            }
            // Constructing device_vectors from the host range does the upload up front,
            // so the timer only sees the scan itself.
            thrust::device_vector<int> dv_in(idata, idata + n);
            thrust::device_vector<int> dv_out(n);

            timer().startGpuTimer();
            thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            timer().endGpuTimer();

            thrust::copy(dv_out.begin(), dv_out.end(), odata);
        }

        int compact(int n, int *odata, const int *idata) {
            if (n <= 0) {
                return 0;
            }
            thrust::device_vector<int> dv_data(idata, idata + n);

            timer().startGpuTimer();
            // logical_not is true for 0, so this drops the zeros in place. remove_if is stable.
            auto newEnd = thrust::remove_if(dv_data.begin(), dv_data.end(), thrust::logical_not<int>());
            int count = static_cast<int>(newEnd - dv_data.begin());
            timer().endGpuTimer();

            thrust::copy(dv_data.begin(), newEnd, odata);
            return count;
        }
    }
}
