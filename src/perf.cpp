/**
 * @file      perf.cpp
 * @brief     Stress test and benchmark modes for the test program
 *            (--stress and --bench in main.cpp).
 */

#include "perf.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include <stream_compaction/radix.h>
#include <stream_compaction/shared_memory.h>

namespace SC = StreamCompaction;

namespace {

using ArrayFn = void (*)(int, int *, const int *);
using CompactFn = int (*)(int, int *, const int *);
using ElapsedFn = float (*)();

struct ScanImpl { const char *name; ArrayFn run; ElapsedFn elapsed; };
struct CompactImpl { const char *name; CompactFn run; ElapsedFn elapsed; };

float cpuTime() { return SC::CPU::timer().getCpuElapsedTimeForPreviousOperation(); }
float naiveTime() { return SC::Naive::timer().getGpuElapsedTimeForPreviousOperation(); }
float efficientTime() { return SC::Efficient::timer().getGpuElapsedTimeForPreviousOperation(); }
float thrustTime() { return SC::Thrust::timer().getGpuElapsedTimeForPreviousOperation(); }
float radixTime() { return SC::Radix::timer().getGpuElapsedTimeForPreviousOperation(); }
float sharedTime() { return SC::SharedMemory::timer().getGpuElapsedTimeForPreviousOperation(); }

const std::vector<ScanImpl> kScans = {
    {"cpu", SC::CPU::scan, cpuTime},
    {"naive", SC::Naive::scan, naiveTime},
    {"efficient", SC::Efficient::scan, efficientTime},
    {"efficient-unoptimized", SC::Efficient::scanUnoptimized, efficientTime},
    {"thrust", SC::Thrust::scan, thrustTime},
    {"shared-naive", SC::SharedMemory::naiveScan, sharedTime},
    {"shared-efficient", SC::SharedMemory::efficientScan, sharedTime},
};

const std::vector<CompactImpl> kCompacts = {
    {"cpu-without-scan", SC::CPU::compactWithoutScan, cpuTime},
    {"cpu-with-scan", SC::CPU::compactWithScan, cpuTime},
    {"efficient", SC::Efficient::compact, efficientTime},
    {"thrust", SC::Thrust::compact, thrustTime},
};

const std::vector<ScanImpl> kSorts = {
    {"cpu-std-sort", SC::CPU::sort, cpuTime},
    {"radix", SC::Radix::sort, radixTime},
    {"thrust-sort", SC::Thrust::sort, thrustTime},
};

std::mt19937 &rng() {
    static std::mt19937 gen(565);
    return gen;
}

void fillRandom(std::vector<int> &v, int n, int maxval) {
    for (int i = 0; i < n; i++) {
        v[i] = static_cast<int>(rng()() % maxval);
    }
}

void fillSigned(std::vector<int> &v, int n) {
    for (int i = 0; i < n; i++) {
        v[i] = static_cast<int>(rng()());
    }
}

// ---------------------------------------------------------------------------
// Stress test
// ---------------------------------------------------------------------------

struct Tally {
    int checks = 0;
    int failures = 0;
};

bool expectEqual(Tally &t, const char *what, int n, const int *expected, const int *got, int count) {
    t.checks++;
    for (int i = 0; i < count; i++) {
        if (expected[i] != got[i]) {
            t.failures++;
            printf("  FAIL %s n=%d: index %d expected %d, got %d\n", what, n, i, expected[i], got[i]);
            return false;
        }
    }
    return true;
}

bool expectCount(Tally &t, const char *what, int n, int expected, int got) {
    if (expected != got) {
        t.checks++;
        t.failures++;
        printf("  FAIL %s n=%d: expected %d elements, got %d\n", what, n, expected, got);
        return false;
    }
    return true;
}

std::vector<int> stressSizes() {
    std::vector<int> sizes = {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
                              127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025,
                              2047, 2048, 2049, 4095, 4096, 4097, 65535, 65536, 65537};
    std::uniform_int_distribution<int> small(1, 5000);
    for (int i = 0; i < 20; i++) {
        sizes.push_back(small(rng()));
    }
    for (int big : {(1 << 20) - 3, (1 << 20) + 1, (1 << 22) + 7}) {
        sizes.push_back(big);
    }
    return sizes;
}

}  // namespace

int runStressTest() {
    Tally tally;
    std::vector<int> sizes = stressSizes();
    int maxN = *std::max_element(sizes.begin(), sizes.end());
    std::vector<int> in(maxN), expected(maxN), got(maxN);
    char label[96];

    const int naiveBlock = SC::Naive::blockSize;
    const int efficientBlock = SC::Efficient::blockSize;
    const int sharedBlock = SC::SharedMemory::blockSize;
    const int sharedBanks = SC::SharedMemory::logNumBanks;

    printf("Stress testing %zu sizes (largest %d)...\n", sizes.size(), maxN);
    for (int n : sizes) {
        bool small = n <= 70000;

        // Scan. The CPU version is itself checked against std::partial_sum.
        fillRandom(in, n, 50);
        expected[0] = 0;
        std::partial_sum(in.begin(), in.begin() + n - 1, expected.begin() + 1);
        for (const ScanImpl &impl : kScans) {
            std::fill(got.begin(), got.begin() + n, -1);
            impl.run(n, got.data(), in.data());
            snprintf(label, sizeof label, "scan/%s", impl.name);
            expectEqual(tally, label, n, expected.data(), got.data(), n);
        }

        // Scan implementations under other launch configurations.
        if (small) {
            for (int bs : {1, 33, 1024}) {
                SC::Naive::blockSize = bs;
                SC::Efficient::blockSize = bs;
                for (ArrayFn fn : {SC::Naive::scan, SC::Efficient::scan, SC::Efficient::scanUnoptimized}) {
                    std::fill(got.begin(), got.begin() + n, -1);
                    fn(n, got.data(), in.data());
                    snprintf(label, sizeof label, "scan/global-memory(blockSize=%d)", bs);
                    expectEqual(tally, label, n, expected.data(), got.data(), n);
                }
            }
            SC::Naive::blockSize = naiveBlock;
            SC::Efficient::blockSize = efficientBlock;

            for (int bs : {1, 2, 4, 16, 64, 256, 1024}) {
                SC::SharedMemory::blockSize = bs;
                if (bs > 1) {
                    std::fill(got.begin(), got.begin() + n, -1);
                    SC::SharedMemory::naiveScan(n, got.data(), in.data());
                    snprintf(label, sizeof label, "scan/shared-naive(blockSize=%d)", bs);
                    expectEqual(tally, label, n, expected.data(), got.data(), n);
                }
                for (int logBanks : {0, 2, 4, 6}) {
                    SC::SharedMemory::logNumBanks = logBanks;
                    std::fill(got.begin(), got.begin() + n, -1);
                    SC::SharedMemory::efficientScan(n, got.data(), in.data());
                    snprintf(label, sizeof label, "scan/shared-efficient(blockSize=%d, logBanks=%d)", bs, logBanks);
                    expectEqual(tally, label, n, expected.data(), got.data(), n);
                }
            }
            SC::SharedMemory::blockSize = sharedBlock;
            SC::SharedMemory::logNumBanks = sharedBanks;
        }

        // Compaction, against std::copy_if. Alternate how dense the zeros are.
        fillRandom(in, n, (n % 3 == 0) ? 2 : 4);
        if (n % 5 == 0) {
            std::fill(in.begin(), in.begin() + n, 0);
        }
        int expectedCount = static_cast<int>(
            std::copy_if(in.begin(), in.begin() + n, expected.begin(), [](int x) { return x != 0; })
            - expected.begin());
        for (const CompactImpl &impl : kCompacts) {
            std::fill(got.begin(), got.begin() + n, -1);
            int count = impl.run(n, got.data(), in.data());
            snprintf(label, sizeof label, "compact/%s", impl.name);
            if (expectCount(tally, label, n, expectedCount, count)) {
                expectEqual(tally, label, n, expected.data(), got.data(), count);
            }
        }

        // Sorting, against std::sort, over a few value distributions.
        for (int dist = 0; dist < 4; dist++) {
            switch (dist) {
                case 0: fillRandom(in, n, 1000); break;
                case 1: fillSigned(in, n); break;
                case 2: std::fill(in.begin(), in.begin() + n, -42); break;
                default:
                    fillRandom(in, n, 2001);
                    for (int i = 0; i < n; i++) in[i] -= 1000;
                    break;
            }
            if (dist == 1 && n >= 2) {
                in[0] = INT_MAX;
                in[n - 1] = INT_MIN;
            }
            std::copy(in.begin(), in.begin() + n, expected.begin());
            std::sort(expected.begin(), expected.begin() + n);
            for (const ScanImpl &impl : kSorts) {
                std::fill(got.begin(), got.begin() + n, -1);
                impl.run(n, got.data(), in.data());
                snprintf(label, sizeof label, "sort/%s(dist=%d)", impl.name, dist);
                expectEqual(tally, label, n, expected.data(), got.data(), n);
            }
        }
    }

    printf("%d checks, %d failures: %s\n", tally.checks, tally.failures,
           tally.failures ? "FAILED" : "all passed");
    return tally.failures ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

namespace {

const int kTrials = 7;

struct Stats { double median; double min; };

// Runs one warm-up call, then kTrials timed calls, and reports the implementation's own timer.
Stats measure(const std::function<void()> &call, ElapsedFn elapsed) {
    call();
    std::vector<double> times;
    for (int i = 0; i < kTrials; i++) {
        call();
        times.push_back(elapsed());
    }
    std::sort(times.begin(), times.end());
    return {times[times.size() / 2], times.front()};
}

void row(const char *experiment, const std::string &impl, int n, int blockSize, int param, Stats s) {
    printf("%s,%s,%d,%d,%d,%.6f,%.6f\n", experiment, impl.c_str(), n, blockSize, param, s.median, s.min);
    fflush(stdout);
}

int blockSizeOf(const std::string &impl) {
    if (impl.rfind("naive", 0) == 0) return SC::Naive::blockSize;
    if (impl.rfind("efficient", 0) == 0) return SC::Efficient::blockSize;
    if (impl.rfind("shared", 0) == 0) return SC::SharedMemory::blockSize;
    if (impl == "radix") return SC::Radix::blockSize;
    return 0;
}

std::vector<int> benchSizes(int minLog, int maxLog) {
    std::vector<int> sizes;
    for (int lg = minLog; lg <= maxLog; lg++) {
        sizes.push_back(1 << lg);
        sizes.push_back((1 << lg) - 3);
    }
    return sizes;
}

void warmUp() {
    std::vector<int> in(4096), out(4096);
    fillRandom(in, 4096, 4);
    for (const ScanImpl &impl : kScans) impl.run(4096, out.data(), in.data());
    for (const CompactImpl &impl : kCompacts) impl.run(4096, out.data(), in.data());
    for (const ScanImpl &impl : kSorts) impl.run(4096, out.data(), in.data());
}

void benchScan(int maxLog) {
    std::vector<int> sizes = benchSizes(8, maxLog);
    std::vector<int> in(1 << maxLog), out(1 << maxLog);
    for (int n : sizes) {
        fillRandom(in, n, 50);
        for (const ScanImpl &impl : kScans) {
            Stats s = measure([&] { impl.run(n, out.data(), in.data()); }, impl.elapsed);
            row("scan", impl.name, n, blockSizeOf(impl.name), 0, s);
        }
    }
}

void benchCompact(int maxLog) {
    std::vector<int> sizes = benchSizes(8, maxLog);
    std::vector<int> in(1 << maxLog), out(1 << maxLog);
    for (int n : sizes) {
        fillRandom(in, n, 4);
        for (const CompactImpl &impl : kCompacts) {
            Stats s = measure([&] { impl.run(n, out.data(), in.data()); }, impl.elapsed);
            row("compact", impl.name, n, blockSizeOf(impl.name), 0, s);
        }
    }
}

void benchSort(int maxLog) {
    std::vector<int> sizes = benchSizes(8, maxLog);
    std::vector<int> in(1 << maxLog), out(1 << maxLog);
    for (int n : sizes) {
        // param 0: values in [0, 1000) (10 radix passes), param 1: any int (32 passes).
        for (int dist = 0; dist < 2; dist++) {
            if (dist == 0) fillRandom(in, n, 1000); else fillSigned(in, n);
            for (const ScanImpl &impl : kSorts) {
                Stats s = measure([&] { impl.run(n, out.data(), in.data()); }, impl.elapsed);
                row("sort", impl.name, n, blockSizeOf(impl.name), dist, s);
            }
        }
    }
}

// Sizes on a linear grid rather than powers of two, to show the padding stair-step:
// implementations that round n up to the next power of two jump at each boundary.
void benchNonPowerOfTwo() {
    const int maxN = 10000000;
    std::vector<int> in(maxN), out(maxN);
    fillRandom(in, maxN, 50);
    const std::vector<std::string> want = {"cpu", "efficient", "efficient-unoptimized", "thrust", "shared-efficient"};
    for (int n = 500000; n <= maxN; n += 250000) {
        for (const ScanImpl &impl : kScans) {
            if (std::find(want.begin(), want.end(), impl.name) == want.end()) continue;
            Stats s = measure([&] { impl.run(n, out.data(), in.data()); }, impl.elapsed);
            row("npot", impl.name, n, blockSizeOf(impl.name), 0, s);
        }
    }
}

void benchBlockSize(int log2n) {
    int n = 1 << log2n;
    std::vector<int> in(n), out(n);
    fillRandom(in, n, 50);
    std::vector<int> compactIn(n);
    fillRandom(compactIn, n, 4);

    for (int bs = 16; bs <= 1024; bs *= 2) {
        SC::Naive::blockSize = SC::Efficient::blockSize = SC::SharedMemory::blockSize = SC::Radix::blockSize = bs;
        for (const ScanImpl &impl : kScans) {
            if (blockSizeOf(impl.name) == 0) continue;
            Stats s = measure([&] { impl.run(n, out.data(), in.data()); }, impl.elapsed);
            row("blocksize", impl.name, n, bs, 0, s);
        }
        Stats s = measure([&] { SC::Efficient::compact(n, out.data(), compactIn.data()); }, efficientTime);
        row("blocksize", "efficient-compact", n, bs, 0, s);
    }
}

void benchBanks(int log2n) {
    int n = 1 << log2n;
    std::vector<int> in(n), out(n);
    fillRandom(in, n, 50);
    int savedBlock = SC::SharedMemory::blockSize;
    int savedBanks = SC::SharedMemory::logNumBanks;
    for (int bs = 64; bs <= 1024; bs *= 2) {
        SC::SharedMemory::blockSize = bs;
        for (int logBanks = 0; logBanks <= 7; logBanks++) {
            SC::SharedMemory::logNumBanks = logBanks;
            Stats s = measure([&] { SC::SharedMemory::efficientScan(n, out.data(), in.data()); }, sharedTime);
            row("banks", "shared-efficient", n, bs, logBanks, s);
        }
    }
    SC::SharedMemory::blockSize = savedBlock;
    SC::SharedMemory::logNumBanks = savedBanks;
}

void printOccupancy() {
    printf("occupancy,threadsPerBlock,shared-naive,shared-efficient(logBanks=0),shared-efficient(logBanks=4)\n");
    for (int bs = 2; bs <= 1024; bs *= 2) {
        printf("occupancy,%d,%d,%d,%d\n", bs, SC::SharedMemory::naiveOccupancy(bs),
               SC::SharedMemory::efficientOccupancy(bs, 0), SC::SharedMemory::efficientOccupancy(bs, 4));
    }
}

}  // namespace

int runBenchmarks(int argc, char *argv[]) {
    std::string which = argc > 0 ? argv[0] : "all";
    int maxLog = argc > 1 ? std::atoi(argv[1]) : 26;

    warmUp();
    printf("experiment,impl,n,blockSize,param,median_ms,min_ms\n");
    if (which == "scan" || which == "all") benchScan(maxLog);
    if (which == "compact" || which == "all") benchCompact(maxLog);
    if (which == "sort" || which == "all") benchSort(std::min(maxLog, 24));
    if (which == "npot" || which == "all") benchNonPowerOfTwo();
    if (which == "blocksize" || which == "all") benchBlockSize(argc > 1 ? maxLog : 24);
    if (which == "banks" || which == "all") benchBanks(argc > 1 ? maxLog : 24);
    if (which == "occupancy" || which == "all") printOccupancy();
    return 0;
}
