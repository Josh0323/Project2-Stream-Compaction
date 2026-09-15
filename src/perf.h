#pragma once

// Randomized correctness sweep over every implementation. Returns 0 if everything matched.
int runStressTest();

// Timing sweeps that print CSV to stdout. args[0] picks the experiment
// (scan, compact, sort, blocksize, banks, occupancy, all).
int runBenchmarks(int argc, char *argv[]);
