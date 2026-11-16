#ifndef GPUSEARCH_H
#define GPUSEARCH_H

#include "search.h"

#include <QString>
#include <atomic>
#include <cstdint>
#include <vector>

// Global on/off switch (set from the preferences, default on).
extern bool g_gpuSearchEnabled;

// GPU prefilter for seed searches (OpenCL, loaded at runtime).
//
// The GPU evaluates a necessary condition for the structure filters that sit
// directly below the root of the condition tree. Every seed it lets through is
// still verified by the exact CPU code, so search results are identical to a
// CPU-only search; the GPU only skips seeds that cannot match.
class GpuFilter
{
public:
    GpuFilter();
    ~GpuFilter();

    GpuFilter(const GpuFilter&) = delete;
    GpuFilter& operator=(const GpuFilter&) = delete;

    // Prepares the device and kernel for the given (already applied) condition
    // tree. Returns false when no usable GPU/driver exists or none of the
    // conditions can be prefiltered; 'info' receives a short description either
    // way (device name, or the reason for not using the GPU).
    bool init(const ConditionTree& tree, int mc, QString *info);

    bool ready() const;
    QString deviceName() const;
    int conditionCount() const;

    // Collects the seeds in [start, start+count) that pass the GPU prefilter,
    // in ascending order. The range must not wrap around 2^64.
    // Returns false on a device error (the GPU is then disabled for good and
    // the caller must examine the whole range on the CPU).
    bool filterRange(uint64_t start, uint64_t count,
                     std::vector<uint64_t>& out, const std::atomic_bool *stop);

    // Same for an explicit list of seeds (order of the list is preserved).
    bool filterList(const uint64_t *list, uint64_t count,
                    std::vector<uint64_t>& out, const std::atomic_bool *stop);

    // Number of seeds per GPU launch that currently keeps launches short.
    uint64_t batchHint() const;

private:
    struct Impl;
    Impl *d;
};

#endif // GPUSEARCH_H
