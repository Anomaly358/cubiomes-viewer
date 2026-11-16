// Verifies the OpenCL prefilter kernel logic (src/gpukernel.cl) on the CPU by
// including it as C++ and comparing against cubiomes' getStructurePos.
//
//   exact kinds (feature/large): kernel result must equal reference result
//   outpost: kernel must never reject a seed the reference accepts
//
// Build/run: see tests/run_gpu_kernel_test.sh

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

extern "C" {
#include "../cubiomes/finders.h"
}

// ---- shim for OpenCL C types/qualifiers --------------------------------
typedef uint64_t ulong;
typedef int64_t  slong;
typedef uint32_t uint;
#define __constant
#define __global
#define __kernel
#include "../src/gpukernel.cl"
static_assert(sizeof(GkCond) == 64, "GkCond layout must match host GpuCond");

static int refCount(int st, int mc, uint64_t seed, const GkCond& c)
{
    int cnt = 0;
    for (int rz = c.rz1; rz <= c.rz2; rz++)
    {
        for (int rx = c.rx1; rx <= c.rx2; rx++)
        {
            Pos p;
            if (!getStructurePos(st, mc, seed, rx, rz, &p))
                continue;
            if (c.useRadius)
            {
                int64_t rsq = p.x * (int64_t)p.x + p.z * (int64_t)p.z;
                if (rsq >= (int64_t)c.rmaxsq)
                    continue;
            }
            else if (p.x < c.x1 || p.x > c.x2 || p.z < c.z1 || p.z > c.z2)
                continue;
            cnt++;
        }
    }
    return cnt;
}

static int floordivi(int a, int b)
{
    int q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0)))
        q--;
    return q;
}

int main()
{
    struct T { int st; int kind; const char *name; };
    const T types[] = {
        { Desert_Pyramid, GK_FEATURE, "desert" },
        { Jungle_Pyramid, GK_FEATURE, "jungle" },
        { Swamp_Hut,      GK_FEATURE, "hut" },
        { Igloo,          GK_FEATURE, "igloo" },
        { Village,        GK_FEATURE, "village" },
        { Ocean_Ruin,     GK_FEATURE, "ruins" },
        { Shipwreck,      GK_FEATURE, "shipwreck" },
        { Ruined_Portal,  GK_FEATURE, "portal" },
        { Ruined_Portal_N,GK_FEATURE, "portal_n" },
        { Ancient_City,   GK_FEATURE, "ancient_city" },
        { Trail_Ruins,    GK_FEATURE, "trails" },
        { Trial_Chambers, GK_FEATURE, "chambers" },
        { Monument,       GK_LARGE,   "monument" },
        { Mansion,        GK_LARGE,   "mansion" },
        { Outpost,        GK_OUTPOST, "outpost" },
    };
    const int mcs[] = {
        MC_1_7, MC_1_9, MC_1_12, MC_1_13, MC_1_15, MC_1_16_1, MC_1_17,
        MC_1_18, MC_1_19, MC_1_20, MC_1_21_1,
    };

    std::mt19937_64 rng(12345);
    long tested = 0, mismatches = 0, passes = 0, outposts_dropped = 0;

    for (const T& t : types)
    {
        for (int mc : mcs)
        {
            StructureConfig sc;
            if (!getStructureConfig(t.st, mc, &sc))
                continue;

            for (int iter = 0; iter < 400; iter++)
            {
                GkCond c;
                memset(&c, 0, sizeof(c));
                c.salt = sc.salt;
                c.regionSize = sc.regionSize;
                c.chunkRange = sc.chunkRange;
                c.kind = t.kind;
                c.count = 1 + (int)(rng() % 3);

                int x1 = (int)(rng() % 8001) - 4000;
                int z1 = (int)(rng() % 8001) - 4000;
                int span = (iter % 2) ? 3000 : 300; // half the cases use small areas (more rejections)
                int w = 16 + (int)(rng() % span);
                int h = 16 + (int)(rng() % span);
                int rs = sc.regionSize << 4;

                if (rng() % 3 == 0)
                {   // radius mode, like cond->rmax
                    int rmax = 32 + (int)(rng() % ((iter % 2) ? 2500 : 300));
                    int r1 = rmax - 1;
                    c.x1 = -r1; c.z1 = -r1; c.x2 = r1; c.z2 = r1;
                    c.useRadius = 1;
                    c.rmaxsq = (uint64_t)((int64_t)r1 * r1 + 1);
                }
                else
                {
                    c.x1 = x1; c.z1 = z1; c.x2 = x1 + w; c.z2 = z1 + h;
                }
                c.rx1 = floordivi(c.x1, rs);
                c.rz1 = floordivi(c.z1, rs);
                c.rx2 = floordivi(c.x2, rs);
                c.rz2 = floordivi(c.z2, rs);

                for (int k = 0; k < 40; k++)
                {
                    uint64_t seed = rng();
                    if (k % 4 == 0) seed &= 0xffffffffffffULL;
                    int ref = refCount(t.st, mc, seed, c) >= c.count;
                    int got = gk_cond_passes(&c, seed);
                    tested++;
                    passes += got;
                    if (t.kind == GK_OUTPOST)
                    {
                        if (ref && !got)
                        {
                            mismatches++;
                            outposts_dropped++;
                        }
                    }
                    else if (ref != got)
                    {
                        if (mismatches < 10)
                            printf("MISMATCH %s mc=%d seed=%llu ref=%d got=%d\n",
                                t.name, mc, (unsigned long long)seed, ref, got);
                        mismatches++;
                    }
                }
            }
        }
    }

    printf("tested=%ld kernel_passes=%ld mismatches=%ld (outpost false negatives=%ld)\n",
        tested, passes, mismatches, outposts_dropped);
    return mismatches ? 1 : 0;
}
