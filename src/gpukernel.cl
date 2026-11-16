// OpenCL prefilter for Cubiomes Viewer structure conditions.
//
// For each candidate seed this kernel evaluates a conservative, necessary
// condition: "at least `count` structure positions (as computed by cubiomes
// getStructurePos, i.e. BEFORE any biome viability test) lie inside the
// condition's search area". Seeds that fail cannot satisfy the full CPU
// condition either, so they can be discarded. Seeds that pass are re-checked
// by the exact CPU code, so results are identical to a CPU-only search.
//
// This file is valid OpenCL C 1.2 and is also included as plain C++ by the
// verification harness (tests/gpu_kernel_test.cpp), see the shim there.

#if defined(GK_OPENCL) || defined(__OPENCL_VERSION__) || defined(__OPENCL_C_VERSION__)
#ifndef GK_OPENCL
#define GK_OPENCL 1
#endif
typedef long slong;     // signed 64-bit
#else
// host-side shim: the verification harness defines ulong, uint, slong
// and the __constant/__global qualifiers before including this file.
#endif

#define GK_FEATURE  0   // getFeaturePos
#define GK_LARGE    1   // getLargeStructurePos (monument, mansion)
#define GK_OUTPOST  2   // feature position + 1/5 attempt roll

#define GK_MAX_CONDS 16

typedef struct
{
    int salt;           // structure salt (int32, sign-extended like cubiomes)
    int regionSize;     // in chunks
    int chunkRange;     // in chunks
    int kind;           // GK_*
    int x1, z1, x2, z2; // search rectangle in blocks (inclusive)
    int rx1, rz1, rx2, rz2; // region range (inclusive)
    int count;          // required number of positions (>= 1)
    int useRadius;      // 0: rectangle test, 1: circle test
    ulong rmaxsq;       // squared radius (strict), only if useRadius
} GkCond;

// ---------------------------------------------------------------------------


// Position of the structure candidate in region (rx,rz) in blocks.
// Mirrors getFeaturePos / getLargeStructurePos from cubiomes exactly.
static inline void gk_region_pos(const GkCond *c, ulong seed, int rx, int rz,
                                 int *px, int *pz)
{
    const ulong K = 0x5deece66dUL;
    const ulong M = 0xffffffffffffUL;
    const ulong b = 0xbUL;
    ulong r = (ulong) c->chunkRange;
    int cx, cz;

    seed = seed + (ulong)rx * 341873128712UL + (ulong)rz * 132897987541UL
                + (ulong)c->salt;
    seed = seed ^ K;

    if (c->kind == GK_LARGE)
    {
        seed = (seed * K + b) & M;
        cx = (int)((seed >> 17) % r);
        seed = (seed * K + b) & M;
        cx += (int)((seed >> 17) % r);

        seed = (seed * K + b) & M;
        cz = (int)((seed >> 17) % r);
        seed = (seed * K + b) & M;
        cz += (int)((seed >> 17) % r);

        cx >>= 1;
        cz >>= 1;
    }
    else
    {
        seed = (seed * K + b) & M;
        if (r & (r - 1))
        {
            cx = (int)((seed >> 17) % r);
            seed = (seed * K + b) & M;
            cz = (int)((seed >> 17) % r);
        }
        else
        {   // Java RNG treats powers of two specially
            cx = (int)((r * (seed >> 17)) >> 31);
            seed = (seed * K + b) & M;
            cz = (int)((r * (seed >> 17)) >> 31);
        }
    }

    *px = (int)(((ulong)rx * (ulong)c->regionSize + (ulong)cx) << 4);
    *pz = (int)(((ulong)rz * (ulong)c->regionSize + (ulong)cz) << 4);
}

// Outpost: cubiomes rolls nextInt(5)==0 with the attempt seed of the position.
static inline int gk_outpost_ok(ulong seed, int px, int pz)
{
    const ulong M = 0xffffffffffffUL;
    ulong s = seed;
    s ^= (ulong)(px >> 4 >> 4) ^ ((ulong)(pz >> 4 >> 4) << 4);
    s = (s ^ 0x5deece66dUL) & M;                       // setSeed
    s = (s * 0x5deece66dUL + 0xbUL) & M;               // next(31) (discarded)
    s = (s * 0x5deece66dUL + 0xbUL) & M;               // next(31) for nextInt(5)
    // nextInt(5): n is not a power of two -> rejection loop on bits % 5.
    // The first draw decides ~always; to stay conservative we accept the
    // candidate if the first draw is accepted as 0 OR rejected (retry).
    int bits = (int)((slong)s >> 17);
    int val = bits % 5;
    if ((int)((uint)bits - (uint)val + 4u) < 0)
        return 1;   // would re-roll: be conservative, keep the candidate
    return val == 0;
}

static inline int gk_cond_passes(const GkCond *c, ulong seed)
{
    int found = 0;
    for (int rz = c->rz1; rz <= c->rz2; rz++)
    {
        for (int rx = c->rx1; rx <= c->rx2; rx++)
        {
            int px, pz;
            gk_region_pos(c, seed, rx, rz, &px, &pz);

            if (c->useRadius)
            {
                slong dx = px, dz = pz;
                ulong rsq = (ulong)(dx * dx + dz * dz);
                if (rsq >= c->rmaxsq)
                    continue;
            }
            else if (px < c->x1 || px > c->x2 || pz < c->z1 || pz > c->z2)
                continue;

            if (c->kind == GK_OUTPOST && !gk_outpost_ok(seed, px, pz))
                continue;

            if (++found >= c->count)
                return 1;
        }
    }
    return 0;
}

static inline int gk_seed_passes(const GkCond *conds, int n, ulong seed)
{
    for (int i = 0; i < n; i++)
        if (!gk_cond_passes(&conds[i], seed))
            return 0;
    return 1;
}

#ifdef GK_OPENCL

__kernel void gk_filter_range(
    __constant GkCond *conds, int nconds,
    ulong start, ulong total,
    __global ulong *out, __global volatile uint *outcount, uint outcap)
{
    ulong gid = get_global_id(0);
    if (gid >= total)
        return;
    GkCond lc[GK_MAX_CONDS];
    for (int i = 0; i < nconds; i++)
        lc[i] = conds[i];
    ulong seed = start + gid;
    if (gk_seed_passes(lc, nconds, seed))
    {
        uint idx = atomic_inc(outcount);
        if (idx < outcap)
            out[idx] = seed;
    }
}

__kernel void gk_filter_list(
    __constant GkCond *conds, int nconds,
    __global const ulong *list, ulong total,
    __global ulong *out, __global volatile uint *outcount, uint outcap)
{
    ulong gid = get_global_id(0);
    if (gid >= total)
        return;
    GkCond lc[GK_MAX_CONDS];
    for (int i = 0; i < nconds; i++)
        lc[i] = conds[i];
    ulong seed = list[gid];
    if (gk_seed_passes(lc, nconds, seed))
    {
        uint idx = atomic_inc(outcount);
        if (idx < outcap)
            out[idx] = seed;
    }
}

#endif
