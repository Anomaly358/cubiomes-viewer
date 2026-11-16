// Exercises GpuFilter (src/gpusearch.cpp) end to end against a mock OpenCL
// driver and compares the survivors with a brute-force cubiomes reference.
#include "search.h"
#include "gpusearch.h"
#include <algorithm>
#include <cstdio>
#include <random>
#include <set>

extern "C" int getStructureConfig_override(int stype, int mc, StructureConfig *sconf)
{ return getStructureConfig(stype, mc, sconf); }

ConditionTree::~ConditionTree() {}

static Condition mk(int type, int x1, int z1, int x2, int z2, int count, int relative, int save)
{
    Condition c; memset(&c, 0, sizeof(c));
    c.type = type; c.x1 = x1; c.z1 = z1; c.x2 = x2; c.z2 = z2;
    c.count = count; c.relative = relative; c.save = save;
    return c;
}

static int floordivi(int a, int b) { int q = a / b; if ((a % b) && ((a < 0) != (b < 0))) q--; return q; }

// reference for one root condition (same semantics as the CPU search before biome checks)
static bool refCond(const Condition& c, int mc, uint64_t seed)
{
    const FilterInfo& fi = g_filterinfo.list[c.type];
    StructureConfig sc; getStructureConfig(fi.stype, mc, &sc);
    int rs = sc.regionSize << 4, cnt = 0;
    for (int rz = floordivi(c.z1, rs); rz <= floordivi(c.z2, rs); rz++)
        for (int rx = floordivi(c.x1, rs); rx <= floordivi(c.x2, rs); rx++)
        {
            Pos p;
            if (!getStructurePos(fi.stype, mc, seed, rx, rz, &p)) continue;
            if (p.x < c.x1 || p.x > c.x2 || p.z < c.z1 || p.z > c.z2) continue;
            cnt++;
        }
    return cnt >= c.count;
}

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL: " __VA_ARGS__); printf("\n"); fails++; } } while (0)

int main()
{
    const int mc = MC_1_18;
    ConditionTree tree;
    tree.condvec.resize(8);
    tree.references.resize(8);
    tree.condvec[0] = mk(F_SELECT, 0,0,0,0, 0, 0, 0);
    tree.condvec[1] = mk(F_VILLAGE,  -350,-350, 350, 350, 3, 0, 1);   // used
    tree.condvec[2] = mk(F_MONUMENT, -450,-450, 450, 450, 1, 0, 2);   // used
    tree.condvec[3] = mk(F_OUTPOST,   -500, -500,  500,  500, 2, 1, 3);   // child of 1: ignored
    tree.condvec[4] = mk(F_HUT,       -500, -500,  500,  500, 0, 0, 4);   // exclusion: ignored
    tree.condvec[5] = mk(F_BIOME,     -500, -500,  500,  500, 1, 0, 5);   // biome filter: ignored
    tree.references[0] = {1, 2, 4, 5};
    tree.references[1] = {3};

    GpuFilter gf;
    QString info;
    CHECK(gf.init(tree, mc, &info), "init failed: %s", info.c_str());
    printf("device: %s, conditions on GPU: %d\n", gf.deviceName().c_str(), gf.conditionCount());
    CHECK(gf.conditionCount() == 2, "expected 2 GPU conditions, got %d", gf.conditionCount());

    // 1) consecutive range vs brute force
    {
        uint64_t base = 0xfedcba9876543210ULL, n = 400000;
        std::vector<uint64_t> got;
        CHECK(gf.filterRange(base, n, got, nullptr), "filterRange failed");
        CHECK(std::is_sorted(got.begin(), got.end()), "range result not sorted");
        std::set<uint64_t> exp;
        for (uint64_t i = 0; i < n; i++)
        {
            uint64_t s = base + i;
            if (refCond(tree.condvec[1], mc, s) && refCond(tree.condvec[2], mc, s)) exp.insert(s);
        }
        std::set<uint64_t> gs(got.begin(), got.end());
        CHECK(gs == exp, "range survivors differ: got %zu expected %zu", gs.size(), exp.size());
        printf("range: %zu/%llu seeds survive (exact match with reference)\n", exp.size(), (unsigned long long)n);
        CHECK(!exp.empty() && exp.size() < n / 2, "test is not selective enough");
    }

    // 2) low 48-bit range crossing 0 boundary of the counter (start at 0)
    {
        std::vector<uint64_t> got;
        CHECK(gf.filterRange(0, 100000, got, nullptr), "filterRange(0) failed");
        std::set<uint64_t> exp;
        for (uint64_t s = 0; s < 100000; s++)
            if (refCond(tree.condvec[1], mc, s) && refCond(tree.condvec[2], mc, s)) exp.insert(s);
        CHECK(std::set<uint64_t>(got.begin(), got.end()) == exp, "range-from-0 differs");
    }

    // 3) explicit list (shuffled)
    {
        std::mt19937_64 rng(7);
        std::vector<uint64_t> list(150000);
        for (auto& s : list) s = rng();
        std::vector<uint64_t> got;
        CHECK(gf.filterList(list.data(), list.size(), got, nullptr), "filterList failed");
        std::set<uint64_t> exp;
        for (uint64_t s : list)
            if (refCond(tree.condvec[1], mc, s) && refCond(tree.condvec[2], mc, s)) exp.insert(s);
        CHECK(std::set<uint64_t>(got.begin(), got.end()) == exp, "list survivors differ");
        printf("list: %zu/%zu survive\n", exp.size(), list.size());
    }

    // 4) survivor-buffer overflow: nearly everything passes (> 2^20 survivors per launch)
    {
        ConditionTree t2;
        t2.condvec.resize(2); t2.references.resize(2);
        t2.condvec[0] = mk(F_SELECT, 0,0,0,0, 0, 0, 0);
        t2.condvec[1] = mk(F_VILLAGE, -6000,-6000, 6000, 6000, 1, 0, 1);
        t2.references[0] = {1};
        GpuFilter g2; QString i2;
        CHECK(g2.init(t2, mc, &i2), "init2 failed: %s", i2.c_str());
        uint64_t n = (1u << 21) + 12345, base = 5000000000ULL;
        std::vector<uint64_t> got;
        CHECK(g2.filterRange(base, n, got, nullptr), "overflow range failed");
        std::set<uint64_t> gs(got.begin(), got.end());
        CHECK(gs.size() == got.size(), "duplicates after overflow split");
        size_t exp = 0, miss = 0;
        for (uint64_t i = 0; i < n; i++)
        {
            bool r = refCond(t2.condvec[1], mc, base + i);
            exp += r;
            if (r && !gs.count(base + i)) miss++;
        }
        CHECK(gs.size() == exp && miss == 0, "overflow case: got %zu expected %zu missing %zu", gs.size(), exp, miss);
        printf("overflow split: %zu survivors, exact\n", exp);
    }

    // 5) tree without usable conditions
    {
        ConditionTree t3; t3.condvec.resize(2); t3.references.resize(2);
        t3.condvec[0] = mk(F_SELECT, 0,0,0,0, 0, 0, 0);
        t3.condvec[1] = mk(F_BIOME, 0,0,10,10, 1, 0, 1);
        t3.references[0] = {1};
        GpuFilter g3; QString i3;
        CHECK(!g3.init(t3, mc, &i3), "init should refuse a tree with nothing to prefilter");
        printf("no-op tree: \"%s\"\n", i3.c_str());
    }

    printf(fails ? "RESULT: %d failure(s)\n" : "RESULT: all host tests passed\n", fails);
    return fails ? 1 : 0;
}
