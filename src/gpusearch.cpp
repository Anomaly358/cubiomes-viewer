#include "gpusearch.h"
#include "cl_min.h"

#include <QElapsedTimer>
#include <QFile>
#include <QLibrary>
#include <QMutex>
#include <QMutexLocker>

#include <algorithm>
#include <cstring>

bool g_gpuSearchEnabled = true;

// ---------------------------------------------------------------------------
// Mirror of GkCond in gpukernel.cl (same field order, natural alignment).
// ---------------------------------------------------------------------------
namespace {

enum { GK_FEATURE = 0, GK_LARGE = 1, GK_OUTPOST = 2 };
const int GK_MAX_CONDS = 16;

struct GpuCond
{
    cl_int salt;
    cl_int regionSize;
    cl_int chunkRange;
    cl_int kind;
    cl_int x1, z1, x2, z2;
    cl_int rx1, rz1, rx2, rz2;
    cl_int count;
    cl_int useRadius;
    cl_ulong rmaxsq;
};
static_assert(sizeof(GpuCond) == 64, "GpuCond must match GkCond in gpukernel.cl");

// Structure types the kernel reproduces exactly (see tests/gpu_kernel_test.cpp).
int kindOf(int stype)
{
    switch (stype)
    {
    case Desert_Pyramid:
    case Jungle_Pyramid:
    case Swamp_Hut:
    case Igloo:
    case Village:
    case Ocean_Ruin:
    case Shipwreck:
    case Ruined_Portal:
    case Ruined_Portal_N:
    case Ancient_City:
    case Trail_Ruins:
    case Trial_Chambers:
        return GK_FEATURE;
    case Monument:
    case Mansion:
        return GK_LARGE;
    case Outpost:
        return GK_OUTPOST;
    default:
        return -1;
    }
}

int64_t floordiv64(int64_t a, int64_t b)
{
    int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0)))
        q--;
    return q;
}

// Translates conditions directly below the root into GPU conditions.
// Only conditions every matching seed must satisfy are used (root children
// are combined with AND), and only the part that does not need biome
// information (position inside the area, minimum count).
std::vector<GpuCond> buildConds(const ConditionTree& tree, int mc)
{
    std::vector<GpuCond> conds;
    if (tree.references.empty())
        return conds;

    for (char idx : tree.references[0])
    {
        if (idx <= 0 || (size_t)idx >= tree.condvec.size())
            continue;
        const Condition& c = tree.condvec[(size_t)idx];
        if (c.type <= 0 || c.type >= FILTER_MAX)
            continue;
        const FilterInfo& fi = g_filterinfo.list[c.type];
        if (fi.cat != CAT_STRUCT)
            continue;
        int kind = kindOf(fi.stype);
        if (kind < 0)
            continue;
        if (c.count < 1)
            continue; // exclusion filters cannot be prefiltered conservatively
        if (mc < fi.mcmin || mc > fi.mcmax)
            continue;

        StructureConfig sc;
        if (!getStructureConfig_override(fi.stype, mc, &sc))
            continue;
        if (sc.regionSize < 1 || sc.chunkRange < 1)
            continue;

        GpuCond g;
        memset(&g, 0, sizeof(g));
        g.salt = sc.salt;
        g.regionSize = sc.regionSize;
        g.chunkRange = sc.chunkRange;
        g.kind = kind;
        g.count = c.count;

        // identical area derivation as testCondAt() with origin (0,0)
        if (c.rmax > 0)
        {
            int64_t r = (int64_t)c.rmax - 1;
            g.x1 = (cl_int)-r; g.z1 = (cl_int)-r;
            g.x2 = (cl_int) r; g.z2 = (cl_int) r;
            g.useRadius = 1;
            g.rmaxsq = (cl_ulong)(r * r + 1);
        }
        else
        {
            g.x1 = c.x1; g.z1 = c.z1; g.x2 = c.x2; g.z2 = c.z2;
        }

        int64_t rs = (int64_t)sc.regionSize << 4;
        g.rx1 = (cl_int)floordiv64(g.x1, rs);
        g.rz1 = (cl_int)floordiv64(g.z1, rs);
        g.rx2 = (cl_int)floordiv64(g.x2, rs);
        g.rz2 = (cl_int)floordiv64(g.z2, rs);
        if (g.rx2 < g.rx1 || g.rz2 < g.rz1)
            continue;

        // Skip areas so large that a single seed would take too long on the
        // GPU (also protects against OS driver watchdog resets).
        int64_t nreg = (int64_t)(g.rx2 - g.rx1 + 1) * (int64_t)(g.rz2 - g.rz1 + 1);
        if (nreg > 20000)
            continue;
        if (nreg < g.count)
            continue; // can never be satisfied; the CPU will say so quickly

        conds.push_back(g);
    }

    // cheapest conditions first so most seeds are rejected early
    std::stable_sort(conds.begin(), conds.end(), [](const GpuCond& a, const GpuCond& b) {
        int64_t na = (int64_t)(a.rx2 - a.rx1 + 1) * (a.rz2 - a.rz1 + 1);
        int64_t nb = (int64_t)(b.rx2 - b.rx1 + 1) * (b.rz2 - b.rz1 + 1);
        return na < nb;
    });
    if ((int)conds.size() > GK_MAX_CONDS)
        conds.resize(GK_MAX_CONDS);
    return conds;
}

} // namespace

// ---------------------------------------------------------------------------

struct GpuFilter::Impl
{
    QMutex mutex;
    QLibrary lib;
    std::atomic<bool> ok{false};
    std::atomic<bool> failed{false};
    QString devname;
    int nconds = 0;
    uint64_t batch = 1u << 18;

    PFN_clGetPlatformIDs          GetPlatformIDs = nullptr;
    PFN_clGetDeviceIDs            GetDeviceIDs = nullptr;
    PFN_clGetDeviceInfo           GetDeviceInfo = nullptr;
    PFN_clCreateContext           CreateContext = nullptr;
    PFN_clCreateCommandQueue      CreateCommandQueue = nullptr;
    PFN_clCreateProgramWithSource CreateProgramWithSource = nullptr;
    PFN_clBuildProgram            BuildProgram = nullptr;
    PFN_clGetProgramBuildInfo     GetProgramBuildInfo = nullptr;
    PFN_clCreateKernel            CreateKernel = nullptr;
    PFN_clCreateBuffer            CreateBuffer = nullptr;
    PFN_clEnqueueWriteBuffer      EnqueueWriteBuffer = nullptr;
    PFN_clEnqueueReadBuffer       EnqueueReadBuffer = nullptr;
    PFN_clEnqueueNDRangeKernel    EnqueueNDRangeKernel = nullptr;
    PFN_clSetKernelArg            SetKernelArg = nullptr;
    PFN_clFinish                  Finish = nullptr;
    PFN_clReleaseMemObject        ReleaseMemObject = nullptr;
    PFN_clReleaseKernel           ReleaseKernel = nullptr;
    PFN_clReleaseProgram          ReleaseProgram = nullptr;
    PFN_clReleaseCommandQueue     ReleaseCommandQueue = nullptr;
    PFN_clReleaseContext          ReleaseContext = nullptr;

    cl_context ctx = nullptr;
    cl_command_queue queue = nullptr;
    cl_program prog = nullptr;
    cl_kernel kRange = nullptr;
    cl_kernel kList = nullptr;
    cl_mem bufConds = nullptr;
    cl_mem bufOut = nullptr;
    cl_mem bufCount = nullptr;
    cl_mem bufList = nullptr;
    size_t listCap = 0;
    enum : cl_uint { outcap = 1u << 20 };

    ~Impl() { cleanup(); }

    void cleanup()
    {
        if (bufList) ReleaseMemObject(bufList);
        if (bufCount) ReleaseMemObject(bufCount);
        if (bufOut) ReleaseMemObject(bufOut);
        if (bufConds) ReleaseMemObject(bufConds);
        if (kList) ReleaseKernel(kList);
        if (kRange) ReleaseKernel(kRange);
        if (prog) ReleaseProgram(prog);
        if (queue) ReleaseCommandQueue(queue);
        if (ctx) ReleaseContext(ctx);
        bufList = bufCount = bufOut = bufConds = nullptr;
        kList = kRange = nullptr;
        prog = nullptr; queue = nullptr; ctx = nullptr;
        ok = false;
    }

    template<typename F> bool load(F& fn, const char *name)
    {
        fn = reinterpret_cast<F>(lib.resolve(name));
        return fn != nullptr;
    }

    bool loadLibrary(QString *err)
    {
        const char *names[] = { "OpenCL", "libOpenCL.so.1", "OpenCL.dll", "/System/Library/Frameworks/OpenCL.framework/OpenCL" };
        for (const char *n : names)
        {
            lib.setFileName(QString::fromLatin1(n));
            if (lib.load())
                break;
        }
        if (!lib.isLoaded())
        {
            *err = "no OpenCL driver found";
            return false;
        }
        bool all =
            load(GetPlatformIDs, "clGetPlatformIDs") &&
            load(GetDeviceIDs, "clGetDeviceIDs") &&
            load(GetDeviceInfo, "clGetDeviceInfo") &&
            load(CreateContext, "clCreateContext") &&
            load(CreateCommandQueue, "clCreateCommandQueue") &&
            load(CreateProgramWithSource, "clCreateProgramWithSource") &&
            load(BuildProgram, "clBuildProgram") &&
            load(GetProgramBuildInfo, "clGetProgramBuildInfo") &&
            load(CreateKernel, "clCreateKernel") &&
            load(CreateBuffer, "clCreateBuffer") &&
            load(EnqueueWriteBuffer, "clEnqueueWriteBuffer") &&
            load(EnqueueReadBuffer, "clEnqueueReadBuffer") &&
            load(EnqueueNDRangeKernel, "clEnqueueNDRangeKernel") &&
            load(SetKernelArg, "clSetKernelArg") &&
            load(Finish, "clFinish") &&
            load(ReleaseMemObject, "clReleaseMemObject") &&
            load(ReleaseKernel, "clReleaseKernel") &&
            load(ReleaseProgram, "clReleaseProgram") &&
            load(ReleaseCommandQueue, "clReleaseCommandQueue") &&
            load(ReleaseContext, "clReleaseContext");
        if (!all)
        {
            *err = "incomplete OpenCL driver";
            return false;
        }
        return true;
    }

    // Picks the GPU with the most compute units.
    bool pickDevice(cl_platform_id *plat, cl_device_id *dev, QString *name)
    {
        cl_uint np = 0;
        if (GetPlatformIDs(0, nullptr, &np) != CL_SUCCESS || np == 0)
            return false;
        std::vector<cl_platform_id> plats(np);
        if (GetPlatformIDs(np, plats.data(), nullptr) != CL_SUCCESS)
            return false;

        cl_uint best = 0;
        bool found = false;
        for (cl_platform_id p : plats)
        {
            cl_uint nd = 0;
            if (GetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) != CL_SUCCESS || nd == 0)
                continue;
            std::vector<cl_device_id> devs(nd);
            if (GetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, devs.data(), nullptr) != CL_SUCCESS)
                continue;
            for (cl_device_id dv : devs)
            {
                cl_bool avail = 0, comp = 0;
                GetDeviceInfo(dv, CL_DEVICE_AVAILABLE, sizeof(avail), &avail, nullptr);
                GetDeviceInfo(dv, CL_DEVICE_COMPILER_AVAILABLE, sizeof(comp), &comp, nullptr);
                if (!avail || !comp)
                    continue;
                cl_uint cu = 0;
                GetDeviceInfo(dv, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);
                cl_bool unified = 0; // prefer discrete GPUs on a tie
                GetDeviceInfo(dv, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(unified), &unified, nullptr);
                cl_uint score = cu * 2 + (unified ? 0 : 1);
                if (!found || score > best)
                {
                    best = score;
                    found = true;
                    *plat = p;
                    *dev = dv;
                    char buf[256] = {0};
                    GetDeviceInfo(dv, CL_DEVICE_NAME, sizeof(buf) - 1, buf, nullptr);
                    *name = QString::fromLatin1(buf).trimmed();
                }
            }
        }
        return found;
    }

    bool setup(const std::vector<GpuCond>& conds, QString *err)
    {
        if (!loadLibrary(err))
            return false;

        cl_platform_id plat = nullptr;
        cl_device_id dev = nullptr;
        if (!pickDevice(&plat, &dev, &devname))
        {
            *err = "no OpenCL GPU found";
            return false;
        }

        cl_int e = 0;
        cl_context_properties props[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)plat, 0 };
        ctx = CreateContext(props, 1, &dev, nullptr, nullptr, &e);
        if (e != CL_SUCCESS || !ctx) { *err = "clCreateContext failed"; return false; }
        queue = CreateCommandQueue(ctx, dev, 0, &e);
        if (e != CL_SUCCESS || !queue) { *err = "clCreateCommandQueue failed"; return false; }

        QFile f(":/gpu/gpukernel.cl");
        if (!f.open(QIODevice::ReadOnly))
        {
            *err = "GPU kernel resource missing";
            return false;
        }
        QByteArray src = f.readAll();
        const char *sp = src.constData();
        size_t sl = (size_t)src.size();
        prog = CreateProgramWithSource(ctx, 1, &sp, &sl, &e);
        if (e != CL_SUCCESS || !prog) { *err = "clCreateProgramWithSource failed"; return false; }
        e = BuildProgram(prog, 1, &dev, "-DGK_OPENCL -cl-std=CL1.2", nullptr, nullptr);
        if (e != CL_SUCCESS)
        {
            size_t n = 0;
            GetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
            QByteArray log(n ? (int)n : 1, 0);
            GetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
            *err = "GPU kernel build failed: " + QString::fromLatin1(log).left(300);
            return false;
        }
        kRange = CreateKernel(prog, "gk_filter_range", &e);
        if (e != CL_SUCCESS || !kRange) { *err = "kernel gk_filter_range missing"; return false; }
        kList = CreateKernel(prog, "gk_filter_list", &e);
        if (e != CL_SUCCESS || !kList) { *err = "kernel gk_filter_list missing"; return false; }

        nconds = (int)conds.size();
        bufConds = CreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                conds.size() * sizeof(GpuCond), (void*)conds.data(), &e);
        if (e != CL_SUCCESS || !bufConds) { *err = "buffer allocation failed"; return false; }
        bufOut = CreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)outcap * sizeof(cl_ulong), nullptr, &e);
        if (e != CL_SUCCESS || !bufOut) { *err = "buffer allocation failed"; return false; }
        bufCount = CreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_uint), nullptr, &e);
        if (e != CL_SUCCESS || !bufCount) { *err = "buffer allocation failed"; return false; }
        ok = true;
        return true;
    }

    // One launch over 'total' items. For the list kernel 'start' is unused and
    // the list buffer must already hold the items.
    // Returns the number of survivors (may exceed outcap) or -1 on error.
    long launch(bool isList, uint64_t start, uint64_t total, std::vector<uint64_t>& out)
    {
        cl_kernel k = isList ? kList : kRange;
        cl_uint zero = 0;
        cl_int e = EnqueueWriteBuffer(queue, bufCount, CL_TRUE, 0, sizeof(zero), &zero, 0, nullptr, nullptr);
        if (e != CL_SUCCESS) return -1;

        cl_int nc = nconds;
        cl_ulong ctot = total;
        cl_ulong cstart = start;
        cl_uint cap = outcap;
        cl_uint a = 0;
        e  = SetKernelArg(k, a++, sizeof(cl_mem), &bufConds);
        e |= SetKernelArg(k, a++, sizeof(nc), &nc);
        if (isList) e |= SetKernelArg(k, a++, sizeof(cl_mem), &bufList);
        else        e |= SetKernelArg(k, a++, sizeof(cstart), &cstart);
        e |= SetKernelArg(k, a++, sizeof(ctot), &ctot);
        e |= SetKernelArg(k, a++, sizeof(cl_mem), &bufOut);
        e |= SetKernelArg(k, a++, sizeof(cl_mem), &bufCount);
        e |= SetKernelArg(k, a++, sizeof(cap), &cap);
        if (e != CL_SUCCESS) return -1;

        size_t gws = (size_t)total;
        e = EnqueueNDRangeKernel(queue, k, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
        if (e != CL_SUCCESS) return -1;

        cl_uint cnt = 0;
        e = EnqueueReadBuffer(queue, bufCount, CL_TRUE, 0, sizeof(cnt), &cnt, 0, nullptr, nullptr);
        if (e != CL_SUCCESS) return -1;

        cl_uint n = cnt < outcap ? cnt : outcap;
        if (n && cnt <= outcap)
        {
            size_t old = out.size();
            out.resize(old + n);
            e = EnqueueReadBuffer(queue, bufOut, CL_TRUE, 0, n * sizeof(cl_ulong), out.data() + old, 0, nullptr, nullptr);
            if (e != CL_SUCCESS) return -1;
        }
        return (long)cnt;
    }

    bool ensureList(size_t n)
    {
        if (bufList && listCap >= n)
            return true;
        if (bufList) { ReleaseMemObject(bufList); bufList = nullptr; }
        cl_int e = 0;
        bufList = CreateBuffer(ctx, CL_MEM_READ_ONLY, n * sizeof(cl_ulong), nullptr, &e);
        if (e != CL_SUCCESS || !bufList) { bufList = nullptr; return false; }
        listCap = n;
        return true;
    }

    // Range launch with automatic splitting on survivor-buffer overflow.
    bool runRange(uint64_t start, uint64_t total, std::vector<uint64_t>& out, int depth = 0)
    {
        size_t before = out.size();
        long n = launch(false, start, total, out);
        if (n < 0)
            return false;
        if ((unsigned long)n > outcap)
        {
            out.resize(before);
            if (total < 2 || depth > 40)
                return false;
            uint64_t h = total / 2;
            return runRange(start, h, out, depth + 1) &&
                   runRange(start + h, total - h, out, depth + 1);
        }
        return true;
    }

    bool runList(const uint64_t *list, uint64_t total, std::vector<uint64_t>& out, int depth = 0)
    {
        if (!ensureList((size_t)total))
            return false;
        cl_int e = EnqueueWriteBuffer(queue, bufList, CL_TRUE, 0, (size_t)total * sizeof(cl_ulong), list, 0, nullptr, nullptr);
        if (e != CL_SUCCESS)
            return false;
        size_t before = out.size();
        long n = launch(true, 0, total, out);
        if (n < 0)
            return false;
        if ((unsigned long)n > outcap)
        {
            out.resize(before);
            if (total < 2 || depth > 40)
                return false;
            uint64_t h = total / 2;
            return runList(list, h, out, depth + 1) &&
                   runList(list + h, total - h, out, depth + 1);
        }
        return true;
    }

    // keeps single launches around 20-80 ms to stay clear of OS GPU watchdogs
    void adapt(qint64 ns)
    {
        if (ns < 20000000 && batch < (1u << 24))
            batch *= 2;
        else if (ns > 80000000 && batch > 4096)
            batch /= 2;
    }
};

// ---------------------------------------------------------------------------

GpuFilter::GpuFilter() : d(new Impl) {}
GpuFilter::~GpuFilter() { delete d; }

bool GpuFilter::init(const ConditionTree& tree, int mc, QString *info)
{
    QString err;
    std::vector<GpuCond> conds = buildConds(tree, mc);
    if (conds.empty())
    {
        if (info) *info = "no GPU-accelerable structure conditions";
        return false;
    }
    if (!d->setup(conds, &err))
    {
        d->cleanup();
        d->failed = true;
        if (info) *info = err;
        return false;
    }
    if (info) *info = d->devname;
    return true;
}

bool GpuFilter::ready() const { return d->ok && !d->failed; }
QString GpuFilter::deviceName() const { return d->devname; }
int GpuFilter::conditionCount() const { return d->nconds; }
uint64_t GpuFilter::batchHint() const { return d->batch; }

bool GpuFilter::filterRange(uint64_t start, uint64_t count,
                            std::vector<uint64_t>& out, const std::atomic_bool *stop)
{
    out.clear();
    QMutexLocker lock(&d->mutex);
    if (!ready())
        return false;
    uint64_t done = 0;
    while (done < count)
    {
        if (stop && *stop)
            return true; // caller checks stop itself
        uint64_t n = std::min<uint64_t>(count - done, d->batch);
        QElapsedTimer t;
        t.start();
        if (!d->runRange(start + done, n, out))
        {
            d->failed = true;
            out.clear();
            return false;
        }
        d->adapt(t.nsecsElapsed());
        done += n;
    }
    std::sort(out.begin(), out.end());
    return true;
}

bool GpuFilter::filterList(const uint64_t *list, uint64_t count,
                           std::vector<uint64_t>& out, const std::atomic_bool *stop)
{
    out.clear();
    QMutexLocker lock(&d->mutex);
    if (!ready())
        return false;
    uint64_t done = 0;
    while (done < count)
    {
        if (stop && *stop)
            return true;
        uint64_t n = std::min<uint64_t>(count - done, d->batch);
        QElapsedTimer t;
        t.start();
        if (!d->runList(list + done, n, out))
        {
            d->failed = true;
            out.clear();
            return false;
        }
        d->adapt(t.nsecsElapsed());
        done += n;
    }
    return true;
}
