// A CPU "OpenCL driver" for testing the host code in src/gpusearch.cpp.
// It implements just the entry points the host uses and runs the kernel logic
// (src/gpukernel.cl, included as C++) in plain loops. Results are produced in
// reverse order on purpose, like unordered GPU atomics.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include "../src/cl_min.h"

typedef uint64_t ulong;
typedef int64_t  slong;
typedef uint32_t uint;
#define __constant
#define __global
#define __kernel
#include "../src/gpukernel.cl"

struct MBuf { std::vector<unsigned char> d; };
struct MKernel { bool isList; std::vector<std::vector<unsigned char>> args; };
struct MProg { std::string src; };

#define API extern "C" __attribute__((visibility("default")))

static int g_plat, g_dev;
long g_mock_launches = 0;
API long mock_launch_count() { return g_mock_launches; }

API cl_int CL_API_CALL clGetPlatformIDs(cl_uint n, cl_platform_id *p, cl_uint *np)
{ if (np) *np = 1; if (p && n) p[0] = (cl_platform_id)&g_plat; return CL_SUCCESS; }
API cl_int CL_API_CALL clGetDeviceIDs(cl_platform_id, cl_device_type t, cl_uint n, cl_device_id *d, cl_uint *nd)
{
    if (!(t & CL_DEVICE_TYPE_GPU)) { if (nd) *nd = 0; return -1; }
    if (nd) *nd = 1; if (d && n) d[0] = (cl_device_id)&g_dev; return CL_SUCCESS;
}
API cl_int CL_API_CALL clGetDeviceInfo(cl_device_id, cl_device_info i, size_t sz, void *v, size_t *r)
{
    if (r) *r = 0;
    switch (i) {
    case CL_DEVICE_AVAILABLE: case CL_DEVICE_COMPILER_AVAILABLE: *(cl_bool*)v = 1; break;
    case CL_DEVICE_MAX_COMPUTE_UNITS: *(cl_uint*)v = 8; break;
    case CL_DEVICE_HOST_UNIFIED_MEMORY: *(cl_bool*)v = 0; break;
    case CL_DEVICE_NAME: snprintf((char*)v, sz, "Mock GPU (CPU emulation)"); break;
    default: break;
    }
    return CL_SUCCESS;
}
API cl_context CL_API_CALL clCreateContext(const cl_context_properties*, cl_uint, const cl_device_id*, void (CL_API_CALL *)(const char*, const void*, size_t, void*), void*, cl_int *e)
{ *e = CL_SUCCESS; return (cl_context)new int(1); }
API cl_command_queue CL_API_CALL clCreateCommandQueue(cl_context, cl_device_id, cl_command_queue_properties, cl_int *e)
{ *e = CL_SUCCESS; return (cl_command_queue)new int(2); }
API cl_program CL_API_CALL clCreateProgramWithSource(cl_context, cl_uint, const char **s, const size_t *l, cl_int *e)
{ *e = CL_SUCCESS; MProg *p = new MProg; p->src.assign(s[0], l[0]); return (cl_program)p; }
API cl_int CL_API_CALL clBuildProgram(cl_program p, cl_uint, const cl_device_id*, const char *opt, void (CL_API_CALL *)(cl_program, void*), void*)
{
    MProg *m = (MProg*)p;
    if (!opt || !strstr(opt, "-DGK_OPENCL")) return -11;           // CL_BUILD_PROGRAM_FAILURE
    if (m->src.find("gk_filter_range") == std::string::npos) return -11;
    return CL_SUCCESS;
}
API cl_int CL_API_CALL clGetProgramBuildInfo(cl_program, cl_device_id, cl_program_build_info, size_t, void *v, size_t *r)
{ if (r) *r = 1; if (v) *(char*)v = 0; return CL_SUCCESS; }
API cl_kernel CL_API_CALL clCreateKernel(cl_program, const char *name, cl_int *e)
{
    MKernel *k = new MKernel; k->args.resize(8);
    if (!strcmp(name, "gk_filter_range")) k->isList = false;
    else if (!strcmp(name, "gk_filter_list")) k->isList = true;
    else { *e = -46; delete k; return nullptr; }
    *e = CL_SUCCESS; return (cl_kernel)k;
}
API cl_mem CL_API_CALL clCreateBuffer(cl_context, cl_mem_flags f, size_t sz, void *host, cl_int *e)
{
    MBuf *b = new MBuf; b->d.assign(sz, 0);
    if ((f & CL_MEM_COPY_HOST_PTR) && host) memcpy(b->d.data(), host, sz);
    *e = CL_SUCCESS; return (cl_mem)b;
}
API cl_int CL_API_CALL clEnqueueWriteBuffer(cl_command_queue, cl_mem m, cl_bool, size_t off, size_t sz, const void *p, cl_uint, const cl_event*, cl_event*)
{ MBuf *b = (MBuf*)m; if (off + sz > b->d.size()) return -30; memcpy(b->d.data() + off, p, sz); return CL_SUCCESS; }
API cl_int CL_API_CALL clEnqueueReadBuffer(cl_command_queue, cl_mem m, cl_bool, size_t off, size_t sz, void *p, cl_uint, const cl_event*, cl_event*)
{ MBuf *b = (MBuf*)m; if (off + sz > b->d.size()) return -30; memcpy(p, b->d.data() + off, sz); return CL_SUCCESS; }
API cl_int CL_API_CALL clSetKernelArg(cl_kernel k, cl_uint i, size_t sz, const void *v)
{
    MKernel *m = (MKernel*)k; if (i >= 8) return -49;
    m->args[i].assign((const unsigned char*)v, (const unsigned char*)v + sz); return CL_SUCCESS;
}
template<class T> static T arg(MKernel *k, int i) { T v; memcpy(&v, k->args[i].data(), sizeof(T)); return v; }

API cl_int CL_API_CALL clEnqueueNDRangeKernel(cl_command_queue, cl_kernel kk, cl_uint, const size_t*, const size_t *gws, const size_t*, cl_uint, const cl_event*, cl_event*)
{
    MKernel *k = (MKernel*)kk;
    g_mock_launches++;
    MBuf *conds = arg<MBuf*>(k, 0);
    int nconds = arg<int>(k, 1);
    if (nconds < 1 || nconds > GK_MAX_CONDS || conds->d.size() < (size_t)nconds * sizeof(GkCond)) return -30;
    const GkCond *cv = (const GkCond*)conds->d.data();
    ulong start = 0; MBuf *list = nullptr;
    if (k->isList) list = arg<MBuf*>(k, 2); else start = arg<ulong>(k, 2);
    ulong total = arg<ulong>(k, 3);
    MBuf *out = arg<MBuf*>(k, 4);
    MBuf *cnt = arg<MBuf*>(k, 5);
    uint cap = arg<uint>(k, 6);
    if (out->d.size() < (size_t)cap * 8) return -30;
    uint *pc = (uint*)cnt->d.data();
    ulong *po = (ulong*)out->d.data();
    for (ulong g = *gws; g-- > 0;)
    {
        if (g >= total) continue;
        ulong seed = k->isList ? ((ulong*)list->d.data())[g] : start + g;
        if (gk_seed_passes(cv, nconds, seed)) { uint idx = (*pc)++; if (idx < cap) po[idx] = seed; }
    }
    return CL_SUCCESS;
}
API cl_int CL_API_CALL clFinish(cl_command_queue) { return CL_SUCCESS; }
API cl_int CL_API_CALL clReleaseMemObject(cl_mem m) { delete (MBuf*)m; return CL_SUCCESS; }
API cl_int CL_API_CALL clReleaseKernel(cl_kernel k) { delete (MKernel*)k; return CL_SUCCESS; }
API cl_int CL_API_CALL clReleaseProgram(cl_program p) { delete (MProg*)p; return CL_SUCCESS; }
API cl_int CL_API_CALL clReleaseCommandQueue(cl_command_queue) { return CL_SUCCESS; }
API cl_int CL_API_CALL clReleaseContext(cl_context) { return CL_SUCCESS; }
