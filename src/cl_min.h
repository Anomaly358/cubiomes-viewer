#ifndef CL_MIN_H
#define CL_MIN_H

// Minimal OpenCL 1.2 declarations, so the viewer builds without an OpenCL SDK
// and loads the vendor driver (OpenCL.dll / libOpenCL.so.1) at runtime.

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#define CL_API_CALL __stdcall
#else
#define CL_API_CALL
#endif

typedef int32_t     cl_int;
typedef uint32_t    cl_uint;
typedef uint64_t    cl_ulong;
typedef cl_uint     cl_bool;
typedef cl_ulong    cl_bitfield;
typedef cl_bitfield cl_device_type;
typedef cl_bitfield cl_mem_flags;
typedef cl_bitfield cl_command_queue_properties;
typedef intptr_t    cl_context_properties;
typedef cl_uint     cl_device_info;
typedef cl_uint     cl_platform_info;
typedef cl_uint     cl_program_build_info;

typedef struct _cl_platform_id    *cl_platform_id;
typedef struct _cl_device_id      *cl_device_id;
typedef struct _cl_context        *cl_context;
typedef struct _cl_command_queue  *cl_command_queue;
typedef struct _cl_mem            *cl_mem;
typedef struct _cl_program        *cl_program;
typedef struct _cl_kernel         *cl_kernel;
typedef struct _cl_event          *cl_event;

enum
{
    CL_SUCCESS                  = 0,
    CL_TRUE                     = 1,
    CL_DEVICE_TYPE_GPU          = 1 << 2,
    CL_DEVICE_TYPE_ALL          = 0xFFFFFFFF,
    CL_DEVICE_TYPE              = 0x1000,
    CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002,
    CL_DEVICE_GLOBAL_MEM_SIZE   = 0x101F,
    CL_DEVICE_AVAILABLE         = 0x1027,
    CL_DEVICE_COMPILER_AVAILABLE= 0x1028,
    CL_DEVICE_NAME              = 0x102B,
    CL_DEVICE_HOST_UNIFIED_MEMORY = 0x1035,
    CL_CONTEXT_PLATFORM         = 0x1084,
    CL_PROGRAM_BUILD_LOG        = 0x1183,
    CL_MEM_READ_WRITE           = 1 << 0,
    CL_MEM_READ_ONLY            = 1 << 2,
    CL_MEM_COPY_HOST_PTR        = 1 << 5,
};

typedef cl_int (CL_API_CALL *PFN_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (CL_API_CALL *PFN_clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
typedef cl_int (CL_API_CALL *PFN_clGetDeviceInfo)(cl_device_id, cl_device_info, size_t, void*, size_t*);
typedef cl_context (CL_API_CALL *PFN_clCreateContext)(const cl_context_properties*, cl_uint, const cl_device_id*,
        void (CL_API_CALL *)(const char*, const void*, size_t, void*), void*, cl_int*);
typedef cl_command_queue (CL_API_CALL *PFN_clCreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
typedef cl_program (CL_API_CALL *PFN_clCreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int (CL_API_CALL *PFN_clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*,
        void (CL_API_CALL *)(cl_program, void*), void*);
typedef cl_int (CL_API_CALL *PFN_clGetProgramBuildInfo)(cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*);
typedef cl_kernel (CL_API_CALL *PFN_clCreateKernel)(cl_program, const char*, cl_int*);
typedef cl_mem (CL_API_CALL *PFN_clCreateBuffer)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
typedef cl_int (CL_API_CALL *PFN_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (CL_API_CALL *PFN_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (CL_API_CALL *PFN_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (CL_API_CALL *PFN_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
typedef cl_int (CL_API_CALL *PFN_clFinish)(cl_command_queue);
typedef cl_int (CL_API_CALL *PFN_clReleaseMemObject)(cl_mem);
typedef cl_int (CL_API_CALL *PFN_clReleaseKernel)(cl_kernel);
typedef cl_int (CL_API_CALL *PFN_clReleaseProgram)(cl_program);
typedef cl_int (CL_API_CALL *PFN_clReleaseCommandQueue)(cl_command_queue);
typedef cl_int (CL_API_CALL *PFN_clReleaseContext)(cl_context);

#endif // CL_MIN_H
