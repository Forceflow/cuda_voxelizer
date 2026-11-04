#pragma once
// Maps the CUDA runtime API used by this project onto HIP. Only on the include
// path of a USE_HIP build; the CUDA build uses the real CUDA headers.

#include <hip/hip_runtime.h>

// helper_cuda.h guards its error-checking and device-selection helpers on
// these CUDA header guards.
#ifndef __DRIVER_TYPES_H__
#define __DRIVER_TYPES_H__
#endif
#ifndef __CUDA_RUNTIME_H__
#define __CUDA_RUNTIME_H__
#endif

#define cudaError hipError_t
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaErrorInvalidValue hipErrorInvalidValue
#define cudaGetErrorName hipGetErrorName
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError

#define cudaDeviceProp hipDeviceProp_t
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaMemGetInfo hipMemGetInfo
#define cudaComputeModeProhibited hipComputeModeProhibited
#define cudaDevAttrClockRate hipDeviceAttributeClockRate
#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor
#define cudaDevAttrComputeMode hipDeviceAttributeComputeMode
#define cudaDevAttrIntegrated hipDeviceAttributeIntegrated
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount

#define cudaMallocManaged hipMallocManaged
#define cudaMemset hipMemset
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyToSymbol hipMemcpyToSymbol
#define cudaMemcpyFromSymbol hipMemcpyFromSymbol

#define cudaEvent_t hipEvent_t
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime

#define cudaOccupancyMaxPotentialBlockSize hipOccupancyMaxPotentialBlockSize
