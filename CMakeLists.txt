CMAKE_MINIMUM_REQUIRED(VERSION 3.13 FATAL_ERROR)

PROJECT(CudaVoxelize LANGUAGES CXX)

IF(MSVC)
ENABLE_LANGUAGE(CUDA)
ENDIF()

FIND_PACKAGE(glm REQUIRED)
FIND_PACKAGE(OpenMP REQUIRED)

IF(NOT MSVC)
FIND_PACKAGE(CUDA REQUIRED)
ENDIF()

SET(CUDA_VOXELIZER_EXECUTABLE cuda_voxelizer)

SET(Trimesh2_INCLUDE_DIR CACHE PATH "Path to Trimesh2 includes")
SET(CUDA_ARCH CACHE STRING "CUDA compute capability. It is prefer to set native value for your video card. Example: 61")

IF(NOT CUDA_ARCH)
	MESSAGE(FATAL_ERROR "You must set CUDA_ARCH variable. For example: 61.")
ENDIF()

IF(NOT Trimesh2_INCLUDE_DIR)
  MESSAGE(FATAL_ERROR "You need to set variable Trimesh2_INCLUDE_DIR")
ENDIF()

FIND_FILE(Trimesh2_TriMesh_h TriMesh.h ${Trimesh2_INCLUDE_DIR})
IF(NOT Trimesh2_TriMesh_h)
	message(FATAL_ERROR "Can't find TriMesh.h in ${Trimesh2_INCLUDE_DIR}")
ENDIF()
MARK_AS_ADVANCED(Trimesh2_TriMesh_h)

SET(Trimesh2_LINK_DIR CACHE PATH "Path to Trimesh2 library dir.")

IF(NOT Trimesh2_LINK_DIR)
  MESSAGE(FATAL_ERROR "You need to set variable Trimesh2_LINK_DIR")
ENDIF()

IF(NOT EXISTS "${Trimesh2_LINK_DIR}")
  MESSAGE(FATAL_ERROR "Trimesh2 library dir does not exist")
ENDIF()

FIND_LIBRARY(Trimesh2_LIBRARY trimesh ${Trimesh2_LINK_DIR})

IF(NOT Trimesh2_LIBRARY)
	message(SEND_ERROR "Can't find libtrimesh.a in ${Trimesh2_LINK_DIR}")
ENDIF()
MARK_AS_ADVANCED(Trimesh2_LIBRARY)

MESSAGE(STATUS "CUDA compute capability set to ${CUDA_ARCH}")
MESSAGE(STATUS "Found Trimesh2 include: ${Trimesh2_TriMesh_h}")
MESSAGE(STATUS "Found Trimesh2 lib: ${Trimesh2_LIBRARY}")

SET(CUDA_VOXELIZER_SRCS
  ./src/main.cpp
  ./src/util_cuda.cpp
  ./src/util_io.cpp
  ./src/cpu_voxelizer.cpp
)
SET(CUDA_VOXELIZER_SRCS_CU
  ./src/voxelize.cu
  ./src/thrust_operations.cu
  ./src/voxelize_solid.cu
)

IF(MSVC)
ADD_EXECUTABLE(
  ${CUDA_VOXELIZER_EXECUTABLE}
  ${CUDA_VOXELIZER_SRCS}
  ${CUDA_VOXELIZER_SRCS_CU})
ELSE()
CUDA_ADD_EXECUTABLE(
  ${CUDA_VOXELIZER_EXECUTABLE}
  ${CUDA_VOXELIZER_SRCS}
  ${CUDA_VOXELIZER_SRCS_CU})
ENDIF()

TARGET_COMPILE_FEATURES(${CUDA_VOXELIZER_EXECUTABLE} PRIVATE cxx_std_11)
TARGET_INCLUDE_DIRECTORIES(${CUDA_VOXELIZER_EXECUTABLE} PRIVATE ${Trimesh2_INCLUDE_DIR})
TARGET_LINK_LIBRARIES (${CUDA_VOXELIZER_EXECUTABLE} ${Trimesh2_LIBRARY} glm::glm OpenMP::OpenMP_CXX)
TARGET_COMPILE_OPTIONS(${CUDA_VOXELIZER_EXECUTABLE} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}>)
