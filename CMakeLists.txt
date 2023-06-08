CMAKE_MINIMUM_REQUIRED(VERSION 3.20 FATAL_ERROR)

PROJECT(CudaVoxelize LANGUAGES CXX CUDA)

FIND_PACKAGE(OpenMP REQUIRED)
FIND_PACKAGE(CUDAToolkit REQUIRED)

SET(CUDA_VOXELIZER_EXECUTABLE cuda_voxelizer)

SET(Trimesh2_INCLUDE_DIR CACHE PATH "Path to Trimesh2 includes")

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

MESSAGE(STATUS "Found Trimesh2 include: ${Trimesh2_TriMesh_h}")
MESSAGE(STATUS "Found Trimesh2 lib: ${Trimesh2_LIBRARY}")

SET(CUDA_VOXELIZER_SRCS
  ./src/main.cpp
  ./src/util_cuda.cpp
  ./src/util_io.cpp
  ./src/cpu_voxelizer.cpp
  ./src/libs/magicavoxel_file_writer/VoxWriter.cpp
)
SET(CUDA_VOXELIZER_SRCS_CU
  ./src/voxelize.cu
  ./src/voxelize_solid.cu
)

ADD_EXECUTABLE(
  ${CUDA_VOXELIZER_EXECUTABLE}
  ${CUDA_VOXELIZER_SRCS}
  ${CUDA_VOXELIZER_SRCS_CU})

TARGET_COMPILE_FEATURES(${CUDA_VOXELIZER_EXECUTABLE} PRIVATE cxx_std_17)
TARGET_INCLUDE_DIRECTORIES(${CUDA_VOXELIZER_EXECUTABLE} PRIVATE ${Trimesh2_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(${CUDA_VOXELIZER_EXECUTABLE} PRIVATE ${Trimesh2_LIBRARY} PRIVATE OpenMP::OpenMP_CXX PRIVATE CUDA::cudart)
