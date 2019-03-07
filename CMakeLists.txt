CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT ( CudaVoxelizer )

SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

FIND_PACKAGE(CUDA QUIET REQUIRED)
FIND_PACKAGE(GLM REQUIRED)
 
SET(CUDA_VOXELIZER_EXECUTABLE cuda_voxelizer)

SET(Trimesh2_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/thirdparty/trimesh2/include" CACHE PATH "Path to Trimesh2 includes")

FIND_FILE(Trimesh2_TriMesh_h TriMesh.h ${Trimesh2_INCLUDE_DIR})
IF(NOT Trimesh2_TriMesh_h)
	message(SEND_ERROR "Can't find TriMesh.h in ${Trimesh2_INCLUDE_DIR}")
ENDIF()
MARK_AS_ADVANCED(Trimesh2_TriMesh_h)
INCLUDE_DIRECTORIES(${Trimesh2_INCLUDE_DIR})

SET(Trimesh2_LINK_DIR "${CMAKE_SOURCE_DIR}/thirdparty/trimesh2/lib.Linux64" CACHE PATH "Path to Trimesh2 libraries")
FIND_LIBRARY(Trimesh2_LIBRARY trimesh ${Trimesh2_LINK_DIR})
IF(NOT Trimesh2_LIBRARY)
	message(SEND_ERROR "Can't find libtrimesh.a in ${Trimesh2_LINK_DIR}")
ENDIF()
MARK_AS_ADVANCED(Trimesh2_LIBRARY)

SET(CUDA_VOXELIZER_SRCS
  ./src/main.cpp
  ./src/util_cuda.cpp
  ./src/util_io.cpp
)
SET(CUDA_VOXELIZER_SRCS_CU
  ./src/voxelize.cu
  ./src/thrust_operations.cu
)

CUDA_ADD_EXECUTABLE(
  ${CUDA_VOXELIZER_EXECUTABLE} 
  ${CUDA_VOXELIZER_SRCS}
  ${CUDA_VOXELIZER_SRCS_CU})

TARGET_INCLUDE_DIRECTORIES(	${CUDA_VOXELIZER_EXECUTABLE} PUBLIC
  ${Trimesh2_INCLUDE_DIR})

TARGET_LINK_LIBRARIES ( ${CUDA_VOXELIZER_EXECUTABLE}
  ${Trimesh2_LIBRARY}
  gomp
)
