# This Dockerfile uses the nvidia cuda container but will work without cuda too

FROM nvidia/cuda:12.1.0-devel-ubuntu20.04
ENV CUDAARCHS='60'
ENV TRIMESH_VERSION='2020.03.04'
ENV CMAKE_VERSION='3.20.4'

WORKDIR /installation/

RUN apt update
RUN apt install -y --no-install-recommends apt-utils
RUN apt install -y build-essential libglm-dev libgomp1 git mesa-common-dev libglu1-mesa-dev libxi-dev wget ninja-build

WORKDIR /installation/cuda_voxelizer

COPY . .

RUN wget -q -O ./cmake-install.sh https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.sh 
RUN chmod u+x ./cmake-install.sh
RUN mkdir /cmake
RUN ./cmake-install.sh --skip-license --prefix=/cmake

WORKDIR /installation/
RUN git clone --single-branch --depth 1 -b ${TRIMESH_VERSION} https://github.com/Forceflow/trimesh2.git trimesh2
WORKDIR /installation/trimesh2
RUN pwd
RUN make all -j $(nproc)

ENV PATH="${PATH}:$HOME"/cmake/bin

WORKDIR /installation/cuda_voxelizer

RUN cmake -GNinja \
        -DTrimesh2_INCLUDE_DIR="/installation/trimesh2/include" \
        -DTrimesh2_LINK_DIR="/installation/trimesh2/lib.Linux64" \
        -S . -B ./build


RUN PATH=$PATH:"$HOME"/cmake/bin
RUN cmake --build ./build --parallel $(nproc)

FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

WORKDIR /app/

COPY --from=0 /installation/cuda_voxelizer/build/cuda_voxelizer ./
COPY --from=0 /installation/cuda_voxelizer/test_models/bunny.OBJ ./

RUN ./cuda_voxelizer -o binvox -thrust -f ./bunny.OBJ -s 256
#RUN ./cuda_voxelizer -o binvox -thrust -f ./bunny.OBJ -s 512
#RUN ./cuda_voxelizer -o binvox -thrust -f ./bunny.OBJ -s 1024
#RUN ./cuda_voxelizer -o binvox -thrust -f ./bunny.OBJ -s 2048

ENTRYPOINT ["/app/cuda_voxelizer"]
