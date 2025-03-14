#!/bin/bash

# This script assumes you have already installed Nvidia CUDA and cudnn
# https://developer.nvidia.com/cuda-downloads
# https://developer.nvidia.com/cudnn

# Check if the CUDA_ARCH_BIN argument is provided
if [ -z "$1" ]; then
    echo "Error: CUDA_ARCH_BIN argument is required."
    echo "Usage: ./build-opencv.sh <CUDA_ARCH_BIN>"
    exit 1
fi

# Assign the provided CUDA_ARCH_BIN argument to a variable
CUDA_ARCH_BIN=$1

# Ensure the virtual environment is activated and VIRTUAL_ENV is set
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: You must activate a virtual environment first."
    echo "Usage: source <path_to_virtualenv>/bin/activate"
    exit 1
fi

cd ./tools/opencv
# Check if the build directory exists
if [ -d "build" ]; then
    echo "Cleaning up the existing build directory..."
    rm -rf build
fi

mkdir -p build && cd build

# Set the CMAKE_INSTALL_PREFIX to the virtual environment's site-packages directory
INSTALL_DIR=$VIRTUAL_ENV/lib/opencv

cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=$INSTALL_DIR \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D WITH_CUDA=ON \
      -D CUDA_ARCH_BIN=$CUDA_ARCH_BIN \
      -D WITH_CUDNN=ON \
      -D WITH_NVCUVID=ON \
      -D WITH_NVCUVENC=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_CUBLAS=1 \
      -D WITH_LIBV4L=ON \
      -D WITH_FFMPEG=ON \
      -D WITH_OPENGL=ON \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_opencv_python3=ON \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D CMAKE_CXX_STANDARD=17 \
      -D CUDA_NVCC_FLAGS="-std=c++17" \
      -D BUILD_opencv_python3=ON \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_DOCS=OFF \
      -D WITH_JAVA=OFF \
      -D WITH_TBB=ON \
      -D WITH_OPENCL=ON \
      -D WITH_MATLAB=OFF \
      ..


# Start the compilation using 8 cores (don't use all cores to avoid outofmemory error)
cmake --build . --parallel 8

