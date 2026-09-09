cmake_minimum_required(VERSION 3.18)

project(Phase1_Tensor LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 75)

find_package(CUDAToolkit REQUIRED)

add_library(tensor_lib
    tensor.cpp
    tensor_ops.cpp
    tensor_print.cpp
    backend.cpp
    cpu_backend.cpp
    cuda_backend.cu
    matmul.cpp
)

target_include_directories(tensor_lib PUBLIC
    .
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(tensor_lib PUBLIC
    CUDA::cudart
    CUDA::cublas
)

if(APPLE)
    target_link_libraries(tensor_lib PUBLIC
        "-framework Accelerate"
    )
else()
    find_package(BLAS REQUIRED)

    target_link_libraries(tensor_lib PUBLIC
        BLAS::BLAS
    )
endif()

add_executable(cuda_tensor_test
    ../../Tests/cuda_tensor_test.cpp
)

target_link_libraries(cuda_tensor_test PRIVATE
    tensor_lib
    CUDA::cudart
)