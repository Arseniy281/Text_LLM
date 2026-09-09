cmake_minimum_required(VERSION 3.10)
project(Phase1_Tensor)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(tensor_lib 
    tensor.cpp
    tensor_ops.cpp
    tensor_print.cpp
    backend.cpp
    cpu_backend.cpp
    matmul.cpp
)

target_include_directories(tensor_lib PUBLIC .)
