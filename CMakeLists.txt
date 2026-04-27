cmake_minimum_required(VERSION 3.28)
project(ggml-sample)

set(CMAKE_CXX_STANDARD 20)

# 查找本机安装的 ggml
find_package(ggml CONFIG REQUIRED)

add_executable(tensor_add tensor_add.cpp)
target_link_libraries(tensor_add PRIVATE ggml::ggml)

add_executable(tensor_basicapi tensor_basicapi.cpp)
target_link_libraries(tensor_basicapi PRIVATE ggml::ggml)

add_executable(tensor_func tensor_func.cpp)
target_link_libraries(tensor_func PRIVATE ggml::ggml)

add_executable(auto_diff auto_diff.cpp)
target_link_libraries(auto_diff PRIVATE ggml::ggml)
