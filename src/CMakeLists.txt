cmake_minimum_required(VERSION 3.11)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project(mnist_deep_ann)

include_directories(lib)

add_subdirectory(lib)
add_subdirectory(bin)
add_subdirectory(test)