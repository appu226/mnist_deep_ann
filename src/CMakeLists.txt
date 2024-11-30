cmake_minimum_required(VERSION 3.12)

include(GenerateExportHeader)


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()


file(TO_NATIVE_PATH ${CMAKE_CURRENT_BINARY_DIR} NATIVE_BINARY_PATH)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/set_path_template.bat" 
    "${CMAKE_CURRENT_BINARY_DIR}/set_path.bat"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Set default compile flags for GCC
if(CMAKE_COMPILER_IS_GNUCXX)
    message(STATUS "GCC detected, adding compile flags")
    set(CMAKE_CXX_FLAGS "-Wall -Wextra")
endif(CMAKE_COMPILER_IS_GNUCXX)
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project(mnist_deep_ann)

include_directories(lib)

add_subdirectory(lib)
add_subdirectory(bin)
add_subdirectory(test)