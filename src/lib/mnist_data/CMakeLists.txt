add_library(mnist_data SHARED mnist_data.h mnist_data.cpp)
generate_export_header(mnist_data)
file(COPY ${CMAKE_CURRENT_BINARY_DIR}/mnist_data_export.h DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})