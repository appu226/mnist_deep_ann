add_library(command_line_options SHARED clo.hpp clo.cpp)
generate_export_header(command_line_options)
file(COPY ${CMAKE_CURRENT_BINARY_DIR}/command_line_options_export.h DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})