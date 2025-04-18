add_library(ann SHARED ann_interface.h ann_interface.cpp simple_activation.h simple_activation.cpp
                       convolution_hierarchy.h convolution_hierarchy.cpp)
generate_export_header(ann)
file(COPY ${CMAKE_CURRENT_BINARY_DIR}/ann_export.h DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})