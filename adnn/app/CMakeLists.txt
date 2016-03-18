include_directories(.)
file(GLOB SOURCES *.cpp *.h)

add_executable(aLibDNNDriver ${SOURCES})
target_link_libraries(aLibDNNDriver aLibDNN ${OpenCL_LIBRARIES} ${CLBLAS_LIBRARIES})
