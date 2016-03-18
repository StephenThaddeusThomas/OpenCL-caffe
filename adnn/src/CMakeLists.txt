include_directories(.)
file(GLOB SOURCES *.cpp *.h)

add_library(aLibDNN ${SOURCES})

#add_subdirectory(aLibDNNDriver)
