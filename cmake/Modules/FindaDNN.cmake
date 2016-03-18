# File:FindaDNN.cmake
# Type:CMake Module
# From:FindOpenCL.cmake
# Date:160228
# 
# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ########################################################################


# Locate an the include files and the library for the AMD DNN (aDNN) implementation.
#
# Defines the following variables:
#
#   ADNN_FOUND - Found the AMD DNN framework
#   ADNN_INCLUDE_DIRS - Include directories
#
# Also defines the library variables below as normal variables.
# These contain debug/optimized keywords when a debugging library is found.
#
#   ADNN_LIBRARIES - libaDNN
#
# Accepts the following variables as input:
#
#   MLOPEN_ROOT - (as a CMake or environment variable)
#                The root directory of the for all the AMD Machine Learning systems
#
#   FIND_LIBRARY_USE_LIB64_PATHS - Global property that controls whether findaDNN should search for
#                              64bit or 32bit libs
#-----------------------
# Example Usage:
#
#    find_package(ADNN REQUIRED)
#    include_directories(${ADNN_INCLUDE_DIRS})   >> OR MLOPEN_INCLUDE_DIRS << 
#
#    add_executable(foo foo.cc)
#    target_link_libraries(foo ${ADNN_LIBRARIES})
## OR
##   set(Caffe_LINKER_LIBS ADNN_LIBRARIES ${Caffe_LINKER_LIBS})

#
#-----------------------

## ON --> OFF
set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS ON)

message(STATUS "TT: /c/AMD/MLopen/caffe/cmake/Modules/FindaDNN.cmake - Looking for aDNN ... ") 

## Do we want aDNN/AMDnnl.h (that is the aDNN directory)

find_path(ADNN_INCLUDE_DIRS
    NAMES AMDnn.h AMDnn.hpp
    HINTS
        ${MLOPEN_ROOT}/incl/aDNN
	${ADNN_PATH}/inc
        $ENV{ADNN_PATH}/inc
        $ENV{MLOPEN_ROOT}/incl/aDNN
    PATHS
        /opt/aDNN/inc
        /c/AMD/MLopen/incl/
	/c/AMD/MLopen/incl/aDNN
    DOC "AMD DNN header file path"
)
mark_as_advanced( ADNN_INCLUDE_DIRS )

# Search for 64bit libs if FIND_LIBRARY_USE_LIB64_PATHS is set to true in the global environment, 32bit libs else
get_property( LIB64 GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS )
message(STATUS "TT: CMAKE_PREFIX_PATH" ${CMAKE_PREFIX_PATH})

if( LIB64 )
message(STATUS "TT: the PATH SUFFIXES may stop from finding the library - 64 - Commenting out") 
    find_library( ADNN_LIBRARIES
        NAMES aDNN
        HINTS
            ${MLOPEN_ROOT}/lib
	    ${ADNN_PATH}/lib    
            $ENV{MLOPEN_ROOT}/lib
            $ENV{ADNN_PATH}/lib
        DOC "aDNN static library path 64bit"
##        PATH_SUFFIXES x86_64 x64 
        PATHS
            /usr/lib
            /c/AMD/MLopen/libs
	    /c/AMD/MLopen/adnn/lib
            /opt/aDNN/lib
	NO_DEFAULT_PATH
    )
else( )
  message(STATUS "TT:  PATH SUFFIXES set to lib - 32")
  # changing this to not have the lib
    find_library( ADNN_LIBRARIES
        NAMES aDNN
        DOC "aDNN static library path 32bit"
	PATH_SUFFIXES lib
        PATHS
            /c/AMD/MLopen
	    /c/AMD/MLopen/addn
	NO_DEFAULT_PATH
    )
endif( )
mark_as_advanced( ADNN_LIBRARIES )

include( FindPackageHandleStandardArgs )
FIND_PACKAGE_HANDLE_STANDARD_ARGS( ADNN DEFAULT_MSG ADNN_LIBRARIES ADNN_INCLUDE_DIRS )

if( NOT ADNN_FOUND )
    message( STATUS "FindaDNN FAILED to find libraries named: aDNN" )
else ()
    message( STATUS "Found aDNN  (include: ${ADNN_INCLUDE_DIRS}, library: ${ADNN_LIBRARIES})")
endif()
