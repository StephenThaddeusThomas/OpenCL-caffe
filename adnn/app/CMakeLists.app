############################################################################################### 
## File:CMakeLists.app           	Application CODE BUILD
## From:CMakeLists.0.TOP+CMakeLists.2.aDNN-app
## Date:160214
## Path:/c/AMD/MLopen/cmake_files
## Link:/c/AMD/MLopen/aDNN/app/CMakeLists.txt
## Role:Application build and link with aDNN and clBLAS and perhaps OpenCL too
## Note:clBLAS_packages.notes
###############################################################################################

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


# Include Files and Libaries - 3 needed: aDNN, clBLAS, and AMD_APP_SDK #############
# also need the standard math library -lm

### TODO - do I have include_directores HERE or in .top ???? #####

# 1 -- [ aDNN - OpenML library by Alex] --
# This is the aDNN library that is build using CMakeLists.lib
# By specifing IMPORTED its not going to be built
## TODO - need to 'install' these files (in .lib) 
include_directories(SYSTEM ${COMMON_INCLUDE_DIR}/aDNN)
#add_library(aDNN STATIC IMPORTED)

# 2 -- [ clBLAS - vector math library ] --
# aDNN requires clBLAS library (modified from clSPARSE)
find_package(clBLAS REQUIRED)
include_directories(SYSTEM ${CLBLAS_INCLUDE_DIRS})
add_library(clBLAS STATIC IMPORTED GLOBAL )

# 3 -- [ AMD APP SDK ]--
# $AMDAPPSDKROOT\include
# $AMDAPPSDKSAMPLESROOT\include
include_directories(SYSTEM ${SDK_PROJECT_DIR}/include)
include_directories(SYSTEM ${SDK_SAMPLES_DIR}/include)

if(WIND32 OR MSVC_IDE)
  set(target_app adnndrvr.exe)
  set_property(GLOBAL PROPERTY USE_FOLDERS TRUE)
  add_executable(${target_app} aLibDNNDriver.cpp  stdafx.cpp )
else()
  ## other settings see AMD/clBLAS/src/CMakeLists.txt
  ## such as setting TARGET_PLATFORM and WIN32 -D_CRT_SECURE_NO_WARNINGS
  ## TODO : what is this...
  ## add_definitions(-std=c++0x) # Note: need to migrate to C++11 on new platform, my laptop C++0x 
  set(target_app adnndrvr.lnx)
  add_executable(${target_app} aLibDNNDriver.cpp )
endif()

# ---[ Application target ] -- 
## CLBLAS_LIBRARIES is defined in FindclBLAS.cmake 
target_link_libraries(${target_app} aDNN ${CLBLAS_LIBRARIES} ${MATH_LIBRARY} ) 

## caffe_default_properties(${the_target})
## caffe_set_runtime_directory(${the_target} "${PROJECT_BINARY_DIR}/test")

#end
