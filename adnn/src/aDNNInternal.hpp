// File:aDNNInternal.hpp
// Path:/c/AMD/MLopen/adnn/src
//  160311 : Merged with Alex changes (repo/alex/apcLibs) 
/**********************************************************************
Copyright ?2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

?	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
?	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/


#ifndef ADNN_INTERNAL_H_
#define ADNN_INTERNAL_H_

//Header Files
#define NOMINMAX // stupid windows.h confused with min() macros in std namespace
#define _USE_MATH_DEFINES
#ifdef __APPLE__
 #include <mach/mach_time.h>  // for mach_absolute_time() and friends
 #include <OpenCL/opencl.h>
#else
 #include <CL/opencl.h>
#endif

#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <map>
#include <string>
#include <limits>
#include <algorithm>    // std::find  and std::min std::maxx

#ifdef WIN32
 #include <io.h>
 #include <windows.h>
 #include <BaseTsd.h>
 #define snprintf _snprintf
 #define vsnprintf _vsnprintf
 #define strcasecmp _stricmp
 #define strncasecmp _strnicmp
 typedef unsigned int uint;
 static double
 mach_absolute_time()   // Windows
    {
      double ret = 0;
      __int64 frec;
      __int64 clocks;
      QueryPerformanceFrequency((LARGE_INTEGER *)&frec);
      QueryPerformanceCounter((LARGE_INTEGER *)&clocks);
      ret = (double)clocks * 1000. / (double)frec;
      return(ret);
    }
#else  // !WIN32 so Linux and APPLE
 #include <unistd.h>
 #include <stdbool.h>
 #include <sys/time.h>
 #include <sys/resource.h>
 typedef  long long int __int64;

 // We want milliseconds. Following code was interpreted from timer.cpp
 static double
 mach_absolute_time()   // Linux 
    {
      double  d=0.0;
      timeval t; t.tv_sec=0;t.tv_usec=0;
      gettimeofday(&t,NULL);
      d=(t.tv_sec*1000.0)+t.tv_usec/1000;  // TT: was 1000000.0 
      return(d);
    }
#endif

#define __FLOAT__ float
typedef __FLOAT__ aDType;
typedef unsigned int uint;   // TT: dtype.h 

static double
subtractTimes(double endTime, double startTime)
{
  double difference = endTime - startTime;
  static double conversion = 0.0;

  if(conversion == 0.0)
    {
#if __APPLE__
      mach_timebase_info_data_t info;
      kern_return_t err = mach_timebase_info(&info);
      
      //Convert the timebase into seconds
      if (err == 0)
	conversion = 1e-9 * (double)info.numer / (double)info.denom;
#else
        conversion = 1.;
#endif
    }
  return conversion * (double)difference;
}

/* Include CLBLAS header. It automatically includes needed OpenCL header,
** so we can drop out explicit inclusion of cl.h header.
*/
#define WITH_CLBLAS
#include <clBLAS.h>

#include "aDNNOCL.hpp"

// SET TO COMPILE any SHARED CPU/GPU code ON CPU
#define ADNN_ACCEL 1 //_DNN_ACCEL_CPU

#include "aDNN.cl.h"
#include "AMDnn.h"
#include "AMDnn.hpp"
#include "aDNNTensor.hpp"

#endif
