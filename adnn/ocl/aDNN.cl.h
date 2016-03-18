/*
 * Copyright (c) 2015 AMD Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 */
#ifndef _ADNN_CL_H_
#define _ADNN_CL_H_

#define _FLOAT					float
#define _FLOAT2					float2
#define _FLOAT4					float4
#define _FLOAT8					float8

#define ADNN_ACCEL_GPU              0
#define ADNN_ACCEL_CPU              1

#define ADNN_POOLING_OP_MAX			0
#define ADNN_POOLING_OP_AVE			1
#define ADNN_POOLING_OP_STC			2


#ifndef FLT_MAX
#define FLT_MAX         3.402823466e+38F        /* max value */
#endif


#if ADNN_ACCEL != ADNN_ACCEL_GPU
#if !defined(native_exp)
#define native_exp exp
#define native_log log
#endif
#else

//#define min fmin
//#define max fmax

#endif



#include "aDNNNeuron.cl.h"
#include "aDNNWeightsUpdate.cl.h"

#endif