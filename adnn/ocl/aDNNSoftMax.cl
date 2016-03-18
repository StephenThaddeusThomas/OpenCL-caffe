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

#include "aDNN.cl.h"

#define ADNN_SM_GROUP_SZ2 1

#define ADNN_SM_GROUP_SZ (ADNN_SM_GROUP_SZ2 * ADNN_SM_GROUP_SZ1 * ADNN_SM_GROUP_SZ0)

inline _FLOAT sum(_FLOAT a, _FLOAT b)
{
	return(a+b);
}

inline void SoftMax(__local _FLOAT *data, __local _FLOAT *tmp_data)
{
	
		int lcl_id0 = get_local_id(0);
		int pos_in = get_group_id(1) * ADNN_SM_IN_STRIDE;

		int pos_lcl = lcl_id0;


// fill empty spots with abs min
		for( int i = pos_lcl; i < ADNN_SM_LCL_DATA_LEN - ADNN_SM_IN_LEN; i+=ADNN_SM_GROUP_SZ0)
		{
			data[i + ADNN_SM_IN_LEN] = -FLT_MAX;
		}

		barrier(CLK_LOCAL_MEM_FENCE);


		{

// find max
// move to temp
			for(int i = lcl_id0; i < (1 << (ADNN_SM_LCL_DATA_LG2LEN - 1));  i+=ADNN_SM_GROUP_SZ0)
			{
				_FLOAT reduce_val = max(data[i], data[i + (1<<(ADNN_SM_LCL_DATA_LG2LEN - 1))]);
				tmp_data[i] = reduce_val;
			}

			barrier(CLK_LOCAL_MEM_FENCE);

			int j = ADNN_SM_LCL_DATA_LG2LEN - 2;
// reduce buffer larger than group size
			for( ; j >= ADNN_SM_GROUP_LG2SZ0; --j)
			{
				for(int i = lcl_id0; i < (1<<j); i+=ADNN_SM_GROUP_SZ0)
				{
					_FLOAT reduce_val = max(tmp_data[i], tmp_data[i + (1 << j)]);
					tmp_data[ i] = reduce_val;
				}

				barrier(CLK_LOCAL_MEM_FENCE);

			}
// reduce buffer not larger than group size

			for( ; j >= 0; --j)
			{
				if ( lcl_id0 < (1 << j))
				{
					_FLOAT reduce_val = max(tmp_data[lcl_id0], tmp_data[lcl_id0 + (1 << j)]);
					tmp_data[lcl_id0] = reduce_val;
				}
				barrier(CLK_LOCAL_MEM_FENCE);

			}

		}

		_FLOAT base = tmp_data[0];
		barrier(CLK_LOCAL_MEM_FENCE);

		for( int i = lcl_id0; i < ADNN_SM_LCL_DATA_LEN; i+=ADNN_SM_GROUP_SZ0)
		{
			_FLOAT sub_val = (data[i] - base);
			data[i] = (i < ADNN_SM_IN_LEN) ? exp(sub_val) : 0;
		}


		barrier(CLK_LOCAL_MEM_FENCE);

		{
// sum
// move to temp
#if 1
			for(int i = lcl_id0; i < (1 << (ADNN_SM_LCL_DATA_LG2LEN - 1));  i+=ADNN_SM_GROUP_SZ0)
			{
				_FLOAT reduce_val = sum(data[i], data[i + (1<<(ADNN_SM_LCL_DATA_LG2LEN - 1))]);
				tmp_data[i] = reduce_val;
			}

			barrier(CLK_LOCAL_MEM_FENCE);

			int j = ADNN_SM_LCL_DATA_LG2LEN - 2;
// reduce buffer larger than group size
			for( ; j >= ADNN_SM_GROUP_LG2SZ0; --j)
			{
				for(int i = lcl_id0; i < (1<<j); i+=ADNN_SM_GROUP_SZ0)
				{
					_FLOAT reduce_val = sum(tmp_data[i], tmp_data[i + (1 << j)]);
					tmp_data[ i] = reduce_val;
				}

				barrier(CLK_LOCAL_MEM_FENCE);

			}


// reduce buffer not larger than group size

			for( ; j >= 0; --j)
			{
				if ( lcl_id0 < (1 << j))
				{
					_FLOAT reduce_val = sum(tmp_data[lcl_id0], tmp_data[lcl_id0 + (1 << j)]);
					tmp_data[lcl_id0] = reduce_val;
				}

				barrier(CLK_LOCAL_MEM_FENCE);	

			}


#else
		if (lcl_id0 == 0 )
		{
			_FLOAT my_accum = 0;
			for(int i = 0; i < ADNN_SM_LCL_DATA_LEN /*ADNN_SM_IN_LEN*/; ++i)
			{
				my_accum += data[i];
			}
			tmp_data[0] = my_accum;

		}
		barrier(CLK_LOCAL_MEM_FENCE);

#endif

		}
//divide by sum (noramalize)
		_FLOAT scaler = 1.f/tmp_data[0];
		barrier(CLK_LOCAL_MEM_FENCE);


		for( int i = pos_lcl; i < ADNN_SM_LCL_DATA_LEN; i+=ADNN_SM_GROUP_SZ0)
		{
			data[ i] *= scaler;
		}
		barrier(CLK_LOCAL_MEM_FENCE);

}

// works only with transposed B
__attribute__((reqd_work_group_size(ADNN_SM_GROUP_SZ0,ADNN_SM_GROUP_SZ1,ADNN_SM_GROUP_SZ2)))
__kernel void aDNN_SM_withCrossEntropyLoss(
       const __global _FLOAT * in,
       const __global _FLOAT * labels,
       __global _FLOAT * out
	   )
{

		
	__local _FLOAT data[ADNN_SM_LCL_DATA_LEN];
	__local _FLOAT tmp_data[(ADNN_SM_LCL_DATA_LEN >> 1)];
	
	int lcl_id0 = get_local_id(0);
	int pos_in = get_group_id(1) * ADNN_SM_IN_STRIDE;

// load raw data
// 
	for(int i = lcl_id0; i < ADNN_SM_IN_LOOP * ADNN_SM_GROUP_SZ0; i += ADNN_SM_GROUP_SZ0)
	{
			data[i] = in[pos_in + i];

	}
	SoftMax(data, tmp_data);

// calc loss
	if (lcl_id0 == 0)
	{
		int index = (int)labels[get_group_id(1)];
		data[index] -= 1.f;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

// write out with norm
	int pos_out = get_group_id(1) * ADNN_SM_OUT_STRIDE;

	for(int i = lcl_id0; i < ADNN_SM_IN_LEN; i += ADNN_SM_GROUP_SZ0)
	{
		out[pos_out + i] = data[i] / (_FLOAT)get_num_groups(1);
	}

}






__attribute__((reqd_work_group_size(ADNN_SM_GROUP_SZ0,ADNN_SM_GROUP_SZ1,ADNN_SM_GROUP_SZ2)))
__kernel void aDNN_SM(
       const __global _FLOAT * in,
       __global _FLOAT * out
	   )
{
		
	__local _FLOAT data[ADNN_SM_LCL_DATA_LEN];
	__local _FLOAT tmp_data[(ADNN_SM_LCL_DATA_LEN >> 1)];
	
	int lcl_id0 = get_local_id(0);
	int pos_in = get_group_id(1) * ADNN_SM_IN_STRIDE;


// load raw data
// 
	for(int i = lcl_id0; i < ADNN_SM_IN_LOOP * ADNN_SM_GROUP_SZ0; i += ADNN_SM_GROUP_SZ0)
	{
		data[i] = in[pos_in + i];
	}



	SoftMax(data, tmp_data);

// write out
	int pos_out = get_group_id(1) * ADNN_SM_OUT_STRIDE;

	for(int i = lcl_id0; i < ADNN_SM_IN_LEN; i += ADNN_SM_GROUP_SZ0)
	{
		out[pos_out + i] = data[i];
	}

}
