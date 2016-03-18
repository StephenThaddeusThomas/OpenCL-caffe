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

#define _FLOAT					float
#define _FLOAT2					float2
#define _FLOAT4					float4
#define _FLOAT8					float8


#define ADNN_MT_GROUP_SZ2 1
#define ADNN_MT_WIDTH  (ADNN_MT_GROUP_SZ0 * ADNN_MT_N_VERT_OUT_PIX + 1)
#define ADNN_MT_HEIGHT  (ADNN_MT_GROUP_SZ1 * ADNN_MT_N_HORIZ_OUT_PIX)
#define ADNN_MT_GROUP_SZ (ADNN_MT_GROUP_SZ2 * ADNN_MT_GROUP_SZ1 * ADNN_MT_GROUP_SZ0)

// works only with transposed B
__attribute__((reqd_work_group_size(ADNN_MT_GROUP_SZ0,ADNN_MT_GROUP_SZ1,ADNN_MT_GROUP_SZ2)))
__kernel void aDNN_MatTrans(
       const __global _FLOAT * mA,
       __global _FLOAT * mB
	   )
{
#if 0
	__local _FLOAT lclTranspTile[ADNN_MT_HEIGHT][ADNN_MT_WIDTH];
	int x_mA = get_global_id(0) * ADNN_MT_N_VERT_OUT_PIX;
	int y_mA = get_global_id(1) * ADNN_MT_N_HORIZ_OUT_PIX;
	int x_lcl = get_local_id(0);
	int y_lcl = get_local_id(1);
	for(int j = get_local_id(1); j <  ADNN_MT_GROUP_SZ1*ADNN_MT_N_HORIZ_OUT_PIX; j+= ADNN_MT_GROUP_SZ1, y_mA += ADNN_MT_GROUP_SZ1)
	{
		for(int i = get_local_id(0); i < ADNN_MT_GROUP_SZ0*ADNN_MT_N_VERT_OUT_PIX; i += ADNN_MT_GROUP_SZ0, x_mA += ADNN_MT_GROUP_SZ0)
		{
			int mA_off = y_mA * ADNN_MT_MA_STRIDE + x_mA;
			lclTranspTile[j][i] = mA[mA_off];
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	int y_mB = get_global_id(0) * ADNN_MT_N_VERT_OUT_PIX;
	int x_mB = get_global_id(1) * ADNN_MT_N_HORIZ_OUT_PIX;

	y_lcl = get_local_id(0);
	x_lcl = get_local_id(1);

	for(int j = get_local_id(0); j < ADNN_MT_GROUP_SZ0*ADNN_MT_N_VERT_OUT_PIX; j += ADNN_MT_GROUP_SZ0, y_mB += ADNN_MT_GROUP_SZ0)
	{
		for(int i = get_local_id(1); i < ADNN_MT_GROUP_SZ1*ADNN_MT_N_HORIZ_OUT_PIX; i += ADNN_MT_GROUP_SZ1, x_mB += ADNN_MT_GROUP_SZ1)
		{
		//  transposed guards
		    if ( x_mB < ADNN_MT_MA_HEIGHT && y_mB < ADNN_MT_MA_WIDTH)
			{
				int mB_off = y_mB * ADNN_MT_MB_STRIDE + x_mB;
				mB[mB_off] = lclTranspTile[i][j];

#if 0
				if ( get_global_id(0) == 0 && get_global_id(1) == 0 )
				{
					printf("K:%d %d %d   %f %f\n", mB_off, i, j, mB[mB_off], lclTranspTile[j][i]);
				}

#endif
			}
		}
	}
#endif

}


#define ADNN_SR_GROUP_SZ1 1
#define ADNN_SR_GROUP_SZ2 1
#define ADNN_SR_GROUP_SZ (ADNN_SR_GROUP_SZ2 * ADNN_SR_GROUP_SZ1 * ADNN_SR_GROUP_SZ0)

//input: matrix, output: vector of row-wise sums
__attribute__((reqd_work_group_size(ADNN_SR_GROUP_SZ0,ADNN_SR_GROUP_SZ1,ADNN_SR_GROUP_SZ2)))
__kernel void aDNN_SumRow(
       const __global _FLOAT * mA,
       __global _FLOAT * vS
	   )
{
	__local _FLOAT row_sums[ADNN_SR_GROUP_SZ];
	int grp_id0 = get_group_id(0);
	int lcl_id0 = get_local_id(0);

	int gpr_off = grp_id0 * ADNN_SR_MA_STRIDE;

	_FLOAT row_sum = 0;
	for(int x = lcl_id0 + gpr_off, i = 0; i <  ADNN_SR_MA_ROW_LOOP; ++i, x += ADNN_SR_GROUP_SZ)
	{
		_FLOAT val = mA[x];
		val = (x < ADNN_SR_MA_WIDTH) ? val : 0;
		row_sum += val;
	}

	row_sums[lcl_id0] = row_sum;

	barrier(CLK_LOCAL_MEM_FENCE);

	for(int i = (ADNN_SR_GROUP_SZ >> 1); i > 0; i >>= 1)
	{
		if ( lcl_id0 < i)
		{
			row_sum += row_sums[ lcl_id0 + i];
			row_sums[ lcl_id0] = row_sum;
		}
	}

	if ( lcl_id0 == 0)
	{
		vS[grp_id0] = row_sum;
	}

}

