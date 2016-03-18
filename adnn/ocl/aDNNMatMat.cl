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

#define ADNN_MM_GROUP_SZ2 1

#define ADNN_MM_LCL_DATA_WIDTH (1<<ADNN_MM_READ_LG2) * ADNN_MM_HORIZ_READ_LOOP
// always square
#define ADNN_MM_OUT_COL_STRIPE_WIDTH  (ADNN_MM_GROUP_SZ0 * ADNN_MM_N_HORIZ_OUT_PIX)
#define ADNN_MM_OUT_ROW_STRIPE_WIDTH  (ADNN_MM_GROUP_SZ1 * ADNN_MM_N_VERT_OUT_PIX)
#define ADNN_MM_GROUP_SZ (ADNN_MM_GROUP_SZ2 * ADNN_MM_GROUP_SZ1 * ADNN_MM_GROUP_SZ0)



// works only with transposed B
__attribute__((reqd_work_group_size(ADNN_MM_GROUP_SZ0,ADNN_MM_GROUP_SZ1,ADNN_MM_GROUP_SZ2)))
__kernel void aDNN_MM_TP(
       const __global _FLOAT * mA,
       const __global _FLOAT * mB,
       __global _FLOAT * mC
	   )
{
		
		__local _FLOAT mB_data[ADNN_MM_LCL_DATA_WIDTH * ADNN_MM_OUT_COL_STRIPE_WIDTH];
		__local _FLOAT mA_data[ADNN_MM_LCL_DATA_WIDTH * ADNN_MM_OUT_ROW_STRIPE_WIDTH];
	
		_FLOAT mC_tile[ADNN_MM_N_VERT_OUT_PIX][ADNN_MM_N_HORIZ_OUT_PIX];

		int x_grp_out = get_group_id(0) * ADNN_MM_GROUP_SZ0 * ADNN_MM_N_HORIZ_OUT_PIX;
		int y_grp_out = get_group_id(1) * ADNN_MM_GROUP_SZ1 * ADNN_MM_N_VERT_OUT_PIX;


		int lcl_id0 = get_local_id(0);
		int lcl_id1 = get_local_id(1);
		int lcl_id = (lcl_id1 << ADNN_MM_GROUP_LG2SZ1) + lcl_id0;
		int lcl_read_id1 = (lcl_id >> ADNN_MM_READ_LG2);
		int lcl_read_id0 = lcl_id - (lcl_read_id1 << ADNN_MM_READ_LG2);


		int x_mA = lcl_read_id0;
//		int x_mB = lcl_read_id0;
		int y_mA = y_grp_out + lcl_read_id1;
		int y_mB = x_grp_out + lcl_read_id1;

// init
		for(int j = 0; j < ADNN_MM_N_VERT_OUT_PIX; ++j)
		{
			for(int i = 0; i < ADNN_MM_N_HORIZ_OUT_PIX; ++i)
			{
				mC_tile[j][i] = 0;
			}
		}

// outer loop over input longest stride - mA and (transformed)mB
// 
		for(int ol = 0; ol < ADNN_MM_OUTER_LOOP; ol++, x_mA+= ADNN_MM_LCL_DATA_WIDTH)
		{
// global and local y coord
			int y_mA_run = y_mA;
			int y_lcl_mA_run = lcl_read_id1;

			for(int rl = 0; rl < ADNN_MM_MA_VERT_READ_LOOP; rl++, y_mA_run += ADNN_MM_MA_VERT_READ_STEP, y_lcl_mA_run += ADNN_MM_MA_VERT_READ_STEP)
			{
// check out off the region condition
				bool invisibleY_mA = y_mA_run >= ADNN_MM_MA_HEIGHT;
// global and local x coord
				int x_mA_run = x_mA;
				int x_lcl_mA_run = lcl_read_id0;

				for(int rk = 0; rk < ADNN_MM_HORIZ_READ_LOOP; rk++, x_mA_run += (1<<ADNN_MM_READ_LG2), x_lcl_mA_run += (1<<ADNN_MM_READ_LG2))
				{
// out off reagion condition
					bool invisibleX_mA = x_mA_run >= ADNN_MM_MA_WIDTH;

// read global/write local
					int mA_off = y_mA_run * ADNN_MM_MA_STRIDE + x_mA_run;
					_FLOAT my_val = mA[mA_off];

					_FLOAT saved_val = (invisibleX_mA || invisibleY_mA )? 0 : my_val;
					int mA_lcl_off = y_lcl_mA_run * ADNN_MM_LCL_DATA_WIDTH + x_lcl_mA_run;
					mA_data[mA_lcl_off] = saved_val;
				}
			}

			int y_mB_run = y_mB;
			int y_lcl_mB_run = lcl_read_id1;
			for(int rl = 0; rl < ADNN_MM_MB_VERT_READ_LOOP; rl++, y_mB_run += ADNN_MM_MB_VERT_READ_STEP, y_lcl_mB_run += ADNN_MM_MB_VERT_READ_STEP)
			{
				bool invisibleY_mB = y_mB_run >= ADNN_MM_MB_HEIGHT;

				int x_mB_run = x_mA; // the same as A, since dimension is the same
				int x_lcl_mB_run = lcl_read_id0;

				for(int rk = 0; rk < ADNN_MM_HORIZ_READ_LOOP; rk++, x_mB_run += (1<<ADNN_MM_READ_LG2), x_lcl_mB_run += (1<<ADNN_MM_READ_LG2))
				{
					int mB_off = y_mB_run * ADNN_MM_MB_STRIDE + x_mB_run;
					_FLOAT my_val = mB[mB_off];
					bool invisibleX_mB = x_mB_run >= ADNN_MM_MB_WIDTH;
					_FLOAT saved_val = (invisibleX_mB || invisibleY_mB )? 0 : my_val;
					int mB_lcl_off = y_lcl_mB_run * ADNN_MM_LCL_DATA_WIDTH + x_lcl_mB_run;
					mB_data[mB_lcl_off] = saved_val;
				}
			}

			barrier(CLK_LOCAL_MEM_FENCE);

// caculate tile

			_FLOAT mA_prv_data[ADNN_MM_N_VERT_OUT_PIX][ADNN_MM_PRV_BUF];
			_FLOAT mB_prv_data[ADNN_MM_N_HORIZ_OUT_PIX][ADNN_MM_PRV_BUF];

// run over mA and ((transformed)mB local sub-row 
			for(int m = 0; m < ADNN_MM_ACCUM_LOOP; m++)
			{
// read chunk of the sub-row


				for(int l = 0; l < ADNN_MM_N_VERT_OUT_PIX; l++)
				{
					for(int k = 0; k < ADNN_MM_PRV_BUF; k++)
					{
						mA_prv_data[l][k] = mA_data[(lcl_id1 * ADNN_MM_N_VERT_OUT_PIX + l) * ADNN_MM_LCL_DATA_WIDTH + m*ADNN_MM_PRV_BUF + k];
					}
				}

				for(int l = 0; l < ADNN_MM_N_HORIZ_OUT_PIX; l++)
				{
					for(int k = 0; k < ADNN_MM_PRV_BUF; k++)
					{
						mB_prv_data[l][k] = mB_data[(lcl_id0 * ADNN_MM_N_HORIZ_OUT_PIX + l) * ADNN_MM_LCL_DATA_WIDTH + m*ADNN_MM_PRV_BUF + k];
					}
				}


// accumulate mat-mat for the -sub-row
				for(int j = 0; j < ADNN_MM_N_VERT_OUT_PIX; ++j)
				{
					for(int i = 0; i < ADNN_MM_N_HORIZ_OUT_PIX; ++i)
					{
						for(int k = 0; k < ADNN_MM_PRV_BUF; k++)
						{
#if 0
						    if ( get_global_id(0) == 8 && get_global_id(1) == 0 && j==0 && i == 0 && ol == 0)
							{
								printf("K:%d %d  %f %f %f\n", m, k, mC_tile[j][i], mA_prv_data[j][k], mB_prv_data[i][k]);
							}

#endif
							mC_tile[j][i] += mB_prv_data[i][k]*mA_prv_data[j][k];
						}
					}
				}


			}

		}

		// write out
		for(int k = 0; k < ADNN_MM_N_VERT_OUT_PIX; ++k)
		{
			int y_out = y_grp_out + lcl_id1*ADNN_MM_N_VERT_OUT_PIX + k;
			for(int l = 0; l < ADNN_MM_N_HORIZ_OUT_PIX; ++l)
			{
				int x_out = x_grp_out + lcl_id0*ADNN_MM_N_HORIZ_OUT_PIX +l;
				if (y_out < ADNN_MM_MC_HEIGHT && x_out < ADNN_MM_MC_WIDTH)
				{
					int mC_off = y_out * ADNN_MM_MC_STRIDE + x_out;
					mC[mC_off] = mC_tile[k][l];
				}
			}
		}
}

// works only with transposed B
__attribute__((reqd_work_group_size(ADNN_MM_GROUP_SZ0,ADNN_MM_GROUP_SZ1,ADNN_MM_GROUP_SZ2)))
__kernel void aDNN_FC(
       const __global _FLOAT * mA,    // bottom
       const __global _FLOAT * mB,    // weights
       const __global _FLOAT * Bias,  // bias
		__global _FLOAT * mC          // top
	   )
{
		
		__local _FLOAT mB_data[ADNN_MM_LCL_DATA_WIDTH * ADNN_MM_OUT_COL_STRIPE_WIDTH];
		__local _FLOAT mA_data[ADNN_MM_LCL_DATA_WIDTH * ADNN_MM_OUT_ROW_STRIPE_WIDTH];
	
		_FLOAT mC_tile[ADNN_MM_N_VERT_OUT_PIX][ADNN_MM_N_HORIZ_OUT_PIX];

		int x_grp_out = get_group_id(0) * ADNN_MM_GROUP_SZ0 * ADNN_MM_N_HORIZ_OUT_PIX;
		int y_grp_out = get_group_id(1) * ADNN_MM_GROUP_SZ1 * ADNN_MM_N_VERT_OUT_PIX;


		int lcl_id0 = get_local_id(0);
		int lcl_id1 = get_local_id(1);
		int lcl_id = (lcl_id1 << ADNN_MM_GROUP_LG2SZ1) + lcl_id0;
		int lcl_read_id1 = (lcl_id >> ADNN_MM_READ_LG2);
		int lcl_read_id0 = lcl_id - (lcl_read_id1 << ADNN_MM_READ_LG2);


		int x_mA = lcl_read_id0;
		int y_mA = y_grp_out + lcl_read_id1;
		int y_mB = x_grp_out + lcl_read_id1;

// init
		for(int j = 0; j < ADNN_MM_N_VERT_OUT_PIX; ++j)
		{
			for(int i = 0; i < ADNN_MM_N_HORIZ_OUT_PIX; ++i)
			{
				int x_bias = x_grp_out + lcl_id0*ADNN_MM_N_HORIZ_OUT_PIX +i;

				mC_tile[j][i] = Bias[x_bias];
			}
		}

// outer loop over input longest stride - mA and (transformed)mB
// 
		for(int ol = 0; ol < ADNN_MM_OUTER_LOOP; ol++, x_mA+= ADNN_MM_LCL_DATA_WIDTH)
		{
// global and local y coord
			int y_mA_run = y_mA;
			int y_lcl_mA_run = lcl_read_id1;

			for(int rl = 0; rl < ADNN_MM_MA_VERT_READ_LOOP; rl++, y_mA_run += ADNN_MM_MA_VERT_READ_STEP, y_lcl_mA_run += ADNN_MM_MA_VERT_READ_STEP)
			{
// check out off the region condition
				bool invisibleY_mA = y_mA_run >= ADNN_MM_MA_HEIGHT;
// global and local x coord
				int x_mA_run = x_mA;
				int x_lcl_mA_run = lcl_read_id0;

				for(int rk = 0; rk < ADNN_MM_HORIZ_READ_LOOP; rk++, x_mA_run += (1<<ADNN_MM_READ_LG2), x_lcl_mA_run += (1<<ADNN_MM_READ_LG2))
				{
// out off reagion condition
					bool invisibleX_mA = x_mA_run >= ADNN_MM_MA_WIDTH;

// read global/write local
					int mA_off = y_mA_run * ADNN_MM_MA_STRIDE + x_mA_run;
					_FLOAT my_val = mA[mA_off];

					_FLOAT saved_val = (invisibleX_mA || invisibleY_mA )? 0 : my_val;
					int mA_lcl_off = y_lcl_mA_run * ADNN_MM_LCL_DATA_WIDTH + x_lcl_mA_run;
					mA_data[mA_lcl_off] = saved_val;
				}
			}

			int y_mB_run = y_mB;
			int y_lcl_mB_run = lcl_read_id1;
			for(int rl = 0; rl < ADNN_MM_MB_VERT_READ_LOOP; rl++, y_mB_run += ADNN_MM_MB_VERT_READ_STEP, y_lcl_mB_run += ADNN_MM_MB_VERT_READ_STEP)
			{
				bool invisibleY_mB = y_mB_run >= ADNN_MM_MB_HEIGHT;

				int x_mB_run = x_mA; // the same as A, since dimension is the same
				int x_lcl_mB_run = lcl_read_id0;

				for(int rk = 0; rk < ADNN_MM_HORIZ_READ_LOOP; rk++, x_mB_run += (1<<ADNN_MM_READ_LG2), x_lcl_mB_run += (1<<ADNN_MM_READ_LG2))
				{
					int mB_off = y_mB_run * ADNN_MM_MB_STRIDE + x_mB_run;
					_FLOAT my_val = mB[mB_off];
					bool invisibleX_mB = x_mB_run >= ADNN_MM_MB_WIDTH;
					_FLOAT saved_val = (invisibleX_mB || invisibleY_mB )? 0 : my_val;
					int mB_lcl_off = y_lcl_mB_run * ADNN_MM_LCL_DATA_WIDTH + x_lcl_mB_run;
					mB_data[mB_lcl_off] = saved_val;
				}
			}

			barrier(CLK_LOCAL_MEM_FENCE);

// caculate tile

			_FLOAT mA_prv_data[ADNN_MM_N_VERT_OUT_PIX][ADNN_MM_PRV_BUF];
			_FLOAT mB_prv_data[ADNN_MM_N_HORIZ_OUT_PIX][ADNN_MM_PRV_BUF];

// run over mA and ((transformed)mB local sub-row 
			for(int m = 0; m < ADNN_MM_ACCUM_LOOP; m++)
			{
// read chunk of the sub-row


				for(int l = 0; l < ADNN_MM_N_VERT_OUT_PIX; l++)
				{
					for(int k = 0; k < ADNN_MM_PRV_BUF; k++)
					{
						mA_prv_data[l][k] = mA_data[(lcl_id1 * ADNN_MM_N_VERT_OUT_PIX + l) * ADNN_MM_LCL_DATA_WIDTH + m*ADNN_MM_PRV_BUF + k];
					}
				}

				for(int l = 0; l < ADNN_MM_N_HORIZ_OUT_PIX; l++)
				{
					for(int k = 0; k < ADNN_MM_PRV_BUF; k++)
					{
						mB_prv_data[l][k] = mB_data[(lcl_id0 * ADNN_MM_N_HORIZ_OUT_PIX + l) * ADNN_MM_LCL_DATA_WIDTH + m*ADNN_MM_PRV_BUF + k];
					}
				}


// accumulate mat-mat for the -sub-row
				for(int j = 0; j < ADNN_MM_N_VERT_OUT_PIX; ++j)
				{
					for(int i = 0; i < ADNN_MM_N_HORIZ_OUT_PIX; ++i)
					{
						for(int k = 0; k < ADNN_MM_PRV_BUF; k++)
						{
#if 0
						    if ( get_global_id(0) == 8 && get_global_id(1) == 0 && j==0 && i == 0 && ol == 0)
							{
								printf("K:%d %d  %f %f %f\n", m, k, mC_tile[j][i], mA_prv_data[j][k], mB_prv_data[i][k]);
							}

#endif
							mC_tile[j][i] += mB_prv_data[i][k]*mA_prv_data[j][k];
						}
					}
				}


			}

		}

		// write out
		for(int k = 0; k < ADNN_MM_N_VERT_OUT_PIX; ++k)
		{
			int y_out = y_grp_out + lcl_id1*ADNN_MM_N_VERT_OUT_PIX + k;
			for(int l = 0; l < ADNN_MM_N_HORIZ_OUT_PIX; ++l)
			{
				int x_out = x_grp_out + lcl_id0*ADNN_MM_N_HORIZ_OUT_PIX +l;
				if (y_out < ADNN_MM_MC_HEIGHT && x_out < ADNN_MM_MC_WIDTH)
				{
					int mC_off = y_out * ADNN_MM_MC_STRIDE + x_out;
					mC[mC_off] = mC_tile[k][l] + Bias[x_out] ;

#if 0
					printf("K:%d %d  %f\n", l, k, Bias[x_out]);
#endif
				}
			}
		}
}


