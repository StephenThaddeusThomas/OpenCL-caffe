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



#define ADNN_POOLING_GROUP_SZ2 1

#ifndef ADNN_POOLING_OP_ID
#define ADNN_POOLING_OP_ID 0
#endif
// max
#if ADNN_POOLING_OP_ID == ADNN_POOLING_OP_MAX
#define ADNN_POOLING_OP(A,B) max(A, B);
#elif ADNN_POOLING_OP_ID == ADNN_POOLING_OP_AVE
#define ADNN_POOLING_OP(A,B) (A + B);
#endif

#define ADNN_POOLING_LCL_DATA_WIDTH (ADNN_POOLING_GROUP_SZ0 *ADNN_POOLING_N_HORIZ_OUT_PIX *ADNN_POOLING_STRIDE + ADNN_POOLING_KERNEL_SZ - 1)
#define ADNN_POOLING_LCL_DATA_HEIGHT (ADNN_POOLING_GROUP_SZ1 *ADNN_POOLING_N_VERT_OUT_PIX *ADNN_POOLING_STRIDE + ADNN_POOLING_KERNEL_SZ - 1)



__attribute__((reqd_work_group_size(ADNN_POOLING_GROUP_SZ0,ADNN_POOLING_GROUP_SZ1,ADNN_POOLING_GROUP_SZ2)))
__kernel void aDNNPoolingAve(
       const __global _FLOAT * bot,
       __global _FLOAT * top
	   )
{
		__local _FLOAT bot_data[ADNN_POOLING_LCL_DATA_WIDTH * ADNN_POOLING_LCL_DATA_HEIGHT];

		int x = get_group_id(0) * ADNN_POOLING_GROUP_SZ0 * ADNN_POOLING_N_HORIZ_OUT_PIX;
		int y = get_group_id(1) * ADNN_POOLING_GROUP_SZ1 * ADNN_POOLING_N_VERT_OUT_PIX;
		int lcl_id0 = get_local_id(0);
		int lcl_id1 = get_local_id(1);
//		int lcl_id = (lcl_id1 << ADNN_POOLING_GROUP_LG2SZ0) + lcl_id0;
		int ob = get_global_id(2); // output * batch_sz
		int b = (int)(float)ob / (float)ADNN_POOLING_N_OUTPUTS;
		int o = ob - b * ADNN_POOLING_N_OUTPUTS;
		int bot_x = x*ADNN_POOLING_STRIDE;
		int bot_y = y*ADNN_POOLING_STRIDE;
		int bot_off = b * ADNN_POOLING_BOT_BATCH_STRIDE + o * ADNN_POOLING_BOT_CHANNEL_STRIDE;


		_FLOAT res[ADNN_POOLING_N_VERT_OUT_PIX][ADNN_POOLING_N_HORIZ_OUT_PIX];
		for( int k = 0; k < ADNN_POOLING_N_VERT_OUT_PIX; k++)
		{
			for(int l = 0; l < ADNN_POOLING_N_HORIZ_OUT_PIX; l++)
			{
				res[k][l] = 0;
			}
		}


// load tile
		for( int b_j = lcl_id1; b_j < ADNN_POOLING_LCL_DATA_HEIGHT; b_j += ADNN_POOLING_GROUP_SZ1)
		{	
			int bot_y_act = bot_y + b_j  - ADNN_POOLING_PAD;

			bool invisibleY = (bot_y_act < 0) || (bot_y_act >= ADNN_POOLING_BOT_HEIGHT);

			int bot_y_off = bot_y_act * ADNN_POOLING_BOT_STRIDE;

			int lcl_off_v = b_j * ADNN_POOLING_LCL_DATA_WIDTH;

			for(int b_i = lcl_id0; b_i < ADNN_POOLING_LCL_DATA_WIDTH; b_i += ADNN_POOLING_GROUP_SZ0)
			{

				int bot_x_act = bot_x + b_i - ADNN_POOLING_PAD;

				
				bool invisibleX = (bot_x_act < 0) || (bot_x_act >= ADNN_POOLING_BOT_WIDTH);

				_FLOAT bot_val = bot[bot_off + bot_y_off + bot_x_act];

				bot_val = (invisibleX || invisibleY) ? 0 : bot_val;
				bot_data[lcl_off_v + b_i] = bot_val;
				
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		int lcl_y = lcl_id1 * ADNN_POOLING_N_VERT_OUT_PIX * ADNN_POOLING_STRIDE;
		int lcl_x = lcl_id0 * ADNN_POOLING_N_HORIZ_OUT_PIX * ADNN_POOLING_STRIDE;
		int lcl_off = lcl_y * ADNN_POOLING_LCL_DATA_WIDTH + lcl_x;

		
		for( int k = 0; k < ADNN_POOLING_N_VERT_OUT_PIX; k++)
		{
			int y_dst = y+ lcl_id1 * ADNN_POOLING_N_VERT_OUT_PIX + k;
			int hstart = y_dst * ADNN_POOLING_STRIDE - ADNN_POOLING_PAD;
			int hend = min(hstart + ADNN_POOLING_KERNEL_SZ, ADNN_POOLING_BOT_HEIGHT + ADNN_POOLING_PAD);
			for(int l = 0; l < ADNN_POOLING_N_HORIZ_OUT_PIX; l++)
			{

				int	x_dst = x+ lcl_id0 * ADNN_POOLING_N_HORIZ_OUT_PIX + l;
				int wstart = x_dst * ADNN_POOLING_STRIDE - ADNN_POOLING_PAD;
				int wend = min(wstart + ADNN_POOLING_KERNEL_SZ, ADNN_POOLING_BOT_WIDTH + ADNN_POOLING_PAD);

				int pool_size = (hend - hstart) * (wend - wstart);

				for( int j = 0; j < ADNN_POOLING_KERNEL_SZ; j++)
				{
					for(int i = 0; i < ADNN_POOLING_KERNEL_SZ; i++)
					{

						_FLOAT bot_val =  bot_data[lcl_off + (k * ADNN_POOLING_STRIDE+j)*ADNN_POOLING_LCL_DATA_WIDTH + (l * ADNN_POOLING_STRIDE+i)];


						res[k][l] = ADNN_POOLING_OP(res[k][l],bot_val);

					}
				}


				res[k][l] *= (_FLOAT)1.f/ (_FLOAT)pool_size;
			}
		}

		int top_y = (y + lcl_id1 * ADNN_POOLING_N_VERT_OUT_PIX);
		int top_x = (x + lcl_id0 * ADNN_POOLING_N_HORIZ_OUT_PIX);
		int top_off = b * ADNN_POOLING_TOP_BATCH_STRIDE + o * ADNN_POOLING_TOP_CHANNEL_STRIDE + top_y * ADNN_POOLING_TOP_STRIDE + top_x;
		for( int k = 0; k < ADNN_POOLING_N_VERT_OUT_PIX; k++)
		{
			for(int l = 0; l < ADNN_POOLING_N_HORIZ_OUT_PIX; l++)
			{
				if (top_y + k < ADNN_POOLING_TOP_HEIGHT && top_x + l < ADNN_POOLING_TOP_WIDTH)
				{	
					top[top_off + k * ADNN_POOLING_TOP_STRIDE +l] = res[k][l];
				}
			}
		}

}



__attribute__((reqd_work_group_size(ADNN_POOLING_GROUP_SZ0,ADNN_POOLING_GROUP_SZ1,ADNN_POOLING_GROUP_SZ2)))
__kernel void aDNNPoolingMax(
       const __global _FLOAT * bot,
       __global _FLOAT * top,
	   __global uchar * max_indx
	   )
{
		__local _FLOAT bot_data[ADNN_POOLING_LCL_DATA_WIDTH * ADNN_POOLING_LCL_DATA_HEIGHT];

		int x = get_group_id(0) * ADNN_POOLING_GROUP_SZ0 * ADNN_POOLING_N_HORIZ_OUT_PIX;
		int y = get_group_id(1) * ADNN_POOLING_GROUP_SZ1 * ADNN_POOLING_N_VERT_OUT_PIX;
		int lcl_id0 = get_local_id(0);
		int lcl_id1 = get_local_id(1);
//		int lcl_id = (lcl_id1 << ADNN_POOLING_GROUP_LG2SZ0) + lcl_id0;
		int ob = get_global_id(2); // output * batch_sz
		int b = (int)(float)ob / (float)ADNN_POOLING_N_OUTPUTS;
		int o = ob - b * ADNN_POOLING_N_OUTPUTS;
		int bot_x = x*ADNN_POOLING_STRIDE;
		int bot_y = y*ADNN_POOLING_STRIDE;
		int bot_off = b * ADNN_POOLING_BOT_BATCH_STRIDE + o * ADNN_POOLING_BOT_CHANNEL_STRIDE;


		_FLOAT res[ADNN_POOLING_N_VERT_OUT_PIX][ADNN_POOLING_N_HORIZ_OUT_PIX];
		uchar max_indxs[ADNN_POOLING_N_VERT_OUT_PIX][ADNN_POOLING_N_HORIZ_OUT_PIX];
		for( int k = 0; k < ADNN_POOLING_N_VERT_OUT_PIX; k++)
		{
			for(int l = 0; l < ADNN_POOLING_N_HORIZ_OUT_PIX; l++)
			{
				res[k][l] =  -FLT_MAX;
			}
		}


// load tile
		for( int b_j = lcl_id1; b_j < ADNN_POOLING_LCL_DATA_HEIGHT; b_j += ADNN_POOLING_GROUP_SZ1)
		{	
			int bot_y_act = bot_y + b_j;

			
			bool invisibleY = (bot_y_act >= ADNN_POOLING_BOT_HEIGHT);

			int bot_y_off = bot_y_act * ADNN_POOLING_BOT_STRIDE;

			int lcl_off_v = b_j * ADNN_POOLING_LCL_DATA_WIDTH;

			for(int b_i = lcl_id0; b_i < ADNN_POOLING_LCL_DATA_WIDTH; b_i += ADNN_POOLING_GROUP_SZ0)
			{

				int bot_x_act = bot_x + b_i;

				bool invisibleX = (bot_x_act >= ADNN_POOLING_BOT_WIDTH);

				_FLOAT bot_val = bot[bot_off + bot_y_off + bot_x_act];

				bot_val = (invisibleX || invisibleY) ? -FLT_MAX : bot_val;
								
				bot_data[lcl_off_v + b_i] = bot_val;
				
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		int lcl_y = lcl_id1 * ADNN_POOLING_N_VERT_OUT_PIX * ADNN_POOLING_STRIDE;
		int lcl_x = lcl_id0 * ADNN_POOLING_N_HORIZ_OUT_PIX * ADNN_POOLING_STRIDE;
		int lcl_off = lcl_y * ADNN_POOLING_LCL_DATA_WIDTH + lcl_x;

		
		for( int k = 0; k < ADNN_POOLING_N_VERT_OUT_PIX; ++k)
		{
			for(int l = 0; l < ADNN_POOLING_N_HORIZ_OUT_PIX; ++l)
			{
				
				for( int j = 0, m_indx = 0; j < ADNN_POOLING_KERNEL_SZ; ++j)
				{
					for(int i = 0; i < ADNN_POOLING_KERNEL_SZ; ++i, ++m_indx)
					{

						_FLOAT bot_val =  bot_data[lcl_off + (k * ADNN_POOLING_STRIDE+j)*ADNN_POOLING_LCL_DATA_WIDTH + (l * ADNN_POOLING_STRIDE+i)];
						if (bot_val >  res[k][l])
						{
							res[k][l] = bot_val;
							max_indxs[k][l] = m_indx;
						}

					}
				}

			}
		}

		int top_y = (y + lcl_id1 * ADNN_POOLING_N_VERT_OUT_PIX);
		int top_x = (x + lcl_id0 * ADNN_POOLING_N_HORIZ_OUT_PIX);
		int top_off = b * ADNN_POOLING_TOP_BATCH_STRIDE + o * ADNN_POOLING_TOP_CHANNEL_STRIDE + top_y * ADNN_POOLING_TOP_STRIDE + top_x;
		int m_indx_off =  b * ADNN_POOLING_MINDX_BATCH_STRIDE + o * ADNN_POOLING_MINDX_CHANNEL_STRIDE + top_y * ADNN_POOLING_MINDX_STRIDE + top_x;
		for( int k = 0; k < ADNN_POOLING_N_VERT_OUT_PIX; k++)
		{
			for(int l = 0; l < ADNN_POOLING_N_HORIZ_OUT_PIX; l++)
			{
				if (top_y + k < ADNN_POOLING_TOP_HEIGHT && top_x + l < ADNN_POOLING_TOP_WIDTH)
				{	
					top[top_off + k * ADNN_POOLING_TOP_STRIDE +l] = res[k][l];
					max_indx[m_indx_off + k * ADNN_POOLING_MINDX_STRIDE +l] = max_indxs[k][l];
				}
			}
		}

}




#if 0
__kernel void aDNNPooling4_1_3x3__MAX(
       const __global _FLOAT * bottom,
       __global _FLOAT4 * top,
	   int step,
	   int bot_stride,
	   int bot_width,
	   int bot_height,
	   int bot_channel_stride,
	   int bot_batch_stride,
	   int top_width,
	   float f_top_width,
	   int top_height,
	   int top_stride,
	   int top_channel_stride,
	   int top_batch_stride
	   )
{
	
		int xy = get_global_id(0);
		int c = get_global_id(1); // channel
		int b = get_global_id(2); // batch
		_FLOAT4 out = 0;
		int y = (int)(float)xy/f_top_width;
		int x = xy - y*top_width;
		int bot_x = x*step * ADNN_POOLING_N_HORIZ_OUT_PIX;
		int bot_y = y*step;
		int bot_off = b* bot_batch_stride  + c* bot_channel_stride + bot_y * bot_stride + bot_x;

// NO BORDER YET!!!
		if ( bot_x + step * ADNN_POOLING_N_HORIZ_OUT_PIX + ADNN_POOLING_KERNEL_SZ - 1 < bot_width && bot_y + ADNN_POOLING_KERNEL_SZ < bot_height)
		{
			out.s0 = bottom[bot_off];
			for(int hk = 1; hk < ADNN_POOLING_KERNEL_SZ; hk++)
			{
				out.s0 = ADNN_POOLING_OP(out.s0, bottom[bot_off + hk]);
			}
			out.s1 = bottom[bot_off + step];
			for(int hk = 1; hk < ADNN_POOLING_KERNEL_SZ; hk++)
			{
				out.s1 = ADNN_POOLING_OP(out.s1, bottom[bot_off + hk]);
			}
			out.s2 = bottom[bot_off + step*2];
			for(int hk = 1; hk < ADNN_POOLING_KERNEL_SZ; hk++)
			{
				out.s2 = ADNN_POOLING_OP(out.s2, bottom[bot_off + hk]);
			}
			out.s3 = bottom[bot_off+ step*3];
			for(int hk = 1; hk < ADNN_POOLING_KERNEL_SZ; hk++)
			{
				out.s3 = ADNN_POOLING_OP(out.s3, bottom[bot_off + hk]);
			}

			for(int k = 1; k < ADNN_POOLING_KERNEL_SZ; k++)
			{
				bot_off += bot_stride;
				for(int hk = 0; hk < ADNN_POOLING_KERNEL_SZ; hk++)
				{
					out.s0 = ADNN_POOLING_OP(out.s0, bottom[bot_off + hk]);
				}
				for(int hk = 0; hk < ADNN_POOLING_KERNEL_SZ; hk++)
				{
					out.s1 = ADNN_POOLING_OP(out.s1, bottom[bot_off + step + hk]);
				}
				for(int hk = 0; hk < ADNN_POOLING_KERNEL_SZ; hk++)
				{
					out.s2 = ADNN_POOLING_OP(out.s2, bottom[bot_off + step*2 + hk]);
				}
				for(int hk = 0; hk < ADNN_POOLING_KERNEL_SZ; hk++)
				{
					out.s3 = ADNN_POOLING_OP(out.s3, bottom[bot_off + step*3 + hk]);
				}
			}
		}

		int out_off = b*(top_batch_stride>>2) + c*(top_channel_stride >>2) + y * (top_stride >>2) + x;
		top[out_off] = out;
	}

__kernel void aDNNPooling4_1_3x3_AVE(
       const __global _FLOAT * bottom,
       __global _FLOAT4 * top,
	   int step,
	   int bot_stride,
	   int bot_width,
	   int bot_height,
	   int bot_channel_stride,
	   int bot_batch_stride,
	   int top_width,
	   float f_top_width,
	   int top_height,
	   int top_stride,
	   int top_channel_stride,
	   int top_batch_stride
	   )
{
	
		int xy = get_global_id(0);
		int c = get_global_id(1); // channel
		int b = get_global_id(2); // batch
		_FLOAT4 out = 0;
		int y = (int)(float)xy/f_top_width;
		int x = xy - y*top_width;
		int bot_x = x*step * ADNN_POOLING_N_HORIZ_OUT_PIX;
		int bot_y = y*step;
		int bot_off = b* bot_batch_stride  + c* bot_channel_stride + bot_y * bot_stride + bot_x;

// NO BORDER YET!!!
		if ( bot_x + step * ADNN_POOLING_N_HORIZ_OUT_PIX + ADNN_POOLING_KERNEL_SZ - 1 < bot_width && bot_y + ADNN_POOLING_KERNEL_SZ < bot_height)
		{
			out.s0 = bottom[bot_off];
			for(int hk = 1; hk < ADNN_POOLING_KERNEL_SZ; hk++)
			{
				out.s0 = ADNN_POOLING_OP(out.s0, bottom[bot_off + hk]);
			}
			out.s1 = bottom[bot_off + step];
			for(int hk = 1; hk < ADNN_POOLING_KERNEL_SZ; hk++)
			{
				out.s1 = ADNN_POOLING_OP(out.s1, bottom[bot_off + hk]);
			}
			out.s2 = bottom[bot_off + step*2];
			for(int hk = 1; hk < ADNN_POOLING_KERNEL_SZ; hk++)
			{
				out.s2 = ADNN_POOLING_OP(out.s2, bottom[bot_off + hk]);
			}
			out.s3 = bottom[bot_off+ step*3];
			for(int hk = 1; hk < ADNN_POOLING_KERNEL_SZ; hk++)
			{
				out.s3 = ADNN_POOLING_OP(out.s3, bottom[bot_off + hk]);
			}

			for(int k = 1; k < ADNN_POOLING_KERNEL_SZ; k++)
			{
				bot_off += bot_stride;
				for(int hk = 0; hk < ADNN_POOLING_KERNEL_SZ; hk++)
				{
					out.s0 = ADNN_POOLING_OP(out.s0, bottom[bot_off + hk]);
				}
				for(int hk = 0; hk < ADNN_POOLING_KERNEL_SZ; hk++)
				{
					out.s1 = ADNN_POOLING_OP(out.s1, bottom[bot_off + step + hk]);
				}
				for(int hk = 0; hk < ADNN_POOLING_KERNEL_SZ; hk++)
				{
					out.s2 = ADNN_POOLING_OP(out.s2, bottom[bot_off + step*2 + hk]);
				}
				for(int hk = 0; hk < ADNN_POOLING_KERNEL_SZ; hk++)
				{
					out.s3 = ADNN_POOLING_OP(out.s3, bottom[bot_off + step*3 + hk]);
				}
			}
		}

		int out_off = b*(top_batch_stride>>2) + c*(top_channel_stride >>2) + y * (top_stride >>2) + x;
		top[out_off] = (out / (_FLOAT4) (ADNN_POOLING_KERNEL_SZ * ADNN_POOLING_KERNEL_SZ));
	}

#endif