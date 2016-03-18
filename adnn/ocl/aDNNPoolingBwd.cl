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

#define ADNN_POOLBWD_GROUP_SZ2 1


#define ADNN_POOLBWD_LCL_DATA_WIDTH (ADNN_POOLBWD_GROUP_SZ0 *ADNN_POOLBWD_N_VERT_OUT_PIX + ADNN_POOLING_KERNEL_SZ + ADNN_POOLING_STRIDE - 1) / ADNN_POOLING_STRIDE
#define ADNN_POOLBWD_LCL_DATA_HEIGHT (ADNN_POOLBWD_GROUP_SZ1 *ADNN_POOLBWD_N_VERT_OUT_PIX  + ADNN_POOLING_KERNEL_SZ + ADNN_POOLING_STRIDE - 1) / ADNN_POOLING_STRIDE


__attribute__((reqd_work_group_size(ADNN_POOLBWD_GROUP_SZ0,ADNN_POOLBWD_GROUP_SZ1,ADNN_POOLBWD_GROUP_SZ2)))
__kernel void aDNNPoolingAveBwd(
       const __global _FLOAT * top_diff,
        __global _FLOAT * bot_diff
	   )
{
		__local _FLOAT lcl_top_diff[ADNN_POOLBWD_LCL_DATA_WIDTH * ADNN_POOLBWD_LCL_DATA_HEIGHT];

		int x = get_group_id(0) * ADNN_POOLBWD_GROUP_SZ0 * ADNN_POOLBWD_N_HORIZ_OUT_PIX;
		int y = get_group_id(1) * ADNN_POOLBWD_GROUP_SZ1 * ADNN_POOLBWD_N_VERT_OUT_PIX;
		int lcl_id0 = get_local_id(0);
		int lcl_id1 = get_local_id(1);
//		int lcl_id = (lcl_id1 << ADNN_POOLBWD_GROUP_LG2SZ1) + lcl_id0;
		int ob = get_global_id(2); // outputs * batch_sz
		int b = (int)(float)ob / (float)ADNN_POOLING_N_OUTPUTS;
		int o = ob - b * ADNN_POOLING_N_OUTPUTS;


		int top_x = (x + ADNN_POOLING_PAD - ADNN_POOLING_KERNEL_SZ) < 0 ? 0 : (x + ADNN_POOLING_PAD - ADNN_POOLING_KERNEL_SZ)/ ADNN_POOLING_STRIDE + 1;
		int top_y = (y + ADNN_POOLING_PAD - ADNN_POOLING_KERNEL_SZ) < 0 ? 0 : (y + ADNN_POOLING_PAD - ADNN_POOLING_KERNEL_SZ) / ADNN_POOLING_STRIDE + 1;
		int top_off = b * ADNN_POOLBWD_TOP_BATCH_STRIDE + o * ADNN_POOLBWD_TOP_CHANNEL_STRIDE;


		_FLOAT res[ADNN_POOLBWD_N_VERT_OUT_PIX][ADNN_POOLBWD_N_HORIZ_OUT_PIX];
		for( int k = 0; k < ADNN_POOLBWD_N_VERT_OUT_PIX; k++)
		{
			for(int l = 0; l < ADNN_POOLBWD_N_HORIZ_OUT_PIX; l++)
			{
				res[k][l] = 0;
			}
		}


// load tile
		for( int tj = lcl_id1; tj < ADNN_POOLBWD_LCL_DATA_HEIGHT; tj += ADNN_POOLBWD_GROUP_SZ1)
		{	
			int top_y_act = top_y + tj;
			int top_y_off = top_y_act * ADNN_POOLBWD_TOP_STRIDE;

			int lcl_off_v = tj * ADNN_POOLBWD_LCL_DATA_WIDTH;

			bool invisibleY = (top_y_act >= ADNN_POOLBWD_TOP_HEIGHT);

			for(int ti = lcl_id0; ti < ADNN_POOLBWD_LCL_DATA_WIDTH; ti += ADNN_POOLBWD_GROUP_SZ0)
			{

				int top_x_act = top_x + ti;
				
				bool invisibleX = (top_x_act >= ADNN_POOLBWD_TOP_WIDTH);

				_FLOAT top_val = top_diff[top_off + top_y_off + top_x_act];


				top_val = (invisibleX || invisibleY)? 0 : top_val;
								
				lcl_top_diff[lcl_off_v + ti] = top_val;
#if 0
				if (lcl_id1==0&&o==0&&b==0)
				{
				  printf("K:in: %d %d %d   %f\n", top_off + top_y_off + top_x_act, top_y_act, top_x_act, top_val);
				}
#endif
				
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		int bot_y = (y + lcl_id1 * ADNN_POOLBWD_N_VERT_OUT_PIX);
		int bot_x = (x + lcl_id0 * ADNN_POOLBWD_N_HORIZ_OUT_PIX);


		for( int k = 0; k < ADNN_POOLBWD_N_VERT_OUT_PIX; k++)
		{

			int h = bot_y + k + ADNN_POOLING_PAD;
			int top_hstart = (h < ADNN_POOLING_KERNEL_SZ) ? 0 : (h - ADNN_POOLING_KERNEL_SZ) / ADNN_POOLING_STRIDE + 1;
			int top_hend = min(h / ADNN_POOLING_STRIDE + 1, ADNN_POOLBWD_TOP_HEIGHT);

			for(int l = 0; l < ADNN_POOLBWD_N_HORIZ_OUT_PIX; l++)
			{


				int	w = bot_x + l + ADNN_POOLING_PAD;
				int top_wstart = (w < ADNN_POOLING_KERNEL_SZ) ? 0 : (w - ADNN_POOLING_KERNEL_SZ) / ADNN_POOLING_STRIDE + 1;
				int top_wend = min(w / ADNN_POOLING_STRIDE + 1, ADNN_POOLBWD_TOP_WIDTH);
		

				for (int top_h = top_hstart; top_h < top_hend; ++top_h)
				{
				    int hstart = top_h * ADNN_POOLING_STRIDE - ADNN_POOLING_PAD;
					int hend = min(hstart + ADNN_POOLING_KERNEL_SZ, ADNN_POOLBWD_BOT_HEIGHT + ADNN_POOLING_PAD);

					for (int top_w =top_wstart; top_w < top_wend; ++top_w)
					{
        // figure out the pooling size
						int wstart = top_w * ADNN_POOLING_STRIDE - ADNN_POOLING_PAD;
						int wend = min(wstart + ADNN_POOLING_KERNEL_SZ, ADNN_POOLBWD_BOT_WIDTH + ADNN_POOLING_PAD);
						int pool_size = (hend - hstart) * (wend - wstart);
						int lcl_top_h = top_h - top_y;
						int lcl_top_w = top_w - top_x;
						_FLOAT add_val = (lcl_top_diff[lcl_top_h *  ADNN_POOLBWD_LCL_DATA_WIDTH + lcl_top_w] / (_FLOAT)pool_size);
						res[k][l] += add_val;
#if 0
				if (bot_x+l==6&&bot_y+k==0&&o==3&&b==0)
				{
				  printf("K:com: %d %d %d %d %d %d   %10.8f %10.8f %10.8f %d\n", k,l,top_h, top_w, lcl_top_h, lcl_top_w, res[k][l], add_val, lcl_top_diff[lcl_top_h *  ADNN_POOLBWD_LCL_DATA_WIDTH + lcl_top_w], pool_size);
				}
#endif
					}
				}


			}
		}

		int bot_off = b * ADNN_POOLBWD_BOT_BATCH_STRIDE + o * ADNN_POOLBWD_BOT_CHANNEL_STRIDE + bot_y * ADNN_POOLBWD_BOT_STRIDE + bot_x;
		for( int k = 0; k < ADNN_POOLBWD_N_VERT_OUT_PIX; k++)
		{
			for(int l = 0; l < ADNN_POOLBWD_N_HORIZ_OUT_PIX; l++)
			{
				if (bot_y + k < ADNN_POOLBWD_BOT_HEIGHT && bot_x + l < ADNN_POOLBWD_BOT_WIDTH)
				{	
					bot_diff[bot_off + k * ADNN_POOLBWD_BOT_STRIDE +l] = res[k][l];
#if 0
					if (lcl_id0==0&&lcl_id1==0&&o==0&&b==0)
					{
						printf("K:out: %d %d %d  %f\n", bot_off + k * ADNN_POOLBWD_BOT_STRIDE +l, k, l, bot_diff[bot_off + k * ADNN_POOLBWD_BOT_STRIDE +l]);
					}
#endif

				}
			}
		}

}


#define ADNN_LCL_BOT_WIDTH (ADNN_POOLBWD_GROUP_SZ0 *ADNN_POOLBWD_N_VERT_OUT_PIX + ADNN_POOLING_KERNEL_SZ)
#define ADNN_LCL_BOT_HEIGHT (ADNN_POOLBWD_GROUP_SZ1 *ADNN_POOLBWD_N_VERT_OUT_PIX  + ADNN_POOLING_KERNEL_SZ)

__attribute__((reqd_work_group_size(ADNN_POOLBWD_GROUP_SZ0,ADNN_POOLBWD_GROUP_SZ1,ADNN_POOLBWD_GROUP_SZ2)))
__kernel void aDNNPoolingMaxBwd(
       const __global _FLOAT * top_df,
       __global _FLOAT * bot_df,
       const __global _FLOAT * top,
       const __global _FLOAT * bot
	   )
{
		__local _FLOAT lcl_top_df[ADNN_POOLBWD_LCL_DATA_WIDTH * ADNN_POOLBWD_LCL_DATA_HEIGHT];
		__local _FLOAT lcl_top[ADNN_POOLBWD_LCL_DATA_WIDTH * ADNN_POOLBWD_LCL_DATA_HEIGHT];
		__local _FLOAT lcl_bot[ADNN_LCL_BOT_WIDTH * ADNN_LCL_BOT_HEIGHT];

		int x = get_group_id(0) * ADNN_POOLBWD_GROUP_SZ0 * ADNN_POOLBWD_N_HORIZ_OUT_PIX;
		int y = get_group_id(1) * ADNN_POOLBWD_GROUP_SZ1 * ADNN_POOLBWD_N_VERT_OUT_PIX;
		int lcl_id0 = get_local_id(0);
		int lcl_id1 = get_local_id(1);
		int ob = get_global_id(2); // outputs * batch_sz
		int b = (int)(float)ob / (float)ADNN_POOLING_N_OUTPUTS;
		int o = ob - b * ADNN_POOLING_N_OUTPUTS;


		int top_x = (x - ADNN_POOLING_KERNEL_SZ) < 0 ? 0 : (x - ADNN_POOLING_KERNEL_SZ)/ ADNN_POOLING_STRIDE + 1;
		int top_y = (y - ADNN_POOLING_KERNEL_SZ) < 0 ? 0 : (y - ADNN_POOLING_KERNEL_SZ) / ADNN_POOLING_STRIDE + 1;
		int top_df_off = b * ADNN_POOLBWD_TOPDF_BATCH_STRIDE + o * ADNN_POOLBWD_TOPDF_CHANNEL_STRIDE;
		int top_off = b * ADNN_POOLBWD_TOP_BATCH_STRIDE + o * ADNN_POOLBWD_TOP_CHANNEL_STRIDE;
		int bot_off = b * ADNN_POOLBWD_BOT_BATCH_STRIDE + o * ADNN_POOLBWD_BOT_CHANNEL_STRIDE;
		int bot_x = x;
		int bot_y = y;

		_FLOAT res[ADNN_POOLBWD_N_VERT_OUT_PIX][ADNN_POOLBWD_N_HORIZ_OUT_PIX];



// load tiles
// top df and top
		for( int tj = lcl_id1; tj < ADNN_POOLBWD_LCL_DATA_HEIGHT; tj += ADNN_POOLBWD_GROUP_SZ1)
		{	
			int top_y_act = top_y + tj;
			int top_df_y_off = top_y_act * ADNN_POOLBWD_TOPDF_STRIDE;
			int top_y_off = top_y_act * ADNN_POOLBWD_TOP_STRIDE;

			int lcl_off_v = tj * ADNN_POOLBWD_LCL_DATA_WIDTH;

			bool invisibleY = (top_y_act >= ADNN_POOLBWD_TOP_HEIGHT);

			for(int ti = lcl_id0; ti < ADNN_POOLBWD_LCL_DATA_WIDTH; ti += ADNN_POOLBWD_GROUP_SZ0)
			{

				int top_x_act = top_x + ti;
				
				bool invisibleX = (top_x_act >= ADNN_POOLBWD_TOP_WIDTH);

				_FLOAT top_df_val = top_df[top_df_off + top_df_y_off + top_x_act];
				top_df_val = (invisibleX || invisibleY)? 0 : top_df_val;
				_FLOAT top_val = top[top_off + top_y_off + top_x_act];
				top_val = (invisibleX || invisibleY)? 0 : top_val;

								
				lcl_top_df[lcl_off_v + ti] = top_df_val;
				lcl_top[lcl_off_v + ti] = top_val;
				
			}
		}

		for( int tj = lcl_id1; tj < ADNN_LCL_BOT_HEIGHT; tj += ADNN_POOLBWD_GROUP_SZ1)
		{	
			int bot_y_act = bot_y + tj;
			int bot_y_off = bot_y_act * ADNN_POOLBWD_BOT_STRIDE;

			int lcl_off_v = tj * ADNN_LCL_BOT_WIDTH;

			bool invisibleY = (bot_y_act >= ADNN_POOLBWD_BOT_HEIGHT);

			for(int ti = lcl_id0; ti < ADNN_LCL_BOT_WIDTH; ti += ADNN_POOLBWD_GROUP_SZ0)
			{

				int bot_x_act = bot_x + ti;
				
				bool invisibleX = (bot_x_act >= ADNN_POOLBWD_BOT_WIDTH);

				_FLOAT bot_val = bot[bot_off + bot_y_off + bot_x_act];
				bot_val = (invisibleX || invisibleY)? 0 : bot_val;

								
				lcl_bot[lcl_off_v + ti] = bot_val;
				
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		int bt_y = (y + lcl_id1 * ADNN_POOLBWD_N_VERT_OUT_PIX);
		int bt_x = (x + lcl_id0 * ADNN_POOLBWD_N_HORIZ_OUT_PIX);

		int lcl_bt_y = (lcl_id1 * ADNN_POOLBWD_N_VERT_OUT_PIX);
		int lcl_bt_x = (lcl_id0 * ADNN_POOLBWD_N_HORIZ_OUT_PIX);


		for( int k = 0; k < ADNN_POOLBWD_N_VERT_OUT_PIX; k++)
		{

			int b_y = bt_y + k;
// top most top y that can be influenced by this bot y

			int tt_y = (b_y -  ADNN_POOLING_KERNEL_SZ + ADNN_POOLING_STRIDE)  / ADNN_POOLING_STRIDE;
			tt_y = (tt_y < 0 ) ? 0 : tt_y;

			for(int l = 0; l < ADNN_POOLBWD_N_HORIZ_OUT_PIX; l++)
			{
				int	b_x = bt_x + l;
// left most top x that can be influenced by this bot x
				int lt_x = (b_x -  ADNN_POOLING_KERNEL_SZ + ADNN_POOLING_STRIDE)  / ADNN_POOLING_STRIDE;
				lt_x = (lt_x < 0 ) ? 0 : lt_x;
				
// find and sum up all tops that have been influenced by particular bot
				res[k][l] = 0;
				for (int th = tt_y; th < tt_y + (ADNN_POOLING_KERNEL_SZ + ADNN_POOLING_STRIDE - 1)  / ADNN_POOLING_STRIDE; ++th)
				{
					int src_y = th * ADNN_POOLING_STRIDE;
					bool invisY = (b_y - src_y > ADNN_POOLING_KERNEL_SZ - 1 || b_y - src_y < 0);
					for (int tw = lt_x; tw < lt_x + (ADNN_POOLING_KERNEL_SZ + ADNN_POOLING_STRIDE - 1)  / ADNN_POOLING_STRIDE; ++tw)
					{
						int lcl_th = th - top_y;
						int lcl_tw = tw - top_x;

						int src_x = tw * ADNN_POOLING_STRIDE;
						bool invisX = (b_x - src_x > ADNN_POOLING_KERNEL_SZ - 1 || b_x - src_x < 0);
						_FLOAT add_val = lcl_top_df[lcl_th *  ADNN_POOLBWD_LCL_DATA_WIDTH + lcl_tw] 
						* (lcl_top[lcl_th *  ADNN_POOLBWD_LCL_DATA_WIDTH + lcl_tw] == lcl_bot[(lcl_bt_y +k) * ADNN_LCL_BOT_WIDTH + lcl_bt_x +l]);
						res[k][l] += ( invisY || invisX)? 0 : add_val;

#if 0
								if (b==0 && o ==5 && b_x == 17 && b_y == 0)
								{
									printf("K:max: %d %d   %13.11f  %13.11f  %13.11f  %13.11f %13.11f\n",
										tw, th,
										res[k][l],
										add_val, 
										lcl_top_df[lcl_th *  ADNN_POOLBWD_LCL_DATA_WIDTH + lcl_tw] ,
										lcl_top[lcl_th *  ADNN_POOLBWD_LCL_DATA_WIDTH + lcl_tw],
										lcl_bot[(lcl_bt_y +k) * ADNN_LCL_BOT_WIDTH + lcl_bt_x +l]
										);
								}
#endif
					}
				}


			}
		}

		int bot_df_off = b * ADNN_POOLBWD_BOTDF_BATCH_STRIDE + o * ADNN_POOLBWD_BOTDF_CHANNEL_STRIDE + bt_y * ADNN_POOLBWD_BOTDF_STRIDE + bt_x;
		for( int k = 0; k < ADNN_POOLBWD_N_VERT_OUT_PIX; k++)
		{
			for(int l = 0; l < ADNN_POOLBWD_N_HORIZ_OUT_PIX; l++)
			{
				if ((bt_y + k) < ADNN_POOLBWD_BOT_HEIGHT && (bt_x + l) < ADNN_POOLBWD_BOT_WIDTH)
				{	
					bot_df[bot_df_off + k * ADNN_POOLBWD_BOTDF_STRIDE +l] = res[k][l];
#if 0
								if (b==0 && o ==0 && bt_x+l == 2 && bt_y+k == 0)
								{
									printf("K:max: %d %d   %13.11f\n",
										k, l,
										res[k][l]
										);
								}
#endif
				}
			}
		}

}

