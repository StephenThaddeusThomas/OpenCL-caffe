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
//#define _DNN_CONV_GROUP_SZ 64
//#define _DNN_CONV_GROUP_SZ0 8
//#define _DNN_CONV_GROUP_SZ1 8

//#define _DNN_CONV_GROUP_LG2SZ0 3
//#define _DNN_CONV_GROUP_LG2SZ1 3

#define _DNN_CONV_GROUP_SZ3 1
//#define _DNN_CONV_N_HORIZ_OUT_PIX 4
//#define _DNN_CONV_N_VERT_OUT_PIX 4
//#define _DNN_CONV_KERNEL_SZ 5
//#define _DNN_CONV_N_CHANNELS 3
//#define _DNN_CONV_BOTTOM_WIDTH	8
//#define _DNN_CONV_BOTTOM_HEIGHT 8

#define _DNN_CONV_GRP_DATA_WIDTH ((((_DNN_CONV_GROUP_SZ0 *_DNN_CONV_N_HORIZ_OUT_PIX + _DNN_CONV_KERNEL_SZ - 1) + 7) >> 2) << 2)
#define _DNN_CONV_GRP_DATA_HEIGHT  (_DNN_CONV_GROUP_SZ1 * _DNN_CONV_N_VERT_OUT_PIX + _DNN_CONV_KERNEL_SZ - 1)


__kernel void aDNNConv4(
       const __global _FLOAT * bottom,
       const __global _FLOAT * weights,
       __global _FLOAT4 * top,
	   float f_xy_groups_horiz,
	   int i_xy_groups_horiz,
	   int bot_width,
	   int bot_height,
	   int bot_stride,
	   int bot_channel_stride,
	   int bot_batch_stride,
	   int top_stride,
	   int top_channel_stride,
	   int top_batch_stride,
	   int n_channels,
	   int kernel_sz
	   )
{
		__local _FLOAT bottom_data[_DNN_CONV_GRP_DATA_WIDTH * _DNN_CONV_GRP_DATA_HEIGHT];
		__local _FLOAT weights_data[_DNN_CONV_KERNEL_SZ * _DNN_CONV_KERNEL_SZ];
	
		int lcl_id = get_local_id(0);
		int lcl_id1 = lcl_id >> _DNN_CONV_GROUP_LG2SZ0;
		int lcl_id0 = lcl_id - (lcl_id1 << _DNN_CONV_GROUP_LG2SZ0);	   	
		int xy = get_group_id(0); // channel position
		int o = get_global_id(1); // output
		int b = get_global_id(2); // batch 
		int pad = _DNN_CONV_PAD;
		int group_row = (int)((float)xy/f_xy_groups_horiz);
		int group_col = xy - group_row * i_xy_groups_horiz;
		int bot_group_x_s = group_col * _DNN_CONV_GROUP_SZ0 *_DNN_CONV_N_HORIZ_OUT_PIX - pad;
		int bot_group_y_s = group_row * _DNN_CONV_GROUP_SZ1 * _DNN_CONV_N_VERT_OUT_PIX - pad;
		int bot_group_x_e = bot_group_x_s + _DNN_CONV_GRP_DATA_WIDTH;
		int bot_group_y_e = bot_group_y_s + _DNN_CONV_GRP_DATA_HEIGHT;
		int top_group_x = group_col * _DNN_CONV_GROUP_SZ0; // *_DNN_CONV_N_HORIZ_OUT_PIX;
		int top_group_y = group_row * _DNN_CONV_GROUP_SZ1; 

		int weights_off = o * _DNN_CONV_KERNEL_SZ * _DNN_CONV_KERNEL_SZ * _DNN_CONV_N_CHANNELS;


		int lcl_vertical_off = 0;
		int lcl_horizontal_off = 0;
		int vertical_read = _DNN_CONV_GRP_DATA_HEIGHT;
		int horizontal_read = _DNN_CONV_GRP_DATA_WIDTH;
		int bottom_group_y = bot_group_y_s;
		int bottom_group_x = bot_group_x_s;
#if 1
		bool on_the_edge = (bot_group_y_s < 0 || bot_group_y_e > bot_height || bot_group_x_s < 0 || bot_group_x_e > bot_width);


		if ( on_the_edge )
		{
			lcl_vertical_off = (bot_group_y_s < 0) ? -bot_group_y_s : 0;
			vertical_read -= lcl_vertical_off;
			bottom_group_y = bot_group_y_s + lcl_vertical_off;
			vertical_read -= (bot_group_y_e > bot_height) ? bottom_group_y - bot_height : 0;
			lcl_horizontal_off = (bot_group_x_s < 0) ? -bot_group_x_s : 0;	 
			horizontal_read -= lcl_horizontal_off;
			bottom_group_x = bot_group_x_s + lcl_horizontal_off;
			horizontal_read -= (bot_group_x_e > bot_width) ? bot_group_x_e - bot_width : 0;
		}
#endif
		bottom_group_x &= ~2;
		horizontal_read = (((horizontal_read + 3) >> 2) << 2);

		_FLOAT4 out0 = 0, out1 = 0, out2 = 0, out3 = 0;

//#pragma unroll 1
		for( int c = 0; c < n_channels/*_DNN_CONV_N_CHANNELS*/; c++)
		{
// get weights per channel per ouput
#if 1
			for(int w = lcl_id; w < _DNN_CONV_KERNEL_SZ * _DNN_CONV_KERNEL_SZ; w += _DNN_CONV_GROUP_SZ)
			{
				weights_data[w] = weights[weights_off + c * _DNN_CONV_KERNEL_SZ * _DNN_CONV_KERNEL_SZ + w];
			}
#endif
// get data per channel
#if 1
			for(int j = lcl_id; on_the_edge && j < _DNN_CONV_GRP_DATA_WIDTH * _DNN_CONV_GRP_DATA_HEIGHT/2; j += _DNN_CONV_GROUP_SZ)
			{
					bottom_data[2*j +0] = 0;
					bottom_data[2*j +1] = 0;
			}

			barrier(CLK_LOCAL_MEM_FENCE);
#endif

			for( int b_j = lcl_id1; b_j < vertical_read; b_j += _DNN_CONV_GROUP_SZ1)
			{			
				for(int b_i = lcl_id0; b_i < horizontal_read; b_i += _DNN_CONV_GROUP_SZ0 * 4)
				{
					*(__local _FLOAT4*)&bottom_data[(lcl_vertical_off + b_j) * _DNN_CONV_GRP_DATA_WIDTH + lcl_horizontal_off + b_i] =
						*(__global _FLOAT4*)&bottom[bot_batch_stride * b + bot_channel_stride *c + bot_stride * (bottom_group_y + b_j) + bottom_group_x + b_i];
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE);
// compute per channel
			int bot_data_off = lcl_id1 * _DNN_CONV_N_VERT_OUT_PIX * _DNN_CONV_GRP_DATA_WIDTH + lcl_id0 * _DNN_CONV_N_HORIZ_OUT_PIX;
		
			_FLOAT4 bt0, bt1, bt2, bt3;


			for(int j = 0; j <  kernel_sz/*_DNN_CONV_KERNEL_SZ*/; j++)
			{

				bt0.s0 = bottom_data[bot_data_off + (j + 0) * _DNN_CONV_GRP_DATA_WIDTH + 0] ;
				bt0.s1 = bottom_data[bot_data_off + (j + 0) * _DNN_CONV_GRP_DATA_WIDTH + 1] ;
				bt0.s2 = bottom_data[bot_data_off + (j + 0) * _DNN_CONV_GRP_DATA_WIDTH + 2] ;
				bt0.s3 = bottom_data[bot_data_off + (j + 0) * _DNN_CONV_GRP_DATA_WIDTH + 3] ;

				bt1.s0 = bottom_data[bot_data_off + (j + 1) * _DNN_CONV_GRP_DATA_WIDTH + 0] ;
				bt1.s1 = bottom_data[bot_data_off + (j + 1) * _DNN_CONV_GRP_DATA_WIDTH + 1] ;
				bt1.s2 = bottom_data[bot_data_off + (j + 1) * _DNN_CONV_GRP_DATA_WIDTH + 2] ;
				bt1.s3 = bottom_data[bot_data_off + (j + 1) * _DNN_CONV_GRP_DATA_WIDTH + 3] ;

				bt2.s0 = bottom_data[bot_data_off + (j + 2) * _DNN_CONV_GRP_DATA_WIDTH + 0] ;
				bt2.s1 = bottom_data[bot_data_off + (j + 2) * _DNN_CONV_GRP_DATA_WIDTH + 1] ;
				bt2.s2 = bottom_data[bot_data_off + (j + 2) * _DNN_CONV_GRP_DATA_WIDTH + 2] ;
				bt2.s3 = bottom_data[bot_data_off + (j + 2) * _DNN_CONV_GRP_DATA_WIDTH + 3] ;

				bt3.s0 = bottom_data[bot_data_off + (j + 3) * _DNN_CONV_GRP_DATA_WIDTH + 0] ;
				bt3.s1 = bottom_data[bot_data_off + (j + 3) * _DNN_CONV_GRP_DATA_WIDTH + 1] ;
				bt3.s2 = bottom_data[bot_data_off + (j + 3) * _DNN_CONV_GRP_DATA_WIDTH + 2] ;
				bt3.s3 = bottom_data[bot_data_off + (j + 3) * _DNN_CONV_GRP_DATA_WIDTH + 3] ;

				for(int i = 0; i <  _DNN_CONV_KERNEL_SZ; i++)
				{
					_FLOAT4 wi4 = (_FLOAT4)weights_data[ _DNN_CONV_KERNEL_SZ * j + i];

					out0 += bt0 * wi4;
					out1 += bt1 * wi4;
					out2 += bt2 * wi4;
					out3 += bt3 * wi4;

					bt0.s0 = bt0.s1;
					bt0.s1 = bt0.s2;
					bt0.s2 = bt0.s3;
					bt0.s3 = bottom_data[bot_data_off + (j + 0) * _DNN_CONV_GRP_DATA_WIDTH + i + 4];

					bt1.s0 = bt1.s1;
					bt1.s1 = bt1.s2;
					bt1.s2 = bt1.s3;
					bt1.s3 = bottom_data[bot_data_off + (j + 1) * _DNN_CONV_GRP_DATA_WIDTH + i + 4];

					bt2.s0 = bt2.s1;
					bt2.s1 = bt2.s2;
					bt2.s2 = bt2.s3;
					bt2.s3 = bottom_data[bot_data_off + (j + 2) * _DNN_CONV_GRP_DATA_WIDTH + i + 4];

					bt3.s0 = bt3.s1;
					bt3.s1 = bt3.s2;
					bt3.s2 = bt3.s3;
					bt3.s3 = bottom_data[bot_data_off + (j + 3) * _DNN_CONV_GRP_DATA_WIDTH + i + 4];

				}
			}

		}

		int top_off = b * (top_batch_stride>>2) + o * (top_channel_stride >> 2) + (top_group_y  + lcl_id1) * _DNN_CONV_N_VERT_OUT_PIX  * (top_stride>>2) + top_group_x + lcl_id0; //X is per 4
		top[top_off] = out0;
		top[top_off + (top_stride>>2)] = out1;
		top[top_off + (top_stride>>2) *2] = out2;
		top[top_off + (top_stride>>2) *3] = out3;
}


__kernel void aDNNConv4_5x5(
       const __global _FLOAT * bottom,
       const __global _FLOAT * weights,
       __global _FLOAT4 * top,
	   float f_xy_groups_horiz,
	   int i_xy_groups_horiz,
	   int bot_width,
	   int bot_height,
	   int bot_stride,
	   int bot_channel_stride,
	   int bot_batch_stride,
	   int top_stride,
	   int top_channel_stride,
	   int top_batch_stride,
	   int n_channels,
	   int kernel_sz
	   )
{
		__local _FLOAT bottom_data[_DNN_CONV_GRP_DATA_WIDTH * _DNN_CONV_GRP_DATA_HEIGHT];
		__local _FLOAT weights_data[_DNN_CONV_KERNEL_SZ * _DNN_CONV_KERNEL_SZ];
	
		int lcl_id = get_local_id(0);
		int lcl_id1 = lcl_id >> _DNN_CONV_GROUP_LG2SZ0;
		int lcl_id0 = lcl_id - (lcl_id1 << _DNN_CONV_GROUP_LG2SZ0);	   	
		int xy = get_group_id(0); // channel position
		int o = get_global_id(1); // output
		int b = get_global_id(2); // batch 
		int pad = _DNN_CONV_PAD;
		int group_row = (int)((float)xy/f_xy_groups_horiz);
		int group_col = xy - group_row * i_xy_groups_horiz;
		int bot_group_x_s = group_col * _DNN_CONV_GROUP_SZ0 *_DNN_CONV_N_HORIZ_OUT_PIX - pad;
		int bot_group_y_s = group_row * _DNN_CONV_GROUP_SZ1 * _DNN_CONV_N_VERT_OUT_PIX - pad;
		int bot_group_x_e = bot_group_x_s + _DNN_CONV_GRP_DATA_WIDTH;
		int bot_group_y_e = bot_group_y_s + _DNN_CONV_GRP_DATA_HEIGHT;
		int top_group_x = group_col * _DNN_CONV_GROUP_SZ0; // *_DNN_CONV_N_HORIZ_OUT_PIX;
		int top_group_y = group_row * _DNN_CONV_GROUP_SZ1; 

		int weights_off = o * _DNN_CONV_KERNEL_SZ * _DNN_CONV_KERNEL_SZ * _DNN_CONV_N_CHANNELS;


		int lcl_vertical_off = 0;
		int lcl_horizontal_off = 0;
		int vertical_read = _DNN_CONV_GRP_DATA_HEIGHT;
		int horizontal_read = _DNN_CONV_GRP_DATA_WIDTH;
		int bottom_group_y = bot_group_y_s;
		int bottom_group_x = bot_group_x_s;
#if 1
		bool on_the_edge = (bot_group_y_s < 0 || bot_group_y_e > bot_height || bot_group_x_s < 0 || bot_group_x_e > bot_width);


		if ( on_the_edge )
		{
			lcl_vertical_off = (bot_group_y_s < 0) ? -bot_group_y_s : 0;
			vertical_read -= lcl_vertical_off;
			bottom_group_y = bot_group_y_s + lcl_vertical_off;
			vertical_read -= (bot_group_y_e > bot_height) ? bottom_group_y - bot_height : 0;
			lcl_horizontal_off = (bot_group_x_s < 0) ? -bot_group_x_s : 0;	 
			horizontal_read -= lcl_horizontal_off;
			bottom_group_x = bot_group_x_s + lcl_horizontal_off;
			horizontal_read -= (bot_group_x_e > bot_width) ? bot_group_x_e - bot_width : 0;
		}
#endif
		bottom_group_x &= ~2;
		horizontal_read = (((horizontal_read + 3) >> 2) << 2);

			 
		_FLOAT4 out0 = 0, out1 = 0, out2 = 0, out3 = 0;

		

//#pragma unroll 1
		for( int c = 0; c < n_channels/*_DNN_CONV_N_CHANNELS*/; c++)
		{
// get weights per channel per ouput
#if 1
			for(int w = lcl_id; w < _DNN_CONV_KERNEL_SZ * _DNN_CONV_KERNEL_SZ; w += _DNN_CONV_GROUP_SZ)
			{
				weights_data[w] = weights[weights_off + c * _DNN_CONV_KERNEL_SZ * _DNN_CONV_KERNEL_SZ + w];
			}
#endif
// get data per channel
			for(int j = lcl_id; on_the_edge && j < _DNN_CONV_GRP_DATA_WIDTH * _DNN_CONV_GRP_DATA_HEIGHT/2; j += _DNN_CONV_GROUP_SZ)
			{
					bottom_data[2*j +0] = 0;
					bottom_data[2*j +1] = 0;
			}

			barrier(CLK_LOCAL_MEM_FENCE);

#if 1
			for( int b_j = lcl_id1; b_j < vertical_read; b_j += _DNN_CONV_GROUP_SZ1)
			{			
				for(int b_i = lcl_id0; b_i < horizontal_read; b_i += _DNN_CONV_GROUP_SZ0 * 4)
				{
					*(__local _FLOAT4*)&bottom_data[(lcl_vertical_off + b_j) * _DNN_CONV_GRP_DATA_WIDTH + lcl_horizontal_off + b_i] =
						*(__global _FLOAT4*)&bottom[bot_batch_stride * b + bot_channel_stride *c + bot_stride * (bottom_group_y + b_j) + bottom_group_x + b_i];
				}
			}
#endif
			barrier(CLK_LOCAL_MEM_FENCE);
// compute per channel
			int bot_data_off = lcl_id1 * _DNN_CONV_N_VERT_OUT_PIX * _DNN_CONV_GRP_DATA_WIDTH + lcl_id0 * _DNN_CONV_N_HORIZ_OUT_PIX;
			_FLOAT4 w4;
			_FLOAT w5th;
			_FLOAT8 bt0, bt1, bt2, bt3;
// 0th filter row
			w4 = *(__local _FLOAT4*)&weights_data[0];
			w5th = weights_data[4];
// 0th data
			bt0 = *(__local _FLOAT8*)&bottom_data[bot_data_off];

			out0.s0 += bt0.s0 * w4.s0 + bt0.s1 * w4.s1 + bt0.s2 * w4.s2 + bt0.s3 * w4.s3 + bt0.s4 * w5th;
			out0.s1 += bt0.s1 * w4.s0 + bt0.s2 * w4.s1 + bt0.s3 * w4.s2 + bt0.s4 * w4.s3 + bt0.s5 * w5th;
			out0.s2 += bt0.s2 * w4.s0 + bt0.s3 * w4.s1 + bt0.s4 * w4.s2 + bt0.s5 * w4.s3 + bt0.s6 * w5th;
			out0.s3 += bt0.s3 * w4.s0 + bt0.s4 * w4.s1 + bt0.s5 * w4.s2 + bt0.s6 * w4.s3 + bt0.s7 * w5th;
// 1st data
			bt1 = *(__local _FLOAT8*)&bottom_data[bot_data_off + _DNN_CONV_GRP_DATA_WIDTH];
			out1.s0 += bt1.s0 * w4.s0 + bt1.s1 * w4.s1 + bt1.s2 * w4.s2 + bt1.s3 * w4.s3 + bt1.s4 * w5th;
			out1.s1 += bt1.s1 * w4.s0 + bt1.s2 * w4.s1 + bt1.s3 * w4.s2 + bt1.s4 * w4.s3 + bt1.s5 * w5th;
			out1.s2 += bt1.s2 * w4.s0 + bt1.s3 * w4.s1 + bt1.s4 * w4.s2 + bt1.s5 * w4.s3 + bt1.s6 * w5th;
			out1.s3 += bt1.s3 * w4.s0 + bt1.s4 * w4.s1 + bt1.s5 * w4.s2 + bt1.s6 * w4.s3 + bt1.s7 * w5th;

// 2nd
			bt2 = *(__local _FLOAT8*)&bottom_data[bot_data_off + _DNN_CONV_GRP_DATA_WIDTH * 2];

			out2.s0 += bt2.s0 * w4.s0 + bt2.s1 * w4.s1 + bt2.s2 * w4.s2 + bt2.s3 * w4.s3 + bt2.s4 * w5th;
			out2.s1 += bt2.s1 * w4.s0 + bt2.s2 * w4.s1 + bt2.s3 * w4.s2 + bt2.s4 * w4.s3 + bt2.s5 * w5th;
			out2.s2 += bt2.s2 * w4.s0 + bt2.s3 * w4.s1 + bt2.s4 * w4.s2 + bt2.s5 * w4.s3 + bt2.s6 * w5th;
			out2.s3 += bt2.s3 * w4.s0 + bt2.s4 * w4.s1 + bt2.s5 * w4.s2 + bt2.s6 * w4.s3 + bt2.s7 * w5th;

// 3d
			bt3 = *(__local _FLOAT8*)&bottom_data[bot_data_off + _DNN_CONV_GRP_DATA_WIDTH * 3];

			out3.s0 += bt3.s0 * w4.s0 + bt3.s1 * w4.s1 + bt3.s2 * w4.s2 + bt3.s3 * w4.s3 + bt3.s4 * w5th;
			out3.s1 += bt3.s1 * w4.s0 + bt3.s2 * w4.s1 + bt3.s3 * w4.s2 + bt3.s4 * w4.s3 + bt3.s5 * w5th;
			out3.s2 += bt3.s2 * w4.s0 + bt3.s3 * w4.s1 + bt3.s4 * w4.s2 + bt3.s5 * w4.s3 + bt3.s6 * w5th;
			out3.s3 += bt3.s3 * w4.s0 + bt3.s4 * w4.s1 + bt3.s5 * w4.s2 + bt3.s6 * w4.s3 + bt3.s7 * w5th;


// 1nd filter row
			w4 = *(__local _FLOAT4*)&weights_data[_DNN_CONV_KERNEL_SZ];
			w5th = weights_data[_DNN_CONV_KERNEL_SZ + 4];

			out0.s0 += bt1.s0 * w4.s0 + bt1.s1 * w4.s1 + bt1.s2 * w4.s2 + bt1.s3 * w4.s3 + bt1.s4 * w5th;
			out0.s1 += bt1.s1 * w4.s0 + bt1.s2 * w4.s1 + bt1.s3 * w4.s2 + bt1.s4 * w4.s3 + bt1.s5 * w5th;
			out0.s2 += bt1.s2 * w4.s0 + bt1.s3 * w4.s1 + bt1.s4 * w4.s2 + bt1.s5 * w4.s3 + bt1.s6 * w5th;
			out0.s3 += bt1.s3 * w4.s0 + bt1.s4 * w4.s1 + bt1.s5 * w4.s2 + bt1.s6 * w4.s3 + bt1.s7 * w5th;

			out1.s0 += bt2.s0 * w4.s0 + bt2.s1 * w4.s1 + bt2.s2 * w4.s2 + bt2.s3 * w4.s3 + bt2.s4 * w5th;
			out1.s1 += bt2.s1 * w4.s0 + bt2.s2 * w4.s1 + bt2.s3 * w4.s2 + bt2.s4 * w4.s3 + bt2.s5 * w5th;
			out1.s2 += bt2.s2 * w4.s0 + bt2.s3 * w4.s1 + bt2.s4 * w4.s2 + bt2.s5 * w4.s3 + bt2.s6 * w5th;
			out1.s3 += bt2.s3 * w4.s0 + bt2.s4 * w4.s1 + bt2.s5 * w4.s2 + bt2.s6 * w4.s3 + bt2.s7 * w5th;


			out2.s0 += bt3.s0 * w4.s0 + bt3.s1 * w4.s1 + bt3.s2 * w4.s2 + bt3.s3 * w4.s3 + bt3.s4 * w5th;
			out2.s1 += bt3.s1 * w4.s0 + bt3.s2 * w4.s1 + bt3.s3 * w4.s2 + bt3.s4 * w4.s3 + bt3.s5 * w5th;
			out2.s2 += bt3.s2 * w4.s0 + bt3.s3 * w4.s1 + bt3.s4 * w4.s2 + bt3.s5 * w4.s3 + bt3.s6 * w5th;
			out2.s3 += bt3.s3 * w4.s0 + bt3.s4 * w4.s1 + bt3.s5 * w4.s2 + bt3.s6 * w4.s3 + bt3.s7 * w5th;

// 4th data 

			bt0 = *(__local _FLOAT8*)&bottom_data[bot_data_off + _DNN_CONV_GRP_DATA_WIDTH * 4];

			out3.s0 += bt0.s0 * w4.s0 + bt0.s1 * w4.s1 + bt0.s2 * w4.s2 + bt0.s3 * w4.s3 + bt0.s4 * w5th;
			out3.s1 += bt0.s1 * w4.s0 + bt0.s2 * w4.s1 + bt0.s3 * w4.s2 + bt0.s4 * w4.s3 + bt0.s5 * w5th;
			out3.s2 += bt0.s2 * w4.s0 + bt0.s3 * w4.s1 + bt0.s4 * w4.s2 + bt0.s5 * w4.s3 + bt0.s6 * w5th;
			out3.s3 += bt0.s3 * w4.s0 + bt0.s4 * w4.s1 + bt0.s5 * w4.s2 + bt0.s6 * w4.s3 + bt0.s7 * w5th;

// 2nd filter row
			w4 = *(__local _FLOAT4*)&weights_data[_DNN_CONV_KERNEL_SZ * 2];
			w5th = weights_data[_DNN_CONV_KERNEL_SZ*2 + 4];

			out0.s0 += bt2.s0 * w4.s0 + bt2.s1 * w4.s1 + bt2.s2 * w4.s2 + bt2.s3 * w4.s3 + bt2.s4 * w5th;
			out0.s1 += bt2.s1 * w4.s0 + bt2.s2 * w4.s1 + bt2.s3 * w4.s2 + bt2.s4 * w4.s3 + bt2.s5 * w5th;
			out0.s2 += bt2.s2 * w4.s0 + bt2.s3 * w4.s1 + bt2.s4 * w4.s2 + bt2.s5 * w4.s3 + bt2.s6 * w5th;
			out0.s3 += bt2.s3 * w4.s0 + bt2.s4 * w4.s1 + bt2.s5 * w4.s2 + bt2.s6 * w4.s3 + bt2.s7 * w5th;

			out1.s0 += bt3.s0 * w4.s0 + bt3.s1 * w4.s1 + bt3.s2 * w4.s2 + bt3.s3 * w4.s3 + bt3.s4 * w5th;
			out1.s1 += bt3.s1 * w4.s0 + bt3.s2 * w4.s1 + bt3.s3 * w4.s2 + bt3.s4 * w4.s3 + bt3.s5 * w5th;
			out1.s2 += bt3.s2 * w4.s0 + bt3.s3 * w4.s1 + bt3.s4 * w4.s2 + bt3.s5 * w4.s3 + bt3.s6 * w5th;
			out1.s3 += bt3.s3 * w4.s0 + bt3.s4 * w4.s1 + bt3.s5 * w4.s2 + bt3.s6 * w4.s3 + bt3.s7 * w5th;

			out2.s0 += bt0.s0 * w4.s0 + bt0.s1 * w4.s1 + bt0.s2 * w4.s2 + bt0.s3 * w4.s3 + bt0.s4 * w5th;
			out2.s1 += bt0.s1 * w4.s0 + bt0.s2 * w4.s1 + bt0.s3 * w4.s2 + bt0.s4 * w4.s3 + bt0.s5 * w5th;
			out2.s2 += bt0.s2 * w4.s0 + bt0.s3 * w4.s1 + bt0.s4 * w4.s2 + bt0.s5 * w4.s3 + bt0.s6 * w5th;
			out2.s3 += bt0.s3 * w4.s0 + bt0.s4 * w4.s1 + bt0.s5 * w4.s2 + bt0.s6 * w4.s3 + bt0.s7 * w5th;


// 5th data 

			bt1 = *(__local _FLOAT8*)&bottom_data[bot_data_off + _DNN_CONV_GRP_DATA_WIDTH * 5];

			out3.s0 += bt1.s0 * w4.s0 + bt1.s1 * w4.s1 + bt1.s2 * w4.s2 + bt1.s3 * w4.s3 + bt1.s4 * w5th;
			out3.s1 += bt1.s1 * w4.s0 + bt1.s2 * w4.s1 + bt1.s3 * w4.s2 + bt1.s4 * w4.s3 + bt1.s5 * w5th;
			out3.s2 += bt1.s2 * w4.s0 + bt1.s3 * w4.s1 + bt1.s4 * w4.s2 + bt1.s5 * w4.s3 + bt1.s6 * w5th;
			out3.s3 += bt1.s3 * w4.s0 + bt1.s4 * w4.s1 + bt1.s5 * w4.s2 + bt1.s6 * w4.s3 + bt1.s7 * w5th;

// 3nd filter row
			w4 = *(__local _FLOAT4*)&weights_data[_DNN_CONV_KERNEL_SZ * 3];
			w5th = weights_data[_DNN_CONV_KERNEL_SZ*3 + 4];

			out0.s0 += bt3.s0 * w4.s0 + bt3.s1 * w4.s1 + bt3.s2 * w4.s2 + bt3.s3 * w4.s3 + bt3.s4 * w5th;
			out0.s1 += bt3.s1 * w4.s0 + bt3.s2 * w4.s1 + bt3.s3 * w4.s2 + bt3.s4 * w4.s3 + bt3.s5 * w5th;
			out0.s2 += bt3.s2 * w4.s0 + bt3.s3 * w4.s1 + bt3.s4 * w4.s2 + bt3.s5 * w4.s3 + bt3.s6 * w5th;
			out0.s3 += bt3.s3 * w4.s0 + bt3.s4 * w4.s1 + bt3.s5 * w4.s2 + bt3.s6 * w4.s3 + bt3.s7 * w5th;

			out1.s0 += bt0.s0 * w4.s0 + bt0.s1 * w4.s1 + bt0.s2 * w4.s2 + bt0.s3 * w4.s3 + bt0.s4 * w5th;
			out1.s1 += bt0.s1 * w4.s0 + bt0.s2 * w4.s1 + bt0.s3 * w4.s2 + bt0.s4 * w4.s3 + bt0.s5 * w5th;
			out1.s2 += bt0.s2 * w4.s0 + bt0.s3 * w4.s1 + bt0.s4 * w4.s2 + bt0.s5 * w4.s3 + bt0.s6 * w5th;
			out1.s3 += bt0.s3 * w4.s0 + bt0.s4 * w4.s1 + bt0.s5 * w4.s2 + bt0.s6 * w4.s3 + bt0.s7 * w5th;

			out2.s0 += bt1.s0 * w4.s0 + bt1.s1 * w4.s1 + bt1.s2 * w4.s2 + bt1.s3 * w4.s3 + bt1.s4 * w5th;
			out2.s1 += bt1.s1 * w4.s0 + bt1.s2 * w4.s1 + bt1.s3 * w4.s2 + bt1.s4 * w4.s3 + bt1.s5 * w5th;
			out2.s2 += bt1.s2 * w4.s0 + bt1.s3 * w4.s1 + bt1.s4 * w4.s2 + bt1.s5 * w4.s3 + bt1.s6 * w5th;
			out2.s3 += bt1.s3 * w4.s0 + bt1.s4 * w4.s1 + bt1.s5 * w4.s2 + bt1.s6 * w4.s3 + bt1.s7 * w5th;

// 6th data 

			bt2 = *(__local _FLOAT8*)&bottom_data[bot_data_off + _DNN_CONV_GRP_DATA_WIDTH * 6];

			out3.s0 += bt2.s0 * w4.s0 + bt2.s1 * w4.s1 + bt2.s2 * w4.s2 + bt2.s3 * w4.s3 + bt2.s4 * w5th;
			out3.s1 += bt2.s1 * w4.s0 + bt2.s2 * w4.s1 + bt2.s3 * w4.s2 + bt2.s4 * w4.s3 + bt2.s5 * w5th;
			out3.s2 += bt2.s2 * w4.s0 + bt2.s3 * w4.s1 + bt2.s4 * w4.s2 + bt2.s5 * w4.s3 + bt2.s6 * w5th;
			out3.s3 += bt2.s3 * w4.s0 + bt2.s4 * w4.s1 + bt2.s5 * w4.s2 + bt2.s6 * w4.s3 + bt2.s7 * w5th;

// 4th (last) filter row
			w4 = *(__local _FLOAT4*)&weights_data[_DNN_CONV_KERNEL_SZ * 4];
			w5th = weights_data[_DNN_CONV_KERNEL_SZ*4 + 4];

			out0.s0 += bt0.s0 * w4.s0 + bt0.s1 * w4.s1 + bt0.s2 * w4.s2 + bt0.s3 * w4.s3 + bt0.s4 * w5th;
			out0.s1 += bt0.s1 * w4.s0 + bt0.s2 * w4.s1 + bt0.s3 * w4.s2 + bt0.s4 * w4.s3 + bt0.s5 * w5th;
			out0.s2 += bt0.s2 * w4.s0 + bt0.s3 * w4.s1 + bt0.s4 * w4.s2 + bt0.s5 * w4.s3 + bt0.s6 * w5th;
			out0.s3 += bt0.s3 * w4.s0 + bt0.s4 * w4.s1 + bt0.s5 * w4.s2 + bt0.s6 * w4.s3 + bt0.s7 * w5th;

			out1.s0 += bt1.s0 * w4.s0 + bt1.s1 * w4.s1 + bt1.s2 * w4.s2 + bt1.s3 * w4.s3 + bt1.s4 * w5th;
			out1.s1 += bt1.s1 * w4.s0 + bt1.s2 * w4.s1 + bt1.s3 * w4.s2 + bt1.s4 * w4.s3 + bt1.s5 * w5th;
			out1.s2 += bt1.s2 * w4.s0 + bt1.s3 * w4.s1 + bt1.s4 * w4.s2 + bt1.s5 * w4.s3 + bt1.s6 * w5th;
			out1.s3 += bt1.s3 * w4.s0 + bt1.s4 * w4.s1 + bt1.s5 * w4.s2 + bt1.s6 * w4.s3 + bt1.s7 * w5th;

			out2.s0 += bt2.s0 * w4.s0 + bt2.s1 * w4.s1 + bt2.s2 * w4.s2 + bt2.s3 * w4.s3 + bt2.s4 * w5th;
			out2.s1 += bt2.s1 * w4.s0 + bt2.s2 * w4.s1 + bt2.s3 * w4.s2 + bt2.s4 * w4.s3 + bt2.s5 * w5th;
			out2.s2 += bt2.s2 * w4.s0 + bt2.s3 * w4.s1 + bt2.s4 * w4.s2 + bt2.s5 * w4.s3 + bt2.s6 * w5th;
			out2.s3 += bt2.s3 * w4.s0 + bt2.s4 * w4.s1 + bt2.s5 * w4.s2 + bt2.s6 * w4.s3 + bt2.s7 * w5th;

			out3.s0 += bt3.s0 * w4.s0 + bt3.s1 * w4.s1 + bt3.s2 * w4.s2 + bt3.s3 * w4.s3 + bt3.s4 * w5th;
			out3.s1 += bt3.s1 * w4.s0 + bt3.s2 * w4.s1 + bt3.s3 * w4.s2 + bt3.s4 * w4.s3 + bt3.s5 * w5th;
			out3.s2 += bt3.s2 * w4.s0 + bt3.s3 * w4.s1 + bt3.s4 * w4.s2 + bt3.s5 * w4.s3 + bt3.s6 * w5th;
			out3.s3 += bt3.s3 * w4.s0 + bt3.s4 * w4.s1 + bt3.s5 * w4.s2 + bt3.s6 * w4.s3 + bt3.s7 * w5th;
		}

		int top_off = b * (top_batch_stride>>2) + o * (top_channel_stride >> 2) + (top_group_y  + lcl_id1) * _DNN_CONV_N_VERT_OUT_PIX  * (top_stride>>2) + top_group_x + lcl_id0; //X is per 4
		top[top_off] = out0;
		top[top_off + (top_stride>>2)] = out1;
		top[top_off + (top_stride>>2) *2] = out2;
		top[top_off + (top_stride>>2) *3] = out3;
}

__kernel void aDNNConv2(
       const __global _FLOAT * bottom,
       const __global _FLOAT * weights,
       __global _FLOAT2 * top,
	   float f_xy_groups_horiz,
	   int i_xy_groups_horiz,
	   int bot_width,
	   int bot_height,
	   int bot_stride,
	   int bot_channel_stride,
	   int bot_batch_stride,
	   int top_stride,
	   int top_channel_stride,
	   int top_batch_stride,
	   int n_channels,
	   int kernel_sz
	   )
{
		__local _FLOAT bottom_data[_DNN_CONV_GRP_DATA_WIDTH * _DNN_CONV_GRP_DATA_HEIGHT];
		__local _FLOAT weights_data[_DNN_CONV_KERNEL_SZ * _DNN_CONV_KERNEL_SZ];
	
		int lcl_id = get_local_id(0);
		int lcl_id1 = lcl_id >> _DNN_CONV_GROUP_LG2SZ0;
		int lcl_id0 = lcl_id - (lcl_id1 << _DNN_CONV_GROUP_LG2SZ0);	   	
		int xy = get_group_id(0); // channel position
		int o = get_global_id(1); // output
		int b = get_global_id(2); // batch 
		int pad = _DNN_CONV_PAD;
		int group_row = (int)((float)xy/f_xy_groups_horiz);
		int group_col = xy - group_row * i_xy_groups_horiz;
		int bot_group_x_s = group_col * _DNN_CONV_GROUP_SZ0 *_DNN_CONV_N_HORIZ_OUT_PIX - pad;
		int bot_group_y_s = group_row * _DNN_CONV_GROUP_SZ1 * _DNN_CONV_N_VERT_OUT_PIX - pad;
		int bot_group_x_e = bot_group_x_s + _DNN_CONV_GRP_DATA_WIDTH;
		int bot_group_y_e = bot_group_y_s + _DNN_CONV_GRP_DATA_HEIGHT;
		int top_group_x = group_col * _DNN_CONV_GROUP_SZ0; // *_DNN_CONV_N_HORIZ_OUT_PIX;
		int top_group_y = group_row * _DNN_CONV_GROUP_SZ1; 

		int weights_off = o * _DNN_CONV_KERNEL_SZ * _DNN_CONV_KERNEL_SZ * _DNN_CONV_N_CHANNELS;


		int lcl_vertical_off = 0;
		int lcl_horizontal_off = 0;
		int vertical_read = _DNN_CONV_GRP_DATA_HEIGHT;
		int horizontal_read = _DNN_CONV_GRP_DATA_WIDTH;
		int bottom_group_y = bot_group_y_s;
		int bottom_group_x = bot_group_x_s;
#if 1
		bool on_the_edge = (bot_group_y_s < 0 || bot_group_y_e > bot_height || bot_group_x_s < 0 || bot_group_x_e > bot_width);


		if ( on_the_edge )
		{
			lcl_vertical_off = (bot_group_y_s < 0) ? -bot_group_y_s : 0;
			vertical_read -= lcl_vertical_off;
			bottom_group_y = bot_group_y_s + lcl_vertical_off;
			vertical_read -= (bot_group_y_e > bot_height) ? bottom_group_y - bot_height : 0;
			lcl_horizontal_off = (bot_group_x_s < 0) ? -bot_group_x_s : 0;	 
			horizontal_read -= lcl_horizontal_off;
			bottom_group_x = bot_group_x_s + lcl_horizontal_off;
			horizontal_read -= (bot_group_x_e > bot_width) ? bot_group_x_e - bot_width : 0;
		}
#endif
		bottom_group_x &= ~2;
		horizontal_read = (((horizontal_read + 3) >> 2) << 2);

		_FLOAT2 out0 = 0, out1 = 0;

//#pragma unroll 1
		for( int c = 0; c < _DNN_CONV_N_CHANNELS; c++)
		{
// get weights per channel per ouput
#if 1
			for(int w = lcl_id; w < _DNN_CONV_KERNEL_SZ * _DNN_CONV_KERNEL_SZ; w += _DNN_CONV_GROUP_SZ)
			{
				weights_data[w] = weights[weights_off + c * _DNN_CONV_KERNEL_SZ * _DNN_CONV_KERNEL_SZ + w];
			}
#endif
// get data per channel
#if 1
			for(int j = lcl_id; on_the_edge && j < _DNN_CONV_GRP_DATA_WIDTH * _DNN_CONV_GRP_DATA_HEIGHT/2; j += _DNN_CONV_GROUP_SZ)
			{
					bottom_data[2*j +0] = 0;
					bottom_data[2*j +1] = 0;
			}

			barrier(CLK_LOCAL_MEM_FENCE);
#endif

			for( int b_j = lcl_id1; b_j < vertical_read; b_j += _DNN_CONV_GROUP_SZ1)
			{			
				for(int b_i = lcl_id0; b_i < horizontal_read; b_i += _DNN_CONV_GROUP_SZ0 * 4)
				{
					*(__local _FLOAT4*)&bottom_data[(lcl_vertical_off + b_j) * _DNN_CONV_GRP_DATA_WIDTH + lcl_horizontal_off + b_i] =
						*(__global _FLOAT4*)&bottom[bot_batch_stride * b + bot_channel_stride *c + bot_stride * (bottom_group_y + b_j) + bottom_group_x + b_i];
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE);
// compute per channel
			int bot_data_off = lcl_id1 * _DNN_CONV_N_VERT_OUT_PIX * _DNN_CONV_GRP_DATA_WIDTH + lcl_id0 * _DNN_CONV_N_HORIZ_OUT_PIX;
		
			_FLOAT2 bt0, bt1;


			for(int j = 0; j <  _DNN_CONV_KERNEL_SZ; j++)
			{

				bt0.s0 = bottom_data[bot_data_off + (j + 0) * _DNN_CONV_GRP_DATA_WIDTH + 0] ;
				bt0.s1 = bottom_data[bot_data_off + (j + 0) * _DNN_CONV_GRP_DATA_WIDTH + 1] ;

				bt1.s0 = bottom_data[bot_data_off + (j + 1) * _DNN_CONV_GRP_DATA_WIDTH + 0] ;
				bt1.s1 = bottom_data[bot_data_off + (j + 1) * _DNN_CONV_GRP_DATA_WIDTH + 1] ;

				for(int i = 0; i <  _DNN_CONV_KERNEL_SZ; i++)
				{
					_FLOAT2 wi2 = (_FLOAT2)weights_data[ _DNN_CONV_KERNEL_SZ * j + i];

					out0 += bt0 * wi2;
					out1 += bt1 * wi2;

					bt0.s0 = bt0.s1;
					bt0.s1 = bottom_data[bot_data_off + (j + 0) * _DNN_CONV_GRP_DATA_WIDTH + i + 2];

					bt1.s0 = bt1.s1;
					bt1.s1 = bottom_data[bot_data_off + (j + 1) * _DNN_CONV_GRP_DATA_WIDTH + i + 2];


				}
			}

		}

		int top_off = b * (top_batch_stride>>1) + o * (top_channel_stride >> 1) + (top_group_y  + lcl_id1) * _DNN_CONV_N_VERT_OUT_PIX  * (top_stride>>1) + top_group_x + lcl_id0; //X is per 4
		top[top_off] = out0;
		top[top_off + (top_stride>>1)] = out1;
}
