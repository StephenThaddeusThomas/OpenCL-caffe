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


#define ADNN_CONV_GROUP_SZ2 1



#define ADNN_CONV_GRP_DATA_WIDTH (ADNN_CONV_GROUP_SZ0 *ADNN_CONV_N_HORIZ_OUT_PIX *ADNN_CONV_STRIDE + ADNN_CONV_KERNEL_SZ - 1)
#define ADNN_CONV_GRP_DATA_HEIGHT  (ADNN_CONV_GROUP_SZ1 * ADNN_CONV_N_VERT_OUT_PIX * ADNN_CONV_STRIDE + ADNN_CONV_KERNEL_SZ - 1)
#define ADNN_CONV_GROUP_SZ (ADNN_CONV_GROUP_SZ1 * ADNN_CONV_GROUP_SZ0)
#define ADNN_CONV_KERN_AREA_SZ (ADNN_CONV_KERNEL_SZ * ADNN_CONV_KERNEL_SZ)   // + bias




__attribute__((reqd_work_group_size(ADNN_CONV_GROUP_SZ0,ADNN_CONV_GROUP_SZ1,ADNN_CONV_GROUP_SZ2)))
__kernel void aDNNConv_GT32_NCHW(
       const __global _FLOAT * bottom,
       const __global _FLOAT * weights,
       const __global _FLOAT * bias,
       __global _FLOAT * top,
	   _FLOAT padding_val
	   )
{
		
		__local _FLOAT bottom_data[ADNN_CONV_GRP_DATA_WIDTH * ADNN_CONV_GRP_DATA_HEIGHT];
		__local _FLOAT weights_data[ADNN_CONV_N_OUTS * ADNN_CONV_KERN_AREA_SZ]; // + bias
	
		int x_out = get_group_id(0) * ADNN_CONV_GROUP_SZ0 * ADNN_CONV_N_HORIZ_OUT_PIX;
		int y_out = get_group_id(1) * ADNN_CONV_GROUP_SZ1 * ADNN_CONV_N_VERT_OUT_PIX;
		int x = x_out *ADNN_CONV_STRIDE;
		int y = y_out *ADNN_CONV_STRIDE;
		int lcl_id0 = get_local_id(0);
		int lcl_id1 = get_local_id(1);
		int lcl_id = (lcl_id1 << ADNN_CONV_GROUP_LG2SZ0) + lcl_id0;
		int ob = get_global_id(2); // output * batch_sz
		int b = (int)(float)ob / (float)ADNN_CONV_N_OUTPUTS;
		int o = ob - b * ADNN_CONV_N_OUTPUTS;

		int bot_off = b * ADNN_CONV_BOT_BATCH_STRIDE;
		int gbl_weights_off = o * ADNN_CONV_WEIGHTS_STRIDE * ADNN_CONV_N_OUTS; // + bias 
		int lcl_weights_off = 0;

		_FLOAT accum[ADNN_CONV_N_OUTS][ADNN_CONV_N_VERT_OUT_PIX][ADNN_CONV_N_HORIZ_OUT_PIX];

		for(int k = 0; k <ADNN_CONV_N_OUTS; k++)
		{

			for(int j = 0; j < ADNN_CONV_N_VERT_OUT_PIX; j++)
			{
				for(int i = 0; i < ADNN_CONV_N_HORIZ_OUT_PIX; i++)
				{

					accum[k][j][i] = 0;
				}
			}
		}

#pragma loop nounroll
		for( int c = 0; c < ADNN_CONV_N_CHANNELS; c++, bot_off += ADNN_CONV_BOT_CHANNEL_STRIDE)
		{
			gbl_weights_off = o * ADNN_CONV_WEIGHTS_STRIDE * ADNN_CONV_N_OUTS + c * ADNN_CONV_KERN_AREA_SZ; // + bias 
			lcl_weights_off = 0;
// get weights per channel
			for(int k = 0; k < ADNN_CONV_N_OUTS; k++, gbl_weights_off += ADNN_CONV_WEIGHTS_STRIDE, lcl_weights_off += ADNN_CONV_KERN_AREA_SZ)
			{
				for(int w = lcl_id; w < ADNN_CONV_KERN_AREA_SZ; w += ADNN_CONV_GROUP_SZ)
				{
					weights_data[lcl_weights_off + w] = weights[gbl_weights_off + w];
				}
			}


			for( int b_j = lcl_id1; b_j < ADNN_CONV_GRP_DATA_HEIGHT; b_j += ADNN_CONV_GROUP_SZ1)
			{	
				int y_act = (y + b_j - ADNN_CONV_PAD);
				bool invisibleY = (y_act < 0) || (y_act >= ADNN_CONV_BOT_VIS_HEIGHT);
//				y_act = (y_act < 0) ? 0 : (y_act >= ADNN_CONV_BOT_VIS_HEIGHT) ? ADNN_CONV_BOT_VIS_HEIGHT - 1: y_act;

				int y_off = y_act * ADNN_CONV_BOT_STRIDE;

				int lcl_off_v = b_j * ADNN_CONV_GRP_DATA_WIDTH;

				for(int b_i = lcl_id0; b_i < ADNN_CONV_GRP_DATA_WIDTH; b_i += ADNN_CONV_GROUP_SZ0)
				{
					int x_act = (x + b_i - ADNN_CONV_PAD);
					bool invisibleX = (x_act < 0) || (x_act >= ADNN_CONV_BOT_VIS_WIDTH);
//					x_act = (x_act < 0) ? 0 : (x_act >= ADNN_CONV_BOT_VIS_WIDTH) ? ADNN_CONV_BOT_VIS_WIDTH - 1 : x_act;	

					_FLOAT bot_val = bottom[bot_off + y_off + x_act];

					bot_val = (invisibleX || invisibleY)? padding_val : bot_val;
								
					bottom_data[lcl_off_v + b_i] = bot_val;
				}
			}



			barrier(CLK_LOCAL_MEM_FENCE);

			#define ADNN_CONV_N_PRV_HORIZ (ADNN_CONV_KERNEL_SZ + (ADNN_CONV_N_HORIZ_OUT_PIX -1) *ADNN_CONV_STRIDE)
			#define ADNN_CONV_N_PRV_VERT ((ADNN_CONV_N_VERT_OUT_PIX - 1)*ADNN_CONV_STRIDE + 1)
			_FLOAT data_stage[ADNN_CONV_N_PRV_VERT][ADNN_CONV_N_PRV_HORIZ];

			int lcl_y = lcl_id1 * ADNN_CONV_N_VERT_OUT_PIX *ADNN_CONV_STRIDE;
			int lcl_x = lcl_id0 * ADNN_CONV_N_HORIZ_OUT_PIX *ADNN_CONV_STRIDE;
			int lcl_bot_off = lcl_y * ADNN_CONV_GRP_DATA_WIDTH + lcl_x;


			int l_j = 0;
			for( ; l_j < ADNN_CONV_N_PRV_VERT; l_j++, lcl_bot_off += ADNN_CONV_GRP_DATA_WIDTH)
			{	

				for(int l_i = 0; l_i < ADNN_CONV_N_PRV_HORIZ; l_i++)
				{

					data_stage[l_j][l_i] = bottom_data[lcl_bot_off + l_i];

				}
			}

			

			int l_j_c = l_j;
			for( ; l_j < l_j_c + ADNN_CONV_KERNEL_SZ; l_j++, lcl_bot_off += ADNN_CONV_GRP_DATA_WIDTH)
			{	

				lcl_weights_off = 0;
				for(int ko = 0; ko < ADNN_CONV_N_OUTS; ko++, lcl_weights_off += ADNN_CONV_KERN_AREA_SZ)
				{
					int k = l_j - ADNN_CONV_N_PRV_VERT;
					for(int j = 0; j < ADNN_CONV_N_VERT_OUT_PIX; j++)
					{
						for(int i = 0; i < ADNN_CONV_N_HORIZ_OUT_PIX; i++)
						{
							for(int l = 0; l < ADNN_CONV_KERNEL_SZ; l++)
							{

								accum[ko][j][i] += data_stage[j*ADNN_CONV_STRIDE][i*ADNN_CONV_STRIDE + l] * weights_data[lcl_weights_off + k*ADNN_CONV_KERNEL_SZ + l]; 	
								
							}
						}
					}

				}

				for(int j = 0; j < ADNN_CONV_N_PRV_VERT - 1; j++)
				{
					for(int i = 0; i < ADNN_CONV_N_PRV_HORIZ; i++)
					{
						data_stage[j][i] = data_stage[j+1][i];
					}
				}

				for(int l_i = 0; l_i < ADNN_CONV_N_PRV_HORIZ; l_i++)
				{
					data_stage[ADNN_CONV_N_PRV_VERT - 1][l_i] = bottom_data[lcl_bot_off + l_i];

				}


			}

		}

		int top_out_off =  o * ADNN_CONV_TOP_CHANNEL_STRIDE * ADNN_CONV_N_OUTS;
		gbl_weights_off = o * ADNN_CONV_WEIGHTS_STRIDE * ADNN_CONV_N_OUTS; 


		for(int ko = 0; ko < ADNN_CONV_N_OUTS; ko++, top_out_off += ADNN_CONV_TOP_CHANNEL_STRIDE, gbl_weights_off += ADNN_CONV_WEIGHTS_STRIDE)
		{
			int out_y = (y_out + lcl_id1 * ADNN_CONV_N_VERT_OUT_PIX);
			int out_x = (x_out + lcl_id0 * ADNN_CONV_N_HORIZ_OUT_PIX);
			_FLOAT bias_val = bias[o + ko];

			int top_off = b * ADNN_CONV_TOP_BATCH_STRIDE + top_out_off +  out_y * ADNN_CONV_TOP_STRIDE + out_x;
			for(int j = 0; j < ADNN_CONV_N_VERT_OUT_PIX; j++, top_off += ADNN_CONV_TOP_STRIDE, out_y++)
			{
				int out_x2 = out_x;
				for(int i = 0; i < ADNN_CONV_N_HORIZ_OUT_PIX; i++, out_x2++)
				{	
#if ADNN_ALIGNED != 1				
					if (out_y < ADNN_CONV_TOP_HEIGHT && out_x2 < ADNN_CONV_TOP_WIDTH)
#endif
					{

						_FLOAT out_val = accum[ko][j][i];
						top[top_off + i] = out_val + bias_val;
#if 0
					if ( o==0 && b==0 && x==0 && y == 0 && lcl_id0 ==0 && lcl_id1==0 && ko == 1 && i==0 && j == 0 )
					{
						printf("K out: %d %d %d %d %d %d %d %d %d %d   %f %f %f\n", o, b, ko, x, y, lcl_id0, lcl_id1, j, i, top_off + i, top[top_off + i], out_val, 
						bias);
					}
#endif
					}
				}
			}
		}
}



