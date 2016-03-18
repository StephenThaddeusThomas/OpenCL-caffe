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

#define ADNN_CONVBWD_GROUP_SZ2 1

#if 1
#define ADNN_CONVBWD_BOT_DATA_WIDTH (ADNN_CONVBWD_GROUP_SZ0 *ADNN_CONVBWD_PRV_TOPDF_W *ADNN_CONV_STRIDE + ADNN_CONV_KERNEL_SZ - 1)
#define ADNN_CONVBWD_BOT_DATA_HEIGHT  (ADNN_CONVBWD_GROUP_SZ1 * ADNN_CONVBWD_PRV_TOPDF_H * ADNN_CONV_STRIDE + ADNN_CONV_KERNEL_SZ - 1)
#define ADNN_CONVBWD_GROUP_SZ (ADNN_CONVBWD_GROUP_SZ1 * ADNN_CONVBWD_GROUP_SZ0)
#define ADNN_CONV_KERN_AREA_SZ (ADNN_CONV_KERNEL_SZ * ADNN_CONV_KERNEL_SZ)

// here we do the following
// global work load is per input x output
// loop over all tiles of a particular input-output pair and accumulate gradient per each kernel(filter) component per input
// accumulate the result over batch size
// TO DO: VERIFY STRIDE

__attribute__((reqd_work_group_size(ADNN_CONVBWD_GROUP_SZ0,ADNN_CONVBWD_GROUP_SZ1,ADNN_CONVBWD_GROUP_SZ2)))
__kernel void aDNNConvBwd_wrt_W(
       const __global _FLOAT * top_df,
       const __global _FLOAT * bot,
// need another pass to sum over teh batch
       __global _FLOAT * weights_df,
	   _FLOAT padding_val
	   )
{

// bot data
		__local _FLOAT bot_data[ADNN_CONVBWD_BOT_DATA_WIDTH * ADNN_CONVBWD_BOT_DATA_HEIGHT];
// private top_df tile
		_FLOAT prv_top_df[ADNN_CONVBWD_PRV_TOPDF_H][ADNN_CONVBWD_PRV_TOPDF_W];
// final sum
		__local _FLOAT weights_df_sum[ ADNN_CONV_KERN_AREA_SZ + 1]; //+ bias
// intermediate sum
		__local _FLOAT weights_df_psum[ADNN_CONV_KERN_AREA_SZ +  (ADNN_CONVBWD_GROUP_SZ -1) * ADNN_CONV_KERNEL_SZ  + 1]; // + bias

		int lcl_id0 = get_local_id(0);
		int lcl_id1 = get_local_id(1);
		int lcl_id = (lcl_id1 << ADNN_CONVBWD_GROUP_LG2SZ0) + lcl_id0;
		int ob = get_global_id(2); // output * inputs * batch_sz
		int b = (int)(float)ob / (float)(ADNN_CONV_N_OUTPUTS * ADNN_CONV_N_INPUTS);
		int co = ob - b * (ADNN_CONV_N_OUTPUTS * ADNN_CONV_N_INPUTS);
//		int co = get_global_id(2); // output * inputs
		int o = (int)((float)co/(float)ADNN_CONV_N_INPUTS); 
		int c = co - o * ADNN_CONV_N_INPUTS;


		for(int i = lcl_id; i < ADNN_CONV_KERN_AREA_SZ + 1; i+= ADNN_CONVBWD_GROUP_SZ)
		{
			weights_df_sum[i] = 0;
	
		}



		barrier(CLK_LOCAL_MEM_FENCE);
	
// over batch		
//		for(int b = 0; b < ADNN_CONV_BATCH_SZ; ++b)
		{
// over single pair of input/output
			int top_df_off = b * ADNN_CONVBWD_TOPDF_BATCH_STRIDE + o * ADNN_CONVBWD_TOPDF_CHANNEL_STRIDE;
			int bot_off = b * ADNN_CONV_BOT_BATCH_STRIDE + c * ADNN_CONV_BOT_CHANNEL_STRIDE;

// over all tiles 
			for( int tv = 0; tv < ADNN_CONVBWD_N_TILES_V; ++tv)
			{
				int y = tv * ADNN_CONVBWD_GROUP_SZ1 * ADNN_CONVBWD_PRV_TOPDF_H;
				for( int th = 0; th < ADNN_CONVBWD_N_TILES_H; ++th)
				{

					int x = th * ADNN_CONVBWD_GROUP_SZ0 * ADNN_CONVBWD_PRV_TOPDF_W;
		
					int x_df = x;
					int y_df = y;
// first read top_df tile
// into private memory
					for(int j = lcl_id1 *ADNN_CONVBWD_PRV_TOPDF_H,k = 0; k < ADNN_CONVBWD_PRV_TOPDF_H; ++k)
					{
						for(int i = lcl_id0 * ADNN_CONVBWD_PRV_TOPDF_W, l = 0; l <  ADNN_CONVBWD_PRV_TOPDF_W; ++l)
						{
							_FLOAT top_val = top_df[top_df_off + (y_df + j + k) * ADNN_CONVBWD_TOPDF_STRIDE + x_df + i + l];
							prv_top_df[k][l] = ( (y_df + j + k) < ADNN_CONV_TOP_HEIGHT && (x_df + i + l) < ADNN_CONV_TOP_WIDTH ) ? top_val : 0;
#if 0
			 if ( c==0 && o == 0  &&
				(
				( (y_df + j + k) * ADNN_CONVBWD_TOPDF_STRIDE + x_df + i + l) == 33
				|| ((y_df + j + k) * ADNN_CONVBWD_TOPDF_STRIDE + x_df + i + l) == 49
				|| ((y_df + j + k) * ADNN_CONVBWD_TOPDF_STRIDE + x_df + i + l) == 65
				|| ((y_df + j + k) * ADNN_CONVBWD_TOPDF_STRIDE + x_df + i + l) == 81
				|| ((y_df + j + k) * ADNN_CONVBWD_TOPDF_STRIDE + x_df + i + l) == 97
				|| ((y_df + j + k) * ADNN_CONVBWD_TOPDF_STRIDE + x_df + i + l) == 113
				|| ((y_df + j + k) * ADNN_CONVBWD_TOPDF_STRIDE + x_df + i + l) == 129
				|| ((y_df + j + k) * ADNN_CONVBWD_TOPDF_STRIDE + x_df + i + l) == 145
				|| ((y_df + j + k) * ADNN_CONVBWD_TOPDF_STRIDE + x_df + i + l) == 161
				|| ((y_df + j + k) * ADNN_CONVBWD_TOPDF_STRIDE + x_df + i + l) == 177
				|| ((y_df + j + k) * ADNN_CONVBWD_TOPDF_STRIDE + x_df + i + l) == 193
				|| ((y_df + j + k) * ADNN_CONVBWD_TOPDF_STRIDE + x_df + i + l) == 209
				)
			 )
			 {
				printf("K: top in: %d %d %d %d  %11.9f\n", lcl_id0, l, lcl_id1, k, prv_top_df[k][l]);
			 }
#endif
						}
					}

// load bot data tile with padding
					int x_bot = x * ADNN_CONV_STRIDE;
					int y_bot = y * ADNN_CONV_STRIDE;

					for( int b_j = lcl_id1; b_j < ADNN_CONVBWD_BOT_DATA_HEIGHT; b_j += ADNN_CONVBWD_GROUP_SZ1)
					{	
						int y_act = (y_bot + b_j - ADNN_CONV_PAD);
						bool invisibleY = (y_act < 0) || (y_act >= ADNN_CONV_BOT_HEIGHT);
//				y_act = (y_act < 0) ? 0 : (y_act >= ADNN_CONV_BOT_VIS_HEIGHT) ? ADNN_CONV_BOT_VIS_HEIGHT - 1: y_act;

						int y_off = y_act * ADNN_CONV_BOT_STRIDE;

						int lcl_off_v = b_j * ADNN_CONVBWD_BOT_DATA_WIDTH;

						for(int b_i = lcl_id0; b_i < ADNN_CONVBWD_BOT_DATA_WIDTH; b_i += ADNN_CONVBWD_GROUP_SZ0)
						{
							int x_act = (x_bot + b_i - ADNN_CONV_PAD);
							bool invisibleX = (x_act < 0) || (x_act >= ADNN_CONV_BOT_WIDTH);
//					x_act = (x_act < 0) ? 0 : (x_act >= ADNN_CONV_BOT_VIS_WIDTH) ? ADNN_CONV_BOT_VIS_WIDTH - 1 : x_act;	

							_FLOAT bot_val = bot[bot_off + y_off + x_act];


							bot_val = (invisibleX || invisibleY)? 0 : bot_val;
								
#if 0
				int k_y = 1;
				int k_x = 1;
			 if ( c==0 && o == 0 && !(invisibleX || invisibleY)  && 
				(
				((y_act - k_y + ADNN_CONV_PAD) * ADNN_CONV_BOT_STRIDE + x_act - k_x +  ADNN_CONV_PAD) == 33
				|| ((y_act - k_y+ ADNN_CONV_PAD) * ADNN_CONV_BOT_STRIDE + x_act  - k_x +  ADNN_CONV_PAD) == 49
				|| ((y_act - k_y + ADNN_CONV_PAD) * ADNN_CONV_BOT_STRIDE + x_act  - k_x +  ADNN_CONV_PAD) == 65
				|| ((y_act - k_y + ADNN_CONV_PAD) * ADNN_CONV_BOT_STRIDE + x_act  - k_x +  ADNN_CONV_PAD) == 81
				|| ((y_act - k_y + ADNN_CONV_PAD) * ADNN_CONV_BOT_STRIDE + x_act  - k_x +  ADNN_CONV_PAD) == 97
				|| ((y_act - k_y + ADNN_CONV_PAD) * ADNN_CONV_BOT_STRIDE + x_act  - k_x +  ADNN_CONV_PAD) == 113
				|| ((y_act - k_y + ADNN_CONV_PAD) * ADNN_CONV_BOT_STRIDE + x_act  - k_x +  ADNN_CONV_PAD) == 129
				|| ((y_act - k_y + ADNN_CONV_PAD) * ADNN_CONV_BOT_STRIDE + x_act  - k_x +  ADNN_CONV_PAD) == 145
				|| ((y_act - k_y + ADNN_CONV_PAD) * ADNN_CONV_BOT_STRIDE + x_act  - k_x +  ADNN_CONV_PAD) == 161
				|| ((y_act - k_y + ADNN_CONV_PAD) * ADNN_CONV_BOT_STRIDE + x_act  - k_x +  ADNN_CONV_PAD) == 177
				|| ((y_act - k_y + ADNN_CONV_PAD) * ADNN_CONV_BOT_STRIDE + x_act  - k_x +  ADNN_CONV_PAD) == 193
				|| ((y_act - k_y + ADNN_CONV_PAD) * ADNN_CONV_BOT_STRIDE + x_act  - k_x +  ADNN_CONV_PAD) == 209
				)
			 )
			 {
				printf("K: bot in: %d  %11.9f\n", ((y_act - k_y + ADNN_CONV_PAD) * ADNN_CONV_BOT_STRIDE  - k_x + x_act +  ADNN_CONV_PAD), bot_val);
			 }
#endif

							bot_data[lcl_off_v + b_i] = bot_val;

						}
					}

					barrier(CLK_LOCAL_MEM_FENCE);
// inner loop
// sum up (convolve) top_df tile with width x height of bot tile per each kernel element
			#define ADNN_CONV_N_PRV_HORIZ (ADNN_CONV_KERNEL_SZ + (ADNN_CONVBWD_PRV_TOPDF_W -1) *ADNN_CONV_STRIDE)
			#define ADNN_CONV_N_PRV_VERT ((ADNN_CONVBWD_PRV_TOPDF_H - 1)*ADNN_CONV_STRIDE + 1)

					_FLOAT prv_bot[ADNN_CONV_N_PRV_VERT][ADNN_CONV_N_PRV_HORIZ];
					_FLOAT prv_accum[ADNN_CONV_KERNEL_SZ];
// prefetch
					int j = lcl_id1 * ADNN_CONVBWD_PRV_TOPDF_H * ADNN_CONV_STRIDE;
					int i = lcl_id0 * ADNN_CONVBWD_PRV_TOPDF_W * ADNN_CONV_STRIDE;
					for(int k = 0; k < ADNN_CONV_N_PRV_VERT; ++k)
					{
						int lcl_off_v = (0 + j + k) * ADNN_CONVBWD_BOT_DATA_WIDTH;
						for(int l = 0; l <   ADNN_CONV_N_PRV_HORIZ; ++l)
						{
							prv_bot[k][l] = bot_data[lcl_off_v + i + l];

						}
					}

					for(int kk = 0; kk < ADNN_CONV_KERNEL_SZ; ++kk)
					{

// convolve: result is a horizontal filter row 
						for(int kl = 0; kl <  ADNN_CONV_KERNEL_SZ; ++kl)
						{
							prv_accum[kl] = 0;
						}

						for(int k = 0; k < ADNN_CONVBWD_PRV_TOPDF_H; ++k)
						{

							for(int kl = 0; kl < ADNN_CONV_KERNEL_SZ; ++kl)
							{

								for(int l = 0; l <  ADNN_CONVBWD_PRV_TOPDF_W; ++l)
								{
									prv_accum[kl] += prv_top_df[k][l] * prv_bot[k * ADNN_CONV_STRIDE][kl + l * ADNN_CONV_STRIDE];
#if 0
			 if ( c==0 && o == 0   && kl == 1 && kk == 1 && prv_top_df[k][l] * prv_bot[k][kl+ l] != 0

			 )
			 {
				printf("K: con2: %d %d %d %d %d   %11.9f %11.9f %11.9f\n", b, lcl_id0,  l, lcl_id1,  k,  prv_accum[kl],  prv_top_df[k][l], prv_bot[k][kl+l]);
			 }
#endif

								}

							}

						}
						for(int kl = 0; kl < ADNN_CONV_KERNEL_SZ; ++kl)
						{

// move into prefix sum area
// shift along with the kernel position
//							weights_df_psum[kk*ADNN_CONV_KERNEL_SZ* (ADNN_CONVBWD_GROUP_SZ >> 1) + lcl_id * ADNN_CONV_KERNEL_SZ + kl] = prv_accum[kl];
							weights_df_psum[kk*ADNN_CONV_KERNEL_SZ + lcl_id * ADNN_CONV_KERNEL_SZ + kl]  = prv_accum[kl];
						
						}


						barrier(CLK_LOCAL_MEM_FENCE);
// make the first step in part sum
#define ADNN_CONVBWD_N_PSUM_STEPS 1
						for(int s = (ADNN_CONVBWD_GROUP_SZ >> 1); s > 0/*(ADNN_CONVBWD_GROUP_SZ >> (1+ADNN_CONVBWD_N_PSUM_STEPS))*/; s >>= 1)
						{
							if ( lcl_id <  s)
							{
								for(int kl = 0; kl < ADNN_CONV_KERNEL_SZ; ++kl)
								{
									    prv_accum[kl] += weights_df_psum[kk*ADNN_CONV_KERNEL_SZ + (lcl_id + s) * ADNN_CONV_KERNEL_SZ + kl];
	//									weights_df_psum[kk*ADNN_CONV_KERNEL_SZ* (ADNN_CONVBWD_GROUP_SZ >> 1) + lcl_id * ADNN_CONV_KERNEL_SZ + kl + s] = prv_accum[kl];
										weights_df_psum[kk*ADNN_CONV_KERNEL_SZ + lcl_id * ADNN_CONV_KERNEL_SZ + kl] = prv_accum[kl];
								}
							}
							barrier(CLK_LOCAL_MEM_FENCE);
						}

// move up						
						for(int k = 0; k < ADNN_CONV_N_PRV_VERT - 1; ++k)
						{
							for(int l = 0; l <  ADNN_CONV_N_PRV_HORIZ; ++l)
							{
								prv_bot[k][l] = prv_bot[k+1][l];

							}
						}

						int lcl_off_v = (ADNN_CONV_N_PRV_VERT + j + kk) * ADNN_CONVBWD_BOT_DATA_WIDTH;
						for(int l = 0; l <   ADNN_CONV_N_PRV_HORIZ; ++l)
						{
							prv_bot[ADNN_CONV_N_PRV_VERT-1][l] = bot_data[lcl_off_v + i + l];

						}


					}


#if 0
			 if ( c==0 && o == 2   && lcl_id == 0

			 )
			 {
				printf("K: top_df sum: %d %11.9f\n", b, weights_df_psum[0]);
			 }
#endif

					int accum_loop = ADNN_CONV_KERN_AREA_SZ;

#if ADNN_CONV_BIAS
// do bias only with  the last channel
					if ( c == ADNN_CONV_N_INPUTS - 1 )
					{
// bias
//					weights_df_psum[ADNN_CONV_KERN_AREA_SZ + lcl_id] = 0;
//					barrier(CLK_LOCAL_MEM_FENCE);
// private sum
						_FLOAT prv_sum = 0;
						for(int k = 0; k < ADNN_CONVBWD_PRV_TOPDF_H; ++k)
						{
							for(int l = 0; l <  ADNN_CONVBWD_PRV_TOPDF_W; ++l)
							{
								prv_sum += prv_top_df[k][l];
							}
						}

// move into prefix sum area
						weights_df_psum[ADNN_CONV_KERN_AREA_SZ + lcl_id] = prv_sum;
						barrier(CLK_LOCAL_MEM_FENCE);
// sum
						for(int s = (ADNN_CONVBWD_GROUP_SZ >> 1); s > 0; s >>= 1)
						{
							if ( lcl_id <  s)
							{
								prv_sum += weights_df_psum[ADNN_CONV_KERN_AREA_SZ + lcl_id + s];
// the last assignemnt is total sum
								weights_df_psum[ADNN_CONV_KERN_AREA_SZ + lcl_id] = prv_sum;
							}
							barrier(CLK_LOCAL_MEM_FENCE);
					
						}

						accum_loop = ADNN_CONV_KERN_AREA_SZ + 1;
					}

#endif
// accumulate over all tiles and batchh

					for(int k = lcl_id; k < accum_loop; k += ADNN_CONVBWD_GROUP_SZ)
					{
						weights_df_sum[k] += weights_df_psum[k];
					}

					barrier(CLK_LOCAL_MEM_FENCE);



				} // horiz tiles

			}  // vertical tiles

		} // batch size
// write out
		int weight_df_off = b* ADNN_CONV_WEIGHTS_CHANNEL_STRIDE + o * ADNN_CONV_WEIGHTS_STRIDE + c * ADNN_CONV_KERN_AREA_SZ;
		int out_loop = ( c == ADNN_CONV_N_INPUTS - 1 ) ? ADNN_CONV_KERN_AREA_SZ + 1 : ADNN_CONV_KERN_AREA_SZ;
		for(int k = lcl_id; k < out_loop; k+= ADNN_CONVBWD_GROUP_SZ)
		{
			 weights_df[weight_df_off + k] = weights_df_sum[k];

#if 0
			 if ( weights_df[weight_df_off + k] != 0 && c==0 && o == 2 /*weight_df_off + k == 1602 || weight_df_off + k == 1603 || weight_df_off + k == 1604 || weight_df_off + k == 1605 c==0 && o == 2 && k == 3*/)
			 {
				printf("K:out: %d %d %d %d  %d %9.8f\n", b, o, c, k, weight_df_off + k, weights_df[weight_df_off + k]);
			 }
#endif
		}

}




#define ADNN_CONVBSUM_GRP_SZ1 1
#define ADNN_CONVBSUM_GRP_SZ2 1


__attribute__((reqd_work_group_size(ADNN_CONVBSUM_GRP_SZ0,ADNN_CONVBSUM_GRP_SZ1,ADNN_CONVBSUM_GRP_SZ2)))
__kernel void aDNNConvBwd_wrt_W_Bsum(
       const __global _FLOAT * weights_df_t,
       __global _FLOAT * weights_df
	   )
{
	int off = get_global_id(0);
	_FLOAT sum = 0;
	for(int b = 0; b < ADNN_CONV_BATCH_SZ; ++b)
	{
		sum += weights_df_t[off + b* ADNN_CONV_WEIGHTS_CHANNEL_STRIDE];
	}
	weights_df[off] = sum;
}


#endif

#define ADNN_CONV_GROUP_SZ2 1
#define ADNN_CONV_GRP_DATA_WIDTH (ADNN_CONV_GROUP_SZ0 *ADNN_CONV_N_HORIZ_OUT_PIX + ADNN_CONV_KERNEL_SZ + ADNN_CONV_STRIDE - 1) / ADNN_CONV_STRIDE
#define ADNN_CONV_GRP_DATA_HEIGHT  (ADNN_CONV_GROUP_SZ1 * ADNN_CONV_N_VERT_OUT_PIX + ADNN_CONV_KERNEL_SZ + ADNN_CONV_STRIDE - 1) / ADNN_CONV_STRIDE
#define ADNN_CONV_GROUP_SZ (ADNN_CONV_GROUP_SZ1 * ADNN_CONV_GROUP_SZ0)
#define ADNN_CONV_KERN_AREA_SZ (ADNN_CONV_KERNEL_SZ * ADNN_CONV_KERNEL_SZ)

// here we do the following
// global workload is batch size * number input channels
// the kernel generates up to ADNN_CONV_N_OUTS input channel (tile)
// for each output channel it generates 1 partial bot_diff pixel by convolving (filter) weights (of the input channel ) with top_diff pixels
// the resulting bot_diff pixels are accumulation of the above.
// STRIDE > 1 is a separate kernel
// TO DO: REMOVE invers here !!!


__attribute__((reqd_work_group_size(ADNN_CONV_GROUP_SZ0,ADNN_CONV_GROUP_SZ1,ADNN_CONV_GROUP_SZ2)))
__kernel void aDNNConvBwd_wrt_B(
       const __global _FLOAT * top_df,
       const __global _FLOAT * weights,
       __global _FLOAT * bot_df,
	   _FLOAT padding_val
	   )
{
		
		__local _FLOAT top_df_data[ADNN_CONV_GRP_DATA_WIDTH * ADNN_CONV_GRP_DATA_HEIGHT];
		__local _FLOAT weights_data[ADNN_CONV_N_OUTS * ADNN_CONV_KERN_AREA_SZ]; // + bias

	
		int x = get_group_id(0) * ADNN_CONV_GROUP_SZ0 * ADNN_CONV_N_HORIZ_OUT_PIX;
		int y = get_group_id(1) * ADNN_CONV_GROUP_SZ1 * ADNN_CONV_N_VERT_OUT_PIX;
		int lcl_id0 = get_local_id(0);
		int lcl_id1 = get_local_id(1);
		int lcl_id = (lcl_id1 << ADNN_CONV_GROUP_LG2SZ0) + lcl_id0;
		int cb = get_global_id(2); // channels * batch_sz
		int c = (int)(float)cb / (float)ADNN_CONV_BATCH_SZ;
		int b = cb - c * ADNN_CONV_BATCH_SZ;

		int top_df_off = b * ADNN_CONVBWD_TOPDF_BATCH_STRIDE;
	

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


		int top_x = (x + ADNN_CONV_PAD - ADNN_CONV_KERNEL_SZ)/ ADNN_CONV_STRIDE + 1;
		int top_y = (y + ADNN_CONV_PAD - ADNN_CONV_KERNEL_SZ) / ADNN_CONV_STRIDE + 1;

// over all outputs
		for( int o = 0; o < ADNN_CONV_N_OUTPUTS; o++, top_df_off += ADNN_CONVBWD_TOPDF_CHANNEL_STRIDE)
		{
			int gbl_weights_off = o * ADNN_CONV_WEIGHTS_STRIDE  + c * ADNN_CONV_KERN_AREA_SZ * ADNN_CONV_N_OUTS; 
			int lcl_weights_off = 0;
// get weights per channel for every new output
			for(int k = 0; k < ADNN_CONV_N_OUTS * ADNN_CONV_KERN_AREA_SZ; k++)
			{
				weights_data[lcl_weights_off + lcl_id + k] = weights[gbl_weights_off + lcl_id + k];
			}


// load top_diff with padding
			for( int b_j = lcl_id1; b_j < ADNN_CONV_GRP_DATA_HEIGHT; b_j += ADNN_CONV_GROUP_SZ1)
			{	
				int y_act = (top_y + b_j);
				bool invisibleY = (y_act < 0) || (y_act >= ADNN_CONV_TOP_HEIGHT);

				int y_off = y_act * ADNN_CONVBWD_TOPDF_STRIDE;

				int lcl_off_v = b_j * ADNN_CONV_GRP_DATA_WIDTH;

				for(int b_i = lcl_id0; b_i < ADNN_CONV_GRP_DATA_WIDTH; b_i += ADNN_CONV_GROUP_SZ0)
				{
					int x_act = (top_x + b_i);
					bool invisibleX = (x_act < 0) || (x_act >= ADNN_CONV_TOP_WIDTH);

					_FLOAT top_df_val = top_df[top_df_off + y_off + x_act];

					top_df_val = (invisibleX || invisibleY)? 0 : top_df_val;
								
					top_df_data[lcl_off_v + b_i] = top_df_val;
				}
			}



			barrier(CLK_LOCAL_MEM_FENCE);


// calculate bot_diff per each output channel

			#define ADNN_CONV_N_PRV_HORIZ (ADNN_CONV_KERNEL_SZ + ADNN_CONV_N_HORIZ_OUT_PIX + ADNN_CONV_STRIDE -1) / ADNN_CONV_STRIDE

			#define ADNN_CONV_N_PRV_VERT (ADNN_CONV_N_VERT_OUT_PIX + ADNN_CONV_STRIDE - 1) / ADNN_CONV_STRIDE
			_FLOAT data_stage[ADNN_CONV_N_PRV_VERT][ADNN_CONV_N_PRV_HORIZ];

			int bot_y = (y + lcl_id1 * ADNN_CONV_N_VERT_OUT_PIX);
			int bot_x = (x + lcl_id0 * ADNN_CONV_N_HORIZ_OUT_PIX);
			int inv_y = (bot_y + ADNN_CONV_PAD - ADNN_CONV_KERNEL_SZ) / ADNN_CONV_STRIDE + 1;
			int inv_x = (bot_x + ADNN_CONV_PAD - ADNN_CONV_KERNEL_SZ) / ADNN_CONV_STRIDE + 1;

			int lcl_y = inv_y - top_y;
			int lcl_x = inv_x - top_x;


	
			for(int j = 0; j < ADNN_CONV_N_PRV_VERT; ++j)
			{	

				for(int i = 0; i < ADNN_CONV_N_PRV_HORIZ; i++)
				{

					data_stage[j][i] = top_df_data[(lcl_y + j)*ADNN_CONV_GRP_DATA_WIDTH + (lcl_x + i)];

				}
			}

			
			int start_inv_y = inv_y ;

			int oy = y + lcl_id1 * ADNN_CONV_N_VERT_OUT_PIX;
			int ox = x + lcl_id0 * ADNN_CONV_N_HORIZ_OUT_PIX;		

			for( int k = 0; k < ADNN_CONV_KERNEL_SZ; k+= ADNN_CONV_STRIDE)
			{	
				start_inv_y = (bot_y + k + ADNN_CONV_PAD - ADNN_CONV_KERNEL_SZ) / ADNN_CONV_STRIDE + 1;

				lcl_weights_off = 0;
				for(int ko = 0; ko < ADNN_CONV_N_OUTS; ko++, lcl_weights_off += ADNN_CONV_KERN_AREA_SZ)
				{
					for(int j = 0; j < ADNN_CONV_N_VERT_OUT_PIX; j++)
					{

						int c_inv_y = (bot_y + j + k + ADNN_CONV_PAD - ADNN_CONV_KERNEL_SZ) / ADNN_CONV_STRIDE + 1;
						int prv_j = c_inv_y - start_inv_y;

						for(int i = 0; i < ADNN_CONV_N_HORIZ_OUT_PIX; i++)
						{
// weights in reverse order
							for(int l = 0; l < ADNN_CONV_KERNEL_SZ; l+=ADNN_CONV_STRIDE)
							{

								int c_inv_x = (bot_x + i + l - ADNN_CONV_KERNEL_SZ + ADNN_CONV_PAD) / ADNN_CONV_STRIDE + 1;
								int prv_i = c_inv_x - inv_x;

								accum[ko][j][i] += data_stage[prv_j][prv_i] * weights_data[lcl_weights_off + (ADNN_CONV_KERNEL_SZ - 1 - k)*ADNN_CONV_KERNEL_SZ + ADNN_CONV_KERNEL_SZ - 1 - l]; 
#if 0									
								if (c == 0 && ko == 0 && oy + j == 0 && ox + i == 0 && data_stage[prv_j][prv_i] * weights_data[lcl_weights_off + (ADNN_CONV_KERNEL_SZ - 1 - k)*ADNN_CONV_KERNEL_SZ + ADNN_CONV_KERNEL_SZ - 1 - l] != 0)
								{
									printf("K:conv: %d %d %d  %11.9f %11.9f %11.9f %11.9f\n",
									ox + i, oy + j,
									o,
									 accum[ko][j][i],	
									 weights_data[lcl_weights_off + (ADNN_CONV_KERNEL_SZ - 1 - k)*ADNN_CONV_KERNEL_SZ + ADNN_CONV_KERNEL_SZ - 1 - l],
									 data_stage[prv_j][prv_i],									 							 
									data_stage[prv_j][prv_i] * weights_data[lcl_weights_off + (ADNN_CONV_KERNEL_SZ - 1 - k)*ADNN_CONV_KERNEL_SZ + ADNN_CONV_KERNEL_SZ - 1 - l]									 
									 );
								}
#endif
																			
							} // for(int l = 0; l < ADNN_CONV_KERNEL_SZ; ++l)
						} // for(int i = 0; i < ADNN_CONV_N_HORIZ_OUT_PIX; i++)
					} // for(int j = 0; j < ADNN_CONV_N_VERT_OUT_PIX; j++)

				} // for(int ko = 0; ko < ADNN_CONV_N_OUTS; ko++, lcl_weights_off += ADNN_CONV_KERN_AREA_SZ)

				for(int j = 0; j < ADNN_CONV_N_PRV_VERT - 1; j++)
				{
					for(int i = 0; i < ADNN_CONV_N_PRV_HORIZ; i++)
					{
						data_stage[j][i] = data_stage[j+1][i];
					}
				}

				for(int l = 0; l < ADNN_CONV_N_PRV_HORIZ; ++l)
				{
					data_stage[ADNN_CONV_N_PRV_VERT - 1][l] = top_df_data[(k+ADNN_CONV_N_PRV_VERT + lcl_y)*ADNN_CONV_GRP_DATA_WIDTH + (l+lcl_x)];

				}

			}


		}  // per all output channels

		for(int ko = 0; ko < ADNN_CONV_N_OUTS; ko++)
		{
			int out_y = y + lcl_id1 * ADNN_CONV_N_VERT_OUT_PIX;
			int out_x = x + lcl_id0 * ADNN_CONV_N_HORIZ_OUT_PIX;


			int bot_df_off = b * ADNN_CONVBWD_BOTDF_BATCH_STRIDE + (c  * ADNN_CONV_N_OUTS + ko) * ADNN_CONVBWD_BOTDF_CHANNEL_STRIDE + out_y * ADNN_CONVBWD_BOTDF_STRIDE + out_x;
			for(int j = 0; j < ADNN_CONV_N_VERT_OUT_PIX; j++, bot_df_off += ADNN_CONVBWD_BOTDF_STRIDE, out_y++)
			{
				int out_x2 = out_x;
				for(int i = 0; i < ADNN_CONV_N_HORIZ_OUT_PIX; i++, out_x2++)
				{	
					if (out_y < ADNN_CONV_BOT_HEIGHT && out_x2 < ADNN_CONV_BOT_WIDTH)
					{

						_FLOAT out_val = accum[ko][j][i];
						bot_df[bot_df_off + i] = out_val;
					}
				}
			}
		}
}



__attribute__((reqd_work_group_size(ADNN_CONV_GROUP_SZ0,ADNN_CONV_GROUP_SZ1,ADNN_CONV_GROUP_SZ2)))
__kernel void aDNNConvBwd_wrt_B2(
       const __global _FLOAT * top_df,
       const __global _FLOAT * weights,
       __global _FLOAT * bot_df,
	   _FLOAT padding_val
	   )
{
		
		__local _FLOAT top_df_data[ADNN_CONV_GRP_DATA_WIDTH * ADNN_CONV_GRP_DATA_HEIGHT];
		__local _FLOAT weights_data[ADNN_CONV_N_OUTS * ADNN_CONV_KERN_AREA_SZ]; // + bias

	
		int x = get_group_id(0) * ADNN_CONV_GROUP_SZ0 * ADNN_CONV_N_HORIZ_OUT_PIX;
		int y = get_group_id(1) * ADNN_CONV_GROUP_SZ1 * ADNN_CONV_N_VERT_OUT_PIX;
		int lcl_id0 = get_local_id(0);
		int lcl_id1 = get_local_id(1);
		int lcl_id = (lcl_id1 << ADNN_CONV_GROUP_LG2SZ0) + lcl_id0;
		int cb = get_global_id(2); // channels * batch_sz
		int c = (int)(float)cb / (float)ADNN_CONV_BATCH_SZ;
		int b = cb - c * ADNN_CONV_BATCH_SZ;

		int top_df_off = b * ADNN_CONVBWD_TOPDF_BATCH_STRIDE;
	

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

#if 0
		int top_x = (x - ADNN_CONV_PAD);
		int top_y = (y - ADNN_CONV_PAD);
		top_x += (top_x < 0) ? -ADNN_CONV_STRIDE + 1 :ADNN_CONV_STRIDE - 1;
		top_y += (top_y < 0) ? -ADNN_CONV_STRIDE + 1 :ADNN_CONV_STRIDE - 1;

		top_x /= ADNN_CONV_STRIDE;
		top_y /= ADNN_CONV_STRIDE;
#endif
// over all outputs
		for( int o = 0; o < ADNN_CONV_N_OUTPUTS; o++, top_df_off += ADNN_CONVBWD_TOPDF_CHANNEL_STRIDE)
		{
			int gbl_weights_off = o * ADNN_CONV_WEIGHTS_STRIDE  + c * ADNN_CONV_KERN_AREA_SZ * ADNN_CONV_N_OUTS; 
			int lcl_weights_off = 0;
// get weights per channel for every new output
			for(int k = 0; k < ADNN_CONV_N_OUTS * ADNN_CONV_KERN_AREA_SZ; k++)
			{
				weights_data[lcl_weights_off + lcl_id + k] = weights[gbl_weights_off + lcl_id + k];
			}

#if 0
// load top_diff with padding
			for( int b_j = lcl_id1; b_j < ADNN_CONV_GRP_DATA_HEIGHT; b_j += ADNN_CONV_GROUP_SZ1)
			{	
				int y_act = (((top_y < 0 )? 0 : top_y) + b_j);
				bool invisibleY = (y_act >= ADNN_CONV_TOP_HEIGHT);

				int y_off = y_act * ADNN_CONVBWD_TOPDF_STRIDE;

				int lcl_off_v = b_j * ADNN_CONV_GRP_DATA_WIDTH;

				for(int b_i = lcl_id0; b_i < ADNN_CONV_GRP_DATA_WIDTH; b_i += ADNN_CONV_GROUP_SZ0)
				{
					int x_act = (((top_x < 0 )? 0 : top_x) + b_i);
					bool invisibleX = (x_act >= ADNN_CONV_TOP_WIDTH);

					_FLOAT top_df_val = top_df[top_df_off + y_off + x_act];

					top_df_val = (invisibleX || invisibleY)? 0 : top_df_val;
								
					top_df_data[lcl_off_v + b_i] = top_df_val;
				}
			}



			barrier(CLK_LOCAL_MEM_FENCE);

#endif
// calculate bot_diff per each output channel

			#define ADNN_CONV_N_PRV_HORIZ (ADNN_CONV_KERNEL_SZ - 1 + ADNN_CONV_N_HORIZ_OUT_PIX + ADNN_CONV_STRIDE -1) / ADNN_CONV_STRIDE
			#define ADNN_CONV_N_PRV_VERT (ADNN_CONV_KERNEL_SZ - 1 + ADNN_CONV_N_VERT_OUT_PIX + ADNN_CONV_STRIDE - 1) / ADNN_CONV_STRIDE
			#define ADNN_CONV_N_ACCUM_STEPS_HORZ (ADNN_CONV_STRIDE / ADNN_CONV_STRIDE)
			#define ADNN_CONV_N_ACCUM_STEPS_VERT (ADNN_CONV_STRIDE / ADNN_CONV_STRIDE)


			_FLOAT data_stage[ADNN_CONV_N_PRV_VERT][ADNN_CONV_N_PRV_HORIZ];

			int bot_y = (y + lcl_id1 * ADNN_CONV_N_VERT_OUT_PIX );
			int bot_x = (x + lcl_id0 * ADNN_CONV_N_HORIZ_OUT_PIX );
			int inv_y = (bot_y - ADNN_CONV_PAD);
			int inv_x = (bot_x - ADNN_CONV_PAD);
			inv_y += ((inv_y < 0 ) ? -ADNN_CONV_STRIDE + 1 : ADNN_CONV_STRIDE - 1) ;
			inv_x += ((inv_x < 0 ) ? -ADNN_CONV_STRIDE + 1 : ADNN_CONV_STRIDE - 1) ;
			inv_y /= ADNN_CONV_STRIDE;
			inv_x /= ADNN_CONV_STRIDE;



//			int lcl_y = inv_y - top_y;
//			int lcl_x = inv_x - top_x;


	
			for(int j = 0; j < ADNN_CONV_N_PRV_VERT; ++j)
			{	

				for(int i = 0; i < ADNN_CONV_N_PRV_HORIZ; i++)
				{

					_FLOAT top_df_val = top_df[top_df_off + (inv_y + j) *  ADNN_CONVBWD_TOPDF_STRIDE + inv_x + i];
					top_df_val = ((inv_y + j) < 0  || (inv_y + j) >= ADNN_CONV_TOP_HEIGHT || (inv_x + i) < 0 || (inv_x + i) >= ADNN_CONV_TOP_HEIGHT) ? 0 : top_df_val;
					data_stage[j][i] = top_df_val;

				}
			}

			
			int oy = y + lcl_id1 * ADNN_CONV_N_VERT_OUT_PIX;
			int ox = x + lcl_id0 * ADNN_CONV_N_HORIZ_OUT_PIX;	
			
			for(int j = 0; j < ADNN_CONV_N_VERT_OUT_PIX; j++)
			{
// top most contributing posion
				int tm_cp = (oy +j - ADNN_CONV_PAD);
				int tm_sp = (tm_cp < 0) ? (tm_cp -ADNN_CONV_STRIDE + 1) : (tm_cp +ADNN_CONV_STRIDE - 1);
				tm_sp /= ADNN_CONV_STRIDE;

				for(int i = 0; i < ADNN_CONV_N_HORIZ_OUT_PIX; i++)
				{
// left most contributing posion
					int lm_cp = (ox +i -ADNN_CONV_PAD );
					int lm_sp = (lm_cp < 0) ? (lm_cp -ADNN_CONV_STRIDE + 1) : (lm_cp +ADNN_CONV_STRIDE - 1);
					lm_sp /= ADNN_CONV_STRIDE;

					lcl_weights_off = 0;
					for(int ko = 0; ko < ADNN_CONV_N_OUTS; ko++, lcl_weights_off += ADNN_CONV_KERN_AREA_SZ)
					{
// weights in reverse order
// vertical contributing position
						for(int v_sp = tm_sp, k0 = 0; k0 < (ADNN_CONV_KERNEL_SZ + ADNN_CONV_STRIDE - 1)/ ADNN_CONV_STRIDE; ++v_sp, ++k0 )
						{

							int k = (v_sp * ADNN_CONV_STRIDE - oy  - j +  ADNN_CONV_PAD);
							bool invisV = (k >= ADNN_CONV_KERNEL_SZ);
							int prv_j = v_sp - inv_y;
// horiz contributing pos
							for(int h_sp = lm_sp, l0 = 0; l0 < (ADNN_CONV_KERNEL_SZ + ADNN_CONV_STRIDE - 1)/ ADNN_CONV_STRIDE; ++h_sp, ++l0)
							{

								int l = (h_sp * ADNN_CONV_STRIDE - ox - i +  ADNN_CONV_PAD);
								bool invisH = (l >= ADNN_CONV_KERNEL_SZ);
								int prv_i = h_sp - inv_x;

								_FLOAT add_val = data_stage[prv_j][prv_i] * weights_data[lcl_weights_off + (ADNN_CONV_KERNEL_SZ - 1 - k)*ADNN_CONV_KERNEL_SZ + ADNN_CONV_KERNEL_SZ - 1 - l];

								add_val = (invisV || invisH) ? 0 : add_val;

								accum[ko][j][i] += add_val; 
#if 0									
								if (c == 0 && ko == 0 && oy + j == 0 && ox + i == 2 && add_val != 0)
								{
									printf("K:cv: %d %d %d %d %d %d %d %d  %11.9f %11.9f %11.9f %11.9f\n",
									o,
									h_sp,
									v_sp,
									inv_x,
									prv_i,
									prv_j,
									ADNN_CONV_KERNEL_SZ - 1 - l,
									(ADNN_CONV_KERNEL_SZ - 1 - k),
									 accum[ko][j][i],	
									 weights_data[lcl_weights_off + (ADNN_CONV_KERNEL_SZ - 1 - k)*ADNN_CONV_KERNEL_SZ + ADNN_CONV_KERNEL_SZ - 1 - l],
									 data_stage[prv_j][prv_i],									 							 
									add_val									 
									 );
								}
#endif

	

							}
						}
					}
				}
			}											



		}  // per all output channels

		for(int ko = 0; ko < ADNN_CONV_N_OUTS; ko++)
		{
			int out_y = y + lcl_id1 * ADNN_CONV_N_VERT_OUT_PIX;
			int out_x = x + lcl_id0 * ADNN_CONV_N_HORIZ_OUT_PIX;


			int bot_df_off = b * ADNN_CONVBWD_BOTDF_BATCH_STRIDE + (c  * ADNN_CONV_N_OUTS + ko) * ADNN_CONVBWD_BOTDF_CHANNEL_STRIDE + out_y * ADNN_CONVBWD_BOTDF_STRIDE + out_x;
			for(int j = 0; j < ADNN_CONV_N_VERT_OUT_PIX; j++, bot_df_off += ADNN_CONVBWD_BOTDF_STRIDE, out_y++)
			{
				int out_x2 = out_x;
				for(int i = 0; i < ADNN_CONV_N_HORIZ_OUT_PIX; i++, out_x2++)
				{	
					if (out_y < ADNN_CONV_BOT_HEIGHT && out_x2 < ADNN_CONV_BOT_WIDTH)
					{

						_FLOAT out_val = accum[ko][j][i];
						bot_df[bot_df_off + i] = out_val;
					}
				}
			}
		}
}

