/*
 * Copyright (c) 2016 AMD Inc.
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
//ADNN_GRP_SZ0              group size in dim 0
//ADNN_GRP_SZ1				group size in dim 1
//ADNN_GRP_LG2SZ0           log2 group size in dim 0
//ADNN_GRP_LG2SZ1           log2 group size in dim 1
//ADNN_GRP_SZ               n of wk-item in the group
//ADNN_N_IN_CHNL			total number of input channels
//ADNN_IN_CHNL_SZ			input channel size
//ADNN_LCL_N_IN_CHNLS		n of localy kept input channels
//ADNN_LCL_LOG2N_IN_CHNLS	log2 of n of localy kept input channels
//ADNN_IN_WIDTH				input width in NCHW layout
//ADNN_IN_HEIGHT			input height stride in NCHW layout
//ADNN_IN_STRIDE			input stride in NCHW layout
//ADNN_IN_CHNL_STRIDE       input channel stride in NCHW layout
//ADNN_IN_BATCH_STRIDE      input batch stride in NCHW layout
//ADNN_BATCH_SZ		        batch szie
//ADNN_FLTR_SZ0             filter 0 dim size
//ADNN_FLTR_PAD_SZ0				filter 0 dim pad
//ADNN_FLTR_STRIDE0			filter 0 dim stride
//ADNN_FLTR_SZ1             filter 1 dim size
//ADNN_FLTR_PAD_SZ1				filter 1 dim pad
//ADNN_FLTR_STRIDE1			filter 1 dim stride
//ADNN_IN_CHNL_LOOP         main input channel loop
//ADNN_WEI_SZ               size of weight buffer
//ADNN_OUT_WIDTH			output width in NCHW layout
//ADNN_OUT_HEIGHT			output height stride in NCHW layout
//ADNN_OUT_STRIDE			output stride in NCHW layout
//ADNN_OUT_CHNL_STRIDE      output channel stride in NCHW layout
//ADNN_OUT_BATCH_STRIDE     output batch stride in NCHW layout
//ADNN_OUT_LOG2WIDTH		log2 of output width in NCHW layout
//ADNN_OUT_LOG2HEIGHT		log2 of output height stride in NCHW layout
//ADNN_N_OUT_PIX_SZ0        n output pixel per wk item in 0 dim
//ADNN_N_OUT_PIX_SZ1		n output pexels per wk item in 1 dim
//ADNN_LOG2N_OUT_PIX_SZ0    log2 of n output pixel per wk item in 0 dim
//ADNN_LOG2N_OUT_PIX_SZ1	log2 of n output pexels per wk item in 1 dim
//ADNN_OUT_PROC_SZ0         size of output processing group in 0 dim
//ADNN_OUT_PROC_SZ1         size of output processing group in 1 dim
//ADNN_OUT_PROC_LOG2SZ0     log2 of size of output processing group in 0 dim
//ADNN_OUT_PROC_LOG2SZ1     log2 of size of output processing group in 1 dim
//ADNN_OUT_TILEPROC_SZ0         size of output tile processing group in 0 dim, 1 tile per 1 outpur channel. 
//ADNN_OUT_TILEPROC_SZ1         size of output tile processing group in 1 dim
//ADNN_OUT_TILEPROC_LOG2SZ0     log2 of size of tile output processing group in 0 dim,  ADNN_OUT_TILEPROC_LOG2SZ0 + ADNN_LOG2N_OUT_PIX_SZ0 = ADNN_OUT_PROC_LOG2SZ0
//ADNN_OUT_TILEPROC_LOG2SZ1     log2 of size of til output processing group in 1 dim
//ADNN_N_OUT_CHNLS			total number of output channels
//ADNN_LCL_N_OUT_CHNLS		n of localy kept output channels
//ADNN_WEIGHTS_STRIDE			weights stride
//ADNN_OUT_TILE_SZ0         size of output tile (dim 0)= size of input tile
//ADNN_OUT_TILE_SZ1         size of output tile (dim 1) = size of input tile
//ADNN_OUT_TILE_LOG2SZ0         log2 size of output tile (dim 0)= size of input tile
//ADNN_OUT_TILE_LOG2SZ1         log2 size of output tile (dim 1) = size of input tile
//ADNN_N_OUT_TILES0         n output tiles (dim 0)= size of input tile
//ADNN_N_OUT_TILES1         n output tiles (dim 1) = size of input tile
//ADNN_LOG2N_OUT_TILES0         log2 n output tiles (dim 0)= size of input tile
//ADNN_LOG2N_OUT_TILES1         log2 n output tiles (dim 1) = size of input tile
//ADNN_N_READ_PROC_SZ0         size of read processors (dim 0)= size of input tile
//ADNN_N_READ_PROC_SZ1         size of read processors (dim 1) = size of input tile
//ADNN_N_READ_PROC_LOG2SZ0     log2 size of read processors (dim 0)= size of input tile
//ADNN_N_READ_PROC_LOG2SZ1     log2 size of read processors (dim 1) = size of input tile
//ADNN_LCL_WEIGHTS          weights are in local memory


// IMPROVEMENTS
// merage with fisrt kernel
// input double buffering: removes one barrier
// compute the first line but do not save it in the bot_staging: reduce the size of the bot_stading by one line, saves reg 

#define ADNN_GRP_SZ2 1


#if ADNN_BIG == 0
#define ADNN_IN_SIZE (ADNN_IN_WIDTH * ADNN_IN_HEIGHT)
#define ADNN_LCL_STRIDE (ADNN_IN_WIDTH + ADNN_FLTR_PAD_SZ0 * 2)
#define ADNN_LCL_HEIGHT (ADNN_IN_HEIGHT + ADNN_FLTR_PAD_SZ1 * 2)
#else
#define ADNN_IN_SIZE ((ADNN_OUT_TILE_SZ0 + ADNN_FLTR_PAD_SZ0 *2) * (ADNN_OUT_TILE_SZ1 + ADNN_FLTR_PAD_SZ1 *2))
#define ADNN_LCL_STRIDE (ADNN_OUT_TILE_SZ0 + ADNN_FLTR_PAD_SZ0 *2)
#define ADNN_LCL_HEIGHT (ADNN_OUT_TILE_SZ1 + ADNN_FLTR_PAD_SZ1 *2)
#endif 
#define ADNN_LCL_SZ (ADNN_LCL_STRIDE * ADNN_LCL_HEIGHT)
#define ADNN_PRV_TILE_H (ADNN_FLTR_SZ0 + ADNN_N_OUT_PIX_SZ0 -1 )
#define ADNN_PRV_TILE_V (ADNN_N_OUT_PIX_SZ1) 
#define ADNN_FLTR_SZ (ADNN_FLTR_SZ0 * ADNN_FLTR_SZ1)

inline void calculateXYPos(int linPos, int width, int *x, int *y)
{
	(*y) = linPos /width;
	(*x) = linPos - (*y) * width; 
}

inline int calculateOffset(int stride, int x, int y)
{
	int ret = y * stride + x;
	return(ret);
}

inline void readDataElem(__local _FLOAT *lcl_data, __global _FLOAT * gbl_data, int linPos, int gbl_width, int gbl_stride, int gbl_base, int lcl_stride, int lcl_base)
{
	int x, y;
	calculateXYPos(linPos, gbl_width, &x, &y);
	int gbl_off = calculateOffset(gbl_stride, x, y);
	gbl_off += gbl_base;
	int lcl_off = calculateOffset(lcl_stride, x, y);
// shift along y and x by pad size
	lcl_off += lcl_base + ADNN_FLTR_PAD_SZ0;
	lcl_data[lcl_off] = gbl_data[gbl_off];
}

// split the group into several input vector processors
// each processor reads its own input channel
inline void readData(__local _FLOAT *lcl_data, __global _FLOAT * gbl_data, int lcl_p_id, int lcl_p_stride, int size, int gbl_width, int gbl_stride, int gbl_base, int lcl_stride, int lcl_base)
{
	
	for(int i = lcl_p_id; i < size; i+= lcl_p_stride)
	{
		readDataElem(lcl_data, gbl_data, i, gbl_width, gbl_stride, gbl_base, lcl_stride, lcl_base);
	}

}
//#if ADNN_WEI_SZ < (1 << 14)
//inline void readWeights(__local _FLOAT *lcl_data, __constant _FLOAT * gbl_data, int p_id, int p_stride, int size, int lcl_stride, int gbl_stride)
//#else
inline void readWeights(__local _FLOAT *lcl_data, const __global _FLOAT * gbl_data, int p_id, int p_stride, int size, int lcl_stride, int gbl_stride)
//#endif
{
	for(int i = p_id; i < size; i+= p_stride)
	{
		int lcl_out = i/lcl_stride;
		int lcl_we = i - mul24(lcl_out, lcl_stride);

		lcl_data[i] = gbl_data[mad24(lcl_out,gbl_stride,lcl_we)];
	}
}

inline void readDataTile(__local _FLOAT *lcl_data, __global _FLOAT * gbl_data,
						int tile_y, int tile_x,
						int gbl_stride, int gbl_base,
						int lcl_stride, int lcl_base,
						int gbl_height, int gbl_width,
						int lcl_height, int lcl_width,
						int lcl_id1, int lcl_id0,
						int lcl_grp_sz1, int lcl_grp_sz0,
						int fltr_pad1, int fltr_pad0,
						_FLOAT padding_val)
{
			for( int j = lcl_id1; j < lcl_height; j += lcl_grp_sz1)
			{	
				int y_act = (j - fltr_pad1);
				bool invisibleY = (tile_y + y_act < 0) || (tile_y + y_act >= gbl_height);

				int y_gbl_off = y_act * gbl_stride;

				int y_lcl_off = j * lcl_stride;

				for(int i = lcl_id0; i < lcl_width; i += lcl_grp_sz0)
				{
					int x_act = (i - fltr_pad0);
					bool invisibleX = (tile_x + x_act < 0) || (tile_x + x_act >= gbl_width);

					_FLOAT val = gbl_data[gbl_base + y_gbl_off + x_act];

					val = (invisibleX || invisibleY)? padding_val : val;
								
					lcl_data[y_lcl_off + i] = val;
				}
			}
}

__attribute__((reqd_work_group_size(ADNN_GRP_SZ0,ADNN_GRP_SZ1,ADNN_GRP_SZ2)))
__kernel void aDNNConv_LE32_NCHW(
       const __global _FLOAT * bot,
//#if ADNN_WEI_SZ < (1 << 14)
//      __constant _FLOAT * weights __attribute__((max_constant_size(ADNN_WEI_SZ))),
//#else
      const __global _FLOAT * weights,
//#endif
       const __global _FLOAT * bias,
	  __global _FLOAT *top,
	   _FLOAT padding_val,
	   int in_main_loop
	   )
{
		__local _FLOAT bot_lcl_data[ADNN_LCL_N_IN_CHNLS * ADNN_LCL_SZ];
		_FLOAT bot_staging[ADNN_PRV_TILE_V][ADNN_PRV_TILE_H];
		_FLOAT wei_staging[ADNN_FLTR_SZ0];
		_FLOAT accum[ADNN_N_OUT_PIX_SZ1][ADNN_N_OUT_PIX_SZ0];
#if ADNN_LCL_WEIGHTS == 1
#define ADNN_LCL_WEIGHTS_SZ (ADNN_LCL_N_IN_CHNLS * ADNN_FLTR_SZ0 * ADNN_FLTR_SZ1 * ADNN_LCL_N_OUT_CHNLS)
#define ADNN_LCL_WEIGHTS_STRIDE (ADNN_LCL_N_IN_CHNLS * ADNN_FLTR_SZ0 * ADNN_FLTR_SZ1)
		__local _FLOAT lcl_weights[ADNN_LCL_WEIGHTS_SZ];
#endif


		int tile_id = get_group_id(0);
		int lcl_id = get_local_id(0);
		int o = get_global_id(1); // output channels block
		int b = get_global_id(2); // batch


#if ADNN_BIG == 0
		for(int i = lcl_id; i < ADNN_LCL_N_IN_CHNLS * ADNN_LCL_SZ; i+= ADNN_GRP_SZ)
		{
			bot_lcl_data[i] = 0;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
#endif

// input processing
// split the group into ADNN_LCL_N_IN_CHNLS input vector processors
// each processor reads its own input channel
		int in_gbl_base = 0;
		int lcl_base = ADNN_LCL_STRIDE * ADNN_FLTR_PAD_SZ1;
		int in_proc_size = (1  << (ADNN_GRP_LG2SZ0 - ADNN_LCL_LOG2N_IN_CHNLS));
		int in_proc_id = (lcl_id >> (ADNN_GRP_LG2SZ0 - ADNN_LCL_LOG2N_IN_CHNLS));
		int in_thread_id = lcl_id - (in_proc_id << (ADNN_GRP_LG2SZ0 - ADNN_LCL_LOG2N_IN_CHNLS));
	
		int gbl_tile_id_y = tile_id / ADNN_N_OUT_TILES0;
		int gbl_tile_id_x = tile_id - mul24(gbl_tile_id_y, ADNN_N_OUT_TILES0);
		int lcl_in1 = (lcl_id >> ADNN_N_READ_PROC_LOG2SZ0);
// local id along x
		int lcl_in0 = lcl_id - (lcl_in1 << ADNN_N_READ_PROC_LOG2SZ0);


// output processing
// find the output my tile processor and my processor in the tile
// local id along y
		int lcl_out1 = (lcl_id >> ADNN_OUT_PROC_LOG2SZ0);
// local id along x
		int lcl_out0 = lcl_id - (lcl_out1 << ADNN_OUT_PROC_LOG2SZ0);
// my processor tile id along y
		int lcl_tile_out1 = lcl_out1 >> (ADNN_OUT_PROC_LOG2SZ1 - ADNN_OUT_N_TILEPROC_LOG2SZ1);
// my processor tile id along x
		int lcl_tile_out0 = lcl_out0 >> (ADNN_OUT_PROC_LOG2SZ0 - ADNN_OUT_N_TILEPROC_LOG2SZ0);
// my processor id along y
		int lcl_p_out1 = lcl_out1 - (lcl_tile_out1 << (ADNN_OUT_PROC_LOG2SZ1 - ADNN_OUT_N_TILEPROC_LOG2SZ1));
// my processor id along x
		int lcl_p_out0 = lcl_out0 - (lcl_tile_out0 << (ADNN_OUT_PROC_LOG2SZ0 - ADNN_OUT_N_TILEPROC_LOG2SZ0));
// output channel id processed by my tiled processor
		int tile_stride = (ADNN_OUT_N_TILEPROC_SZ0);
		int lcl_out_chnl_id = mad24(lcl_tile_out1, tile_stride, lcl_tile_out0);
		int my_out_channel = o*ADNN_LCL_N_OUT_CHNLS +  lcl_out_chnl_id;

		for(int j = 0; j < ADNN_N_OUT_PIX_SZ1; ++j)
		{
			for(int i = 0; i < ADNN_N_OUT_PIX_SZ0; ++i)
			{
				accum[j][i] = 0;
			}
		}

	
// main loop over all input channels		 
		for( int c = 0; c < ADNN_IN_CHNL_LOOP; ++c)
		{

// make sure we won't overwrite before processing everything
// this could be avoided by using double buffering -speed up about 2-3%
			barrier(CLK_LOCAL_MEM_FENCE);

// read ADNN_LCL_N_IN_CHNLS input channesl cooperatively
			in_gbl_base = b * ADNN_IN_BATCH_STRIDE + (c*ADNN_LCL_N_IN_CHNLS + in_proc_id) * ADNN_IN_CHNL_STRIDE;

#if ADNN_BIG == 0

			readData(&bot_lcl_data[in_proc_id * ADNN_LCL_SZ], bot, in_thread_id, in_proc_size, ADNN_IN_SIZE, ADNN_IN_WIDTH, ADNN_IN_STRIDE, in_gbl_base, ADNN_LCL_STRIDE, lcl_base);
#else

			int tile_y = (gbl_tile_id_y << ADNN_OUT_TILE_LOG2SZ1);
			int tile_x = (gbl_tile_id_x << ADNN_OUT_TILE_LOG2SZ0);
			in_gbl_base += tile_y * ADNN_IN_STRIDE + tile_x;
			readDataTile(bot_lcl_data, bot,
						tile_y, tile_x,
						ADNN_IN_STRIDE, in_gbl_base,
						ADNN_LCL_STRIDE, 0,
						ADNN_IN_HEIGHT, ADNN_IN_WIDTH, 
						(ADNN_OUT_TILE_SZ1 + ADNN_FLTR_PAD_SZ1 *2), (ADNN_OUT_TILE_SZ0 + ADNN_FLTR_PAD_SZ0 *2),
						lcl_in1, lcl_in0,
						ADNN_N_READ_PROC_SZ1, ADNN_N_READ_PROC_SZ0,
						ADNN_FLTR_PAD_SZ1, ADNN_FLTR_PAD_SZ0,
						padding_val);

#endif
#if ADNN_LCL_WEIGHTS == 1
//			int weigths_off = o*ADNN_LCL_N_OUT_CHNLS*ADNN_WEIGHTS_STRIDE + c*ADNN_LCL_N_IN_CHNLS * ADNN_FLTR_SZ;
			readWeights(lcl_weights, &weights[o*ADNN_LCL_N_OUT_CHNLS*ADNN_WEIGHTS_STRIDE + c*ADNN_LCL_N_IN_CHNLS * ADNN_FLTR_SZ],
							lcl_id, ADNN_GRP_SZ, ADNN_LCL_WEIGHTS_SZ, ADNN_LCL_WEIGHTS_STRIDE, ADNN_WEIGHTS_STRIDE);
#endif

// make sure we've read everything
			barrier(CLK_LOCAL_MEM_FENCE);

			int lcl_weights_base = lcl_out_chnl_id * ADNN_FLTR_SZ * ADNN_LCL_N_IN_CHNLS;

			for(int cc = 0; cc < ADNN_LCL_N_IN_CHNLS; ++cc)
			{
/// load and convolve with first ADNN_PRV_TILE_V rows of raw data
				int jj = 0;
				int k = 0;
// loads weights - row 0
				for(int i = 0; i < ADNN_FLTR_SZ0; ++i)
				{
#if ADNN_LCL_WEIGHTS == 1
					wei_staging[i] = lcl_weights[lcl_weights_base + cc * ADNN_FLTR_SZ + k*ADNN_FLTR_SZ0 + i];
#else
					wei_staging[i] = weights[my_out_channel * ADNN_WEIGHTS_STRIDE + (c*ADNN_LCL_N_IN_CHNLS + cc) * ADNN_FLTR_SZ + k*ADNN_FLTR_SZ0 + i];
#endif
				}

// loads data and convolve
                int bot_lcl_data_base = mul24(cc, ADNN_LCL_SZ) + mul24(mad24(lcl_p_out1, ADNN_N_OUT_PIX_SZ1,jj),ADNN_LCL_STRIDE) + mad24(lcl_p_out0, ADNN_N_OUT_PIX_SZ0,0);
				for( ; jj < ADNN_PRV_TILE_V; ++jj)
				{	

					for(int i = 0; i < ADNN_PRV_TILE_H; ++i)
					{
					    int local_addr = bot_lcl_data_base + i;
						bot_staging[jj][i] = bot_lcl_data[local_addr];
					}
	

                    bot_lcl_data_base += ADNN_LCL_STRIDE;

					for(int i = 0; i < ADNN_N_OUT_PIX_SZ0; ++i)
					{
						for(int l = 0; l < ADNN_FLTR_SZ0; ++l)
						{
							_FLOAT prev_accum = accum[jj][i];
							accum[jj][i] += bot_staging[jj][i + l] * wei_staging[l]; 	
						}
					}
// WHY??
#if !(ADNN_FLTR_SZ0==3 && (ADNN_IN_WIDTH <=8 || ADNN_IN_HEIGHT <=8))					
					mem_fence(CLK_LOCAL_MEM_FENCE);
#endif
				}


// convolution of ADNN_N_OUT_PIX_SZ0 x ADNN_N_OUT_PIX_SZ1 pixels
				int prev_jj = jj;
                int bot_lcl_data_base2 = mul24(cc, ADNN_LCL_SZ) + mul24(mad24(lcl_p_out1, ADNN_N_OUT_PIX_SZ1,jj),ADNN_LCL_STRIDE) + mad24(lcl_p_out0, ADNN_N_OUT_PIX_SZ0,0);
				for( ; jj < prev_jj + ADNN_FLTR_SZ1 - 1; ++jj)
				{	
// load weights
					k = jj - ADNN_PRV_TILE_V + 1;
					for(int i = 0; i < ADNN_FLTR_SZ0; ++i)
					{
#if ADNN_LCL_WEIGHTS == 1
						wei_staging[i] = lcl_weights[lcl_weights_base + cc * ADNN_FLTR_SZ + k*ADNN_FLTR_SZ0 + i];
#else
						wei_staging[i] = weights[my_out_channel * ADNN_WEIGHTS_STRIDE + (c*ADNN_LCL_N_IN_CHNLS + cc) * ADNN_FLTR_SZ + k*ADNN_FLTR_SZ0 + i];
#endif
					}

// convolve
					for(int j = 0; j < ADNN_N_OUT_PIX_SZ1 - 1; ++j)
					{
// move raw data up one row
						for(int i = 0; i < ADNN_PRV_TILE_H; ++i)
						{
							bot_staging[j][i] = bot_staging[j+1][i];
						}

						for(int i = 0; i < ADNN_N_OUT_PIX_SZ0; i++)
						{
							for(int l = 0; l < ADNN_FLTR_SZ0; l++)
							{
								accum[j][i] += bot_staging[(j+1)][i + l] * wei_staging[l]; 	

							}
						}

					}


// load the last row with new raw data
					for(int i = 0; i < ADNN_PRV_TILE_H; ++i)
					{
					    int local_addr = bot_lcl_data_base2+i;
						bot_staging[ADNN_PRV_TILE_V - 1][i] = bot_lcl_data[local_addr];
					}
//					mem_fence(CLK_LOCAL_MEM_FENCE);

                    bot_lcl_data_base2+=ADNN_LCL_STRIDE; 
// convolve
					for(int i = 0; i < ADNN_N_OUT_PIX_SZ0; i++)
					{
						for(int l = 0; l < ADNN_FLTR_SZ0; l++)
						{
							accum[ADNN_N_OUT_PIX_SZ1 - 1][i] += bot_staging[(ADNN_N_OUT_PIX_SZ1 - 1)*ADNN_FLTR_STRIDE1][i*ADNN_FLTR_STRIDE0 + l] * wei_staging[l]; 																				

						}
					}

				}
			}

		} // for( int c = 0; c < ADNN_IN_CHNL_LOOP; c+= ADNN_LCL_N_IN_CHNLS)
	
	

// if I'm out off output channels, return.
		if ( my_out_channel >= ADNN_N_OUT_CHNLS)
		{
			return;
		}
		

		_FLOAT  bias_val = bias[my_out_channel];

// write out
// TO DO:: make it coalesed or write4 
		int out_x = lcl_p_out0 *ADNN_N_OUT_PIX_SZ0 + (gbl_tile_id_x << ADNN_OUT_TILE_LOG2SZ0);
		int out_y = lcl_p_out1 *ADNN_N_OUT_PIX_SZ1 + (gbl_tile_id_y << ADNN_OUT_TILE_LOG2SZ1);
		int top_off = b * ADNN_OUT_BATCH_STRIDE + my_out_channel*ADNN_IN_CHNL_STRIDE +  out_y *ADNN_OUT_STRIDE + out_x;
		for(int j = 0; j < ADNN_N_OUT_PIX_SZ1; ++j, ++out_y, top_off += ADNN_OUT_STRIDE)
		{
			int out_x2 = out_x;
			for(int i = 0; i < ADNN_N_OUT_PIX_SZ0; ++i, ++out_x2)
			{
#if ADNN_ALIGNED != 1				
				if (out_y < ADNN_OUT_HEIGHT && out_x2 < ADNN_OUT_WIDTH)
#endif
				{

					_FLOAT out_val = accum[j][i];
					top[top_off + i] = out_val + bias_val;

				}
			}
		}


}



