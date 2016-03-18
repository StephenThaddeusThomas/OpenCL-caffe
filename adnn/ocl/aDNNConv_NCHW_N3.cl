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
//ADNN_GRP_SZ2				group size in dim 2
//ADNN_GRP_SZ               n of wk-item in the group
//ADNN_N_IN_CHNLS			total number of input channels
//ADNN_LCL_N_IN_CHNLS		n of localy kept input channels
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
//ADNN_N_OUT_CHNLS			total number of output channels
//ADNN_LCL_N_OUT_CHNLS		n of localy kept output channels
//ADNN_OUT_WIDTH			output width in NCHW layout
//ADNN_OUT_HEIGHT			output height stride in NCHW layout
//ADNN_OUT_STRIDE			output stride in NCHW layout
//ADNN_OUT_CHNL_STRIDE      output channel stride in NCHW layout
//ADNN_OUT_BATCH_STRIDE     output batch stride in NCHW layout
//ADNN_N_OUT_PIX_SZ0        n output pixel per wk item in 0 dim
//ADNN_N_OUT_PIX_SZ1		n output pexels per wk item in 1 dim
//ADNN_N_IN_PIX_SZ0        n input pixels per wk item in 0 dim
//ADNN_N_IN_PIX_SZ1		n input pexels per wk item in 1 dim
//ADNN_WEIGHTS_STRIDE			weights stride
//ADNN_WEI_SZ               size of weight buffer
//ADNN_N_STACKS           n of separate data stacks
//ADNN_N_PROCS1           n of processors per stack 1 dim
//ADNN_N_PROCS0           n of processors per stack 0 dim

// inputs are taken from different stacks of batches - to use the same filters
//#define ADNN_N_PRPOCS (ADNN_GRP_SZ/ADNN_LCL_N_IN_CHNLS)   // n of wk-items per input, 
#define ADNN_LCL_BOT_WIDTH (ADNN_N_IN_PIX_SZ0 * ADNN_N_PROCS0 + ADNN_FLTR_PAD_SZ0 * 2)
#define ADNN_LCL_BOT_HEIGHT  (ADNN_N_IN_PIX_SZ1 * ADNN_N_PROCS1 + ADNN_FLTR_PAD_SZ1 * 2)
#define ADNN_LCL_BOT_SIZE (ADNN_LCL_BOT_WIDTH * ADNN_LCL_BOT_HEIGHT)
#define ADNN_LCL_WEI_SIZE (ADNN_LCL_N_OUT_CHNLS * ADNN_FLTR_SZ0 * ADNN_FLTR_SZ1) 
#define ADNN_PVT_BOT_WIDTH (ADNN_N_IN_PIX_SZ0 + ADNN_FLTR_PAD_SZ0 * 2) 
#define ADNN_INNER_OUT_LOOP (ADNN_N_IN_PIX_SZ1 + ADNN_FLTR_PAD_SZ0 * 2) 

#define ADNN_PVT_OUT_DATA_HEIGHT ADNN_N_OUT_PIX_SZ1 * ADNN_LCL_N_OUT_CHNLS
#define ADNN_PVT_OUT_DATA_SZ ADNN_PVT_OUT_DATA_HEIGHT * ADNN_N_OUT_PIX_SZ0


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
	lcl_off += lcl_base;
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
#if ADNN_WEI_SZ < (1 << 14)
inline void readWeights(__local _FLOAT *lcl_data, __constant _FLOAT * gbl_data, int p_id, int p_stride, int size, int lcl_stride, int gbl_stride)
#else
inline void readWeights(__local _FLOAT *lcl_data, __global _FLOAT * gbl_data, int p_id, int p_stride, int size, int lcl_stride, int gbl_stride)
#endif
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
__kernel void aDNNConv_NCHW_N3(
       const __global _FLOAT * bot,
#if ADNN_WEI_SZ < (1 << 14)
      __constant _FLOAT * weights __attribute__((max_constant_size(ADNN_WEI_SZ))),
#else
      __global _FLOAT * weights,
#endif
       const __global _FLOAT * bias,
	  __global _FLOAT *top,
	   _FLOAT padding_val,
	   int in_main_loop
	   )
{
	__local _FLOAT lcl_bot[ADNN_LCL_BOT_SIZE * ADNN_LCL_N_IN_CHNLS];
	__local _FLOAT lcl_wei[ADNN_LCL_WEI_SIZE];

	_FLOAT pvt_bot_dat[ADNN_PVT_BOT_WIDTH * ADNN_N_OUT_PIX_SZ1];
	_FLOAT pvt_wei_dat[ADNN_FLTR_SZ0];
	_FLOAT pvt_top_dat[ADNN_PVT_OUT_DATA_SZ];

	int x_out_grp = get_group_id(0) * ADNN_N_PROCS0 *  ADNN_N_OUT_PIX_SZ0;
	int y_out_grp = get_group_id(1) * ADNN_N_PROCS1 * ADNN_N_OUT_PIX_SZ1;

	int lcl_id = mad24(get_local_id(1),ADNN_GRP_SZ0, get_local_id(0));
	int lcl_proc = lcl_id/(ADNN_N_PROCS0*ADNN_N_PROCS1);  // input id from diff stack
	int lcl_in_proc_id = -mad24(lcl_proc, (ADNN_N_PROCS0*ADNN_N_PROCS1), -lcl_id);  // wk item id for the input to make a coalesed read
	int lcl_proc_id1 = lcl_in_proc_id/ ADNN_N_PROCS0;  // 
	int lcl_proc_id0 = -mad24(lcl_proc_id1, ADNN_N_PROCS0, -lcl_in_proc_id);  // 
	int x_in_lcl = mul24(lcl_proc_id0, ADNN_N_OUT_PIX_SZ0);
	int y_in_lcl = mul24(lcl_proc_id1, ADNN_N_OUT_PIX_SZ1);

	int ob = get_global_id(2);
	int o_id = ob / ADNN_N_STACKS;  // block of outputs
	int b_id = -mad24(o_id, ADNN_N_STACKS, -ob);         // block of batchs
// my batch
	int b = b_id* ADNN_LCL_N_IN_CHNLS + lcl_proc;

// include STRIDE later
	int x_out = x_out_grp + x_in_lcl;
	int y_out = y_out_grp + y_in_lcl;

	int in_off = b * ADNN_IN_BATCH_STRIDE;
	int wei_off = mul24(o_id, ADNN_LCL_N_OUT_CHNLS * ADNN_N_IN_CHNLS * ADNN_FLTR_SZ0 * ADNN_FLTR_SZ1);

	int lcl_off = mul24(lcl_proc, ADNN_LCL_BOT_SIZE) + mad24(y_in_lcl, ADNN_LCL_BOT_WIDTH, x_in_lcl);

#if ADNN_BIG == 0
	for(int i = lcl_id; i < ADNN_LCL_BOT_SIZE * ADNN_LCL_N_IN_CHNLS;  i += ADNN_GRP_SZ)
	{
		lcl_bot[i] = 0;
	}
#endif

	for(int i = 0; i < ADNN_PVT_OUT_DATA_SZ; ++i)
	{
		pvt_top_dat[i] = 0;
	}

	for(int c = 0; c < ADNN_N_IN_CHNLS; c++, in_off += ADNN_IN_CHNL_STRIDE, wei_off += ADNN_FLTR_SZ0 * ADNN_FLTR_SZ1)
	{

		barrier(CLK_LOCAL_MEM_FENCE);

// put weights for all our outputs for this input into LDS 
		for(int i = lcl_id; i < ADNN_LCL_N_OUT_CHNLS * ADNN_FLTR_SZ0 * ADNN_FLTR_SZ1; i += ADNN_GRP_SZ)
		{
			int lcl_o = i/(ADNN_FLTR_SZ0 * ADNN_FLTR_SZ1);
			int lcl_o_i = i - lcl_o * (ADNN_FLTR_SZ0 * ADNN_FLTR_SZ1);

			lcl_wei[i] = weights[wei_off + lcl_o * ADNN_N_IN_CHNLS * ADNN_FLTR_SZ0 * ADNN_FLTR_SZ1 + lcl_o_i];
		}

#if ADNN_BIG

			int tile_y = y_out_grp;
			int tile_x = x_out_grp;
			readDataTile(lcl_bot,
						bot,
						tile_y,
						tile_x,
						ADNN_IN_STRIDE,
						(in_off + tile_y * ADNN_IN_STRIDE + tile_x),
						ADNN_LCL_BOT_WIDTH,
						0,
						ADNN_IN_HEIGHT,
						ADNN_IN_WIDTH, 
						ADNN_LCL_BOT_HEIGHT,
						ADNN_LCL_BOT_WIDTH,
						lcl_proc_id1,
						lcl_proc_id0,
						ADNN_N_PROCS1,
						ADNN_N_PROCS0,
						ADNN_FLTR_PAD_SZ1,
						ADNN_FLTR_PAD_SZ0,
						padding_val);


#else
			int lcl_base = ADNN_LCL_BOT_WIDTH * ADNN_FLTR_PAD_SZ1 + ADNN_FLTR_PAD_SZ0;
			readData(&lcl_bot[lcl_proc * ADNN_LCL_BOT_SIZE + lcl_base],
				 bot,
				lcl_in_proc_id,
				(ADNN_N_PROCS0*ADNN_N_PROCS1),
				(ADNN_IN_WIDTH*ADNN_IN_HEIGHT),
				ADNN_IN_WIDTH,
				ADNN_IN_STRIDE,
				in_off,
				ADNN_LCL_BOT_WIDTH,
				0);

#endif
		barrier(CLK_LOCAL_MEM_FENCE);

// get first ADNN_N_OUT_PIX_SZ1 - 1 lines
		int j = 0;
		int lcl_off2 = lcl_off;
		for(; j < ADNN_N_OUT_PIX_SZ1 - 1; ++j)
		{

// read input data
			for(int i = 0; i < ADNN_PVT_BOT_WIDTH; ++i)
			{
				pvt_bot_dat[j * ADNN_PVT_BOT_WIDTH + i] = lcl_bot[lcl_off2 + i];
			}	
			lcl_off2 += ADNN_LCL_BOT_WIDTH;

		}


// convolve over the filter
		int lcl_wei_off = 0;
		for(; j < ADNN_INNER_OUT_LOOP; ++j,  lcl_wei_off+= ADNN_FLTR_SZ0)
		{

// read input data
			for(int i = 0; i < ADNN_PVT_BOT_WIDTH; ++i)
			{
				pvt_bot_dat[(ADNN_N_OUT_PIX_SZ1 - 1) * ADNN_PVT_BOT_WIDTH + i] = lcl_bot[lcl_off2 + i];			
			}	
			lcl_off2 += ADNN_LCL_BOT_WIDTH;
// convolve over all weights
			int lcl_wei_off2 = lcl_wei_off;

			for(int o = 0; o < ADNN_LCL_N_OUT_CHNLS; ++o, lcl_wei_off2 += ADNN_FLTR_SZ1 * ADNN_FLTR_SZ0)
			{
// read weights
				for(int w = 0; w < ADNN_FLTR_SZ0; ++w)
				{
					pvt_wei_dat[w] = lcl_wei[lcl_wei_off2 + w];
				}

// convolve over the tile
				for( int pj = 0; pj < ADNN_N_OUT_PIX_SZ1; ++pj)
				{
					for(int pi = 0; pi < ADNN_N_OUT_PIX_SZ0; ++pi)
					{

						for(int m = 0; m < ADNN_FLTR_SZ0; ++m)
						{
							pvt_top_dat[(o * ADNN_N_OUT_PIX_SZ1 + pj) * ADNN_N_OUT_PIX_SZ0 + pi] += pvt_bot_dat[pj * ADNN_PVT_BOT_WIDTH + pi + m] * pvt_wei_dat[m];

						}
					}
				}
				
			}
// move up
			for(int j = 0; j < (ADNN_N_OUT_PIX_SZ1 - 1); ++j)
			{
				for(int i = 0; i < ADNN_PVT_BOT_WIDTH; ++i)
				{
					pvt_bot_dat[j * ADNN_PVT_BOT_WIDTH + i] = pvt_bot_dat[(j+1) * ADNN_PVT_BOT_WIDTH + i];
				}
			}


		}  //for(; j < ADNN_INNER_OUT_LOOP; ++j, in_off2 += ADNN_IN_STRIDE)



	} // for(int c = 0; c < ADNN_N_IN_CHNLS; c++, in_off += ADNN_IN_CHNL_STRIDE, we_off += ADNN_FLTR_SZ0 * ADNN_FLTR_SZ1)

#if ADNN_BATCH_ALIGNED == 0
	if ( b >= ADNN_BATCH_SZ)
	{
		return;
	}
#endif


// write to all outputs
	int top_off = b * ADNN_OUT_BATCH_STRIDE + o_id * ADNN_LCL_N_OUT_CHNLS * ADNN_OUT_CHNL_STRIDE +  y_out *ADNN_OUT_STRIDE + x_out;

	for(int o = 0; o < ADNN_LCL_N_OUT_CHNLS; ++o, top_off += ADNN_OUT_CHNL_STRIDE)
	{

#if ADNN_OUT_ALIGNED == 0
		if ( o_id * ADNN_LCL_N_OUT_CHNLS + o >= ADNN_N_OUT_CHNLS)
		{
			return;
		}
#endif
		_FLOAT  bias_val = bias[o_id * ADNN_LCL_N_OUT_CHNLS + o];

		int top_off2 = top_off;
		for(int j = 0; j < ADNN_N_OUT_PIX_SZ1; ++j, top_off2 += ADNN_OUT_STRIDE)
		{
			for(int i = 0; i < ADNN_N_OUT_PIX_SZ0; ++i)
			{
#if ADNN_ALIGNED == 0
				if ( y_out + j < ADNN_OUT_HEIGHT && x_out + i < ADNN_OUT_WIDTH)
#endif
					top[top_off2 + i] = pvt_top_dat[(o * ADNN_N_OUT_PIX_SZ1 + j) * ADNN_N_OUT_PIX_SZ0 + i] + bias_val;

			}
		}
	}


}

