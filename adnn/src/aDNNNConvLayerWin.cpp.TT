/**********************************************************************
Copyright ?2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

?	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
?	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/
// to share code with between CPU and GPU

#include "aDNNInternal.hpp"

#define ADNN_ELEMENT_WISE_TRANSOFRMS 1
#define ADNN_ALG_WIN2x2_3x3 1       // 2x2

// TODO: REMOVE DEFINE
#define ADNN_WWT_PEROUTPUT 0  // 1 per output; otherwise per input
#define ADNN_WIT_OUTLINEAR 1  // 1 output transformed input scanline-wise; 0 - tiled.
#define ADNN_WEWM_OUTLINEAR 1 // 1 read/write element-wise multiply scanline-wise; 0 - tiled

namespace adnn
{




	/************************************************************************************************************************
	**
	**			CONVOLUTIONAL LAYER (winograd alg)
	**
	************************************************************************************************************************/


	/************************************************************************************************************************
	**
	**			FORWARD PROPAGATION
	**
	************************************************************************************************************************/

	int aDNNodeConv::ConstructFwdWin_NCHW(void)
	{
		int ret = 0;
		//ADNN_WWT_GRP_SZ0		weights transform group size in dim 0
		//ADNN_WWT_GRP_SZ1		weights transform group size in dim 1
		//ADNN_WWT_GRP_SZ2		weights transform group size in dim 2
		//ADNN_WWT_PEROUTPUT            weights linear over output; otherwise linear per input
		//ADNN_WIT_GRP_SZ0		input transform group size in dim 0
		//ADNN_WIT_GRP_LG2SZ0		log
		//ADNN_WIT_GRP_SZ1		input transform group size in dim 1
		//ADNN_WIT_GRP_SZ2		input transform group size in dim 2
		//ADNN_WIT_OUTLINEAR            inverse tarsnform layout - linear; otherwise tiled per block1xblock0 block
		//ADNN_W_BATCH_SZ		batch size
		//ADNN_W_N_ICHNLS               total number of input channels
		//ADNN_W_N_OCHNLS               total number of output channels
		//ADNN_W_HEIGH                  input height
		//ADNN_W_WIDTH			input width
		//ADNN_W_ISTRIDE		input stride
		//ADNN_W_ICHNL_STRIDE		input channel stride
		//ADNN_W_IBATCH_STRIDE          input batch strride
		//ADNN_W_BSTRIDE		blocked(trasformed input) stride
		//ADNN_W_BCHNL_STRIDE		blocked(trasformed input) channel stride
		//ADNN_W_BBATCH_STRIDE		blocked(trasformed input) batch stride
		//ADNN_W_TILE1			tile: 2x2 or 4x4
		//ADNN_W_TILE0
		//ADNN_W_BLOCK0			block size dim 0 4x4 or 6x6
		//ADNN_W_BLOCK1			block size dim 1
		//ADNN_W_FLTR0			filter size dim 0
		//ADNN_W_FLTR1			filter size dim 1
		//ADNN_W_PAD1,
		//ADNN_W_PAD0
		//ADNN_WIT_LCL_N_IN_CHNLS	n of localy kept input channels
		//ADNN_WIT_LCL_LOG2N_IN_CHNLS	log2 of n of localy kept input channels
		//ADNN_WIT_N_IN_TILES0		n total input tiles dim 0 for > 32x32
		//ADNN_WIT_N_IN_TILES1		n total input tiles dim 1 for > 32x32
		//ADNN_WIT_IN_TILE_LOG2SZ1      log size of input tile for > 32x32
		//ADNN_WIT_IN_TILE_LOG2SZ0      log size of input tile for > 32x32
		//ADNN_WIT_IN_PROC_SZ		n of processors reading 1 input channel for < 32x32 :  (1  << (ADNN_WIT_GRP_LG2SZ0 - ADNN_LCL_LOG2N_IN_CHNLS))
		//ADNN_WIT_READ_SZ1		length to read dim 1
		//ADNN_WIT_READ_SZ0,		lenght to read dim 0
		//ADNN_WIT_N_TILEPROCS1         n proc in proc tile dim 1
		//ADNN_WIT_N_TILEPROCS0         n proc in proc tile dim 0
		//ADNN_WIT_BLKD_TILE_SZ1	size of blocked tiles dim 1 for > 32x32
		//ADNN_WIT_BLKD_TILE_SZ0	size of blocked tiles dim 0 for > 32x32
		//ADNN_WIT_BIG                  input > 32x32
		//ADNN_WIT_IN_SIZE              input block size
		//ADNN_WIT_LCL_HEIGHT		local memory height (ADNN_IN_HEIGHT + ADNN_W_PAD1 * 2)
		//ADNN_WIT_LCL_WIDTH		local memory width (ADNN_IN_WIDTH + ADNN_W_PAD0 * 2)
		//ADNN_WIT_LCL_STRIDE		local memory stride ADNN_LCL_WIDTH
		//ADNN_WIT_LCL_SZ		local memory size (ADNN_WIT_LCL_STRIDE * ADNN_LCL_HEIGHT)
		//invert transform
		//ADNN_WMIT_GRP_SZ0
		//ADNN_WMIT_GRP_SZ1
		//ADNN_WMIT_GRP_SZ2
		//ADNN_WMIT_LCL_N_IN_CHNLS      n of lcl input channels - the  same id but from different batchs
		//ADNN_WMIT_LCL_N_OUT_CHNLS     n of lcl out channels
		//ADNN_WMIT_IN_PROC_SZ          n of processors reading 1 input channel
		//ADNN_W_BHEIGH                 height of the transformed buffer
		//ADNN_W_BWIDTH                 width of tarnsformed data buffer in pixels
		//ADNN_W_OBATCH_STRIDE
		//ADNN_W_OCHNL_STRIDE
		//ADNN_W_OSTRIDE
		// multiply
		//ADNN_WMT_GRP_SZ0		group size
		//ADNN_WMT_GRP_SZ1
		//ADNN_WMT_GRP_SZ2
		//ADNN_WMT_LCL_N_OUT_CHNLS      n of local output channels
		//ADNN_WMT_PIXPER_WKITM         n input pixels per wk-item
		//ADNN_WMT_WKITMSPER_BLOCK      n wk items per block
		//WIT_LCL_TFM_WIDTH		input tile transformed size 0
		//WIT_LCL_TFM_HEIGHT            input tile transformed size 1


// TODO: make it controllable
		win2x2_ = (ADNN_ALG_WIN2x2_3x3) ? true : false;
		int ALG_WIN2x2_3x3 = (win2x2_) ? 1 : 0;
		int ELEMENT_WISE_TRANSOFRMS = ADNN_ELEMENT_WISE_TRANSOFRMS;

		int pad0 = getPad();
		int stride0 = getKernelStride();
		int kernel_size0 = getKernelSz();
		int pad1 = getPad(0, 1);
		int stride1 = getKernelStride(0, 1);
		int kernel_size1 = getKernelSz(0, 1);


		const aDNNTensor & bot = getBotFwd();
		const aDNNTensor & top = getTopFwd();
		const aDNNTensor & wei = getBotWeightsFwd();

		int width = (int)bot.getDim(aDNN_TENSOR_WIDTH);
		int height = (int)bot.getDim(aDNN_TENSOR_HEIGHT);
		int inputs = (int)bot.getDim(aDNN_TENSOR_DEPTH);
		int batch_sz = (int)bot.getDim(aDNN_TENSOR_BATCH);
		int bot_stride = (int)bot.getStride(aDNN_TENSOR_WIDTH);
		int bot_channel_stride = (int)bot.getStride(aDNN_TENSOR_HEIGHT);
		int bot_batch_stride = (int)bot.getStride(aDNN_TENSOR_DEPTH);

		int top_stride = (int)top.getStride(aDNN_TENSOR_WIDTH);
		int top_channel_stride = (int)top.getStride(aDNN_TENSOR_HEIGHT);
		int top_height_stride = top_channel_stride / top_stride;
		int	top_batch_stride = (int)top.getStride(aDNN_TENSOR_DEPTH);

		int outputs = (int)top.getDim(aDNN_TENSOR_DEPTH);
		int height_out = (int)top.getDim(aDNN_TENSOR_HEIGHT);
		int width_out = (int)top.getDim(aDNN_TENSOR_WIDTH);


		int weights_hight = (int)wei.getDim(aDNN_TENSOR_HEIGHT);
		int weights_stride = (int)wei.getStride(aDNN_TENSOR_WIDTH);


		int tile_sz0, tile_sz1;
		int block_sz0, block_sz1;
#if ADNN_ALG_WIN2x2_3x3
		tile_sz0_ = 2;
		tile_sz1_ = 2;
#else
		tile_sz0_ = 4;
		tile_sz1_ = 4;
#endif

		tile_sz0 = tile_sz0_;
		tile_sz1 = tile_sz1_;
		block_sz0 = tile_sz0 + pad0 * 2;
		block_sz1 = tile_sz1 + pad1 * 2;


		int weights_sz = outputs * inputs* kernel_size0 * kernel_size1;

		// transformed weights weights
		adnn_data_parameters tweights_params;
		wei.getParams(tweights_params);
		for (int i = 0; i < ADNN_MAX_TENSOR_DIM; ++i)
		{
			tweights_params.strides[i] = 0;
		}

		tweights_params.dims[1] = inputs * block_sz0 * block_sz1;

		aDNNTensor & tweights_slot = createSlot(getWeightsNm() + ADNN_WIN_TRANSFORM_NM, tweights_params);

		if (getDebugLevel() & 1)
		{
			cloneSlot(getWeightsNm() + ADNN_WIN_TRANSFORM_NM + ADNN_VERIFY_NM, tweights_slot);
		}


		int WWT_GRP_SZ0 = (inputs <= 4) ? 4 : 8;	// weights transform group size in dim 0
		int WWT_GRP_SZ1 = (64 / WWT_GRP_SZ0);		// weights transform group size in dim 1
		int WWT_GRP_SZ2 = 1;						//weights transform group size in dim 2
		// transformed layout
		int wwt_peroutput = ADNN_WWT_PEROUTPUT;
		// trasform input

		adnn_data_parameters tinput_params;
		bot.getParams(tinput_params);
		for (int i = 0; i < ADNN_MAX_TENSOR_DIM; ++i)
		{
			tinput_params.strides[i] = 0;
		}

		int input_chnl_sz = height * width;

		int WIT_BIG = (input_chnl_sz > 32 * 32) ? 1 : 0;                 //input > 32x32

		// input tarnsform
		int WIT_IN_TILE_SZ1 = (WIT_BIG) ? 32 : height;   // size of input data tile per group
		int WIT_IN_TILE_SZ0 = (WIT_BIG) ? 32 : width;
		int tile_log2sz0 = (int)ceil(log((double)WIT_IN_TILE_SZ1) / log(2.));
		int tile_log2sz1 = (int)ceil(log((double)WIT_IN_TILE_SZ0) / log(2.));
		int n_tiles_TILE1 = ((WIT_IN_TILE_SZ1 + tile_sz1 - 1) / tile_sz1); // n of local tiles per group
		int n_tiles_TILE0 = ((WIT_IN_TILE_SZ0 + tile_sz0 - 1) / tile_sz0);
		int wit_n_tiles1 = (height + WIT_IN_TILE_SZ1 - 1) / WIT_IN_TILE_SZ1;   // input in tile per group
		int wit_n_tiles0 = (width + WIT_IN_TILE_SZ0 - 1) / WIT_IN_TILE_SZ0;   // input in tile per group


		int n_proc_tiles1 = (height + tile_sz1 - 1) / tile_sz1; // total number of local tiles vertically
		int theight = n_proc_tiles1 * block_sz1;
		int n_proc_tiles0 = ((width + tile_sz0 - 1) / tile_sz0); // toatal number of local tiles horizontally
		int twidth = n_proc_tiles0 * block_sz0;
		tinput_params.dims[2] = theight;
		tinput_params.dims[3] = twidth;

		aDNNTensor & tbot = createSlot(getBotNm() + ADNN_WIN_TRANSFORM_NM, tinput_params);
		if (getDebugLevel() & 1)
		{
			cloneSlot(getBotNm() + ADNN_WIN_TRANSFORM_NM + ADNN_VERIFY_NM, tbot);
		}
		// outputs
		tinput_params.dims[1] = outputs;

		aDNNTensor & ttop = createSlot(getTopNm() + ADNN_WIN_TRANSFORM_NM, tinput_params);

		if (getDebugLevel() & 1)
		{
			cloneSlot(getTopNm() + ADNN_WIN_TRANSFORM_NM + ADNN_VERIFY_NM, ttop);
		}

		int tbot_stride = (int)tbot.getStride(aDNN_TENSOR_WIDTH);
		int tbot_channel_stride = (int)tbot.getStride(aDNN_TENSOR_HEIGHT);
		int tbot_batch_stride = (int)tbot.getStride(aDNN_TENSOR_DEPTH);

		int ttop_stride = (int)ttop.getStride(aDNN_TENSOR_WIDTH);
		int ttop_channel_stride = (int)ttop.getStride(aDNN_TENSOR_HEIGHT);
		int ttop_batch_stride = (int)ttop.getStride(aDNN_TENSOR_DEPTH);

		int WIT_GRP_SZ0; 					// input transform group size in dim 0
		int grp_dim1;
		int grp_dim0;
		int WIT_N_TILEPROCS1;
		int WIT_N_TILEPROCS0;
		if (win2x2_)
		{
			WIT_GRP_SZ0 = 256;
			WIT_N_TILEPROCS1 = 16;
			WIT_N_TILEPROCS0 = 16;
			grp_dim1 = 16;
			grp_dim0 = 16;
		}
		else
		{
			WIT_GRP_SZ0 = 64; 					// input transform group size in dim 0
			WIT_N_TILEPROCS1 = 8;
			WIT_N_TILEPROCS0 = 8;
			grp_dim1 = 8;
			grp_dim0 = 8;
		}

		int wit_n_proctiles1 = 1; // number of local tiles per wavefront
		int wit_n_proctiles0 = 1;


		int WIT_GRP_LG2SZ0 = (int)ceil(log((double)WIT_GRP_SZ0) / log(2.));		// log
		int WIT_GRP_SZ1 = 1; 					// input transform group size in dim 1
		int WIT_GRP_SZ2 = 1;					// input transform group size in dim 2
		int WIT_IN_SIZE = WIT_IN_TILE_SZ1 * WIT_IN_TILE_SZ0;             // input block size
		int WIT_LCL_HEIGHT = 0;				// local memory height(ADNN_IN_HEIGHT + ADNN_W_PAD1 * 2)
		int WIT_LCL_WIDTH = 0;				//local memory width(ADNN_IN_WIDTH + ADNN_W_PAD0 * 2)
		int WIT_LCL_N_IN_CHNLS = 1;
		int WIT_IN_PROC_SZ = 0;
		int WIT_READ_SZ1 = 0;
		int WIT_READ_SZ0 = 0;
		int WIT_BLKD_TILE_SZ1 = n_tiles_TILE1 * block_sz1; // width per transformed(blocked) data per group
		int WIT_BLKD_TILE_SZ0 = n_tiles_TILE0 * block_sz0;
		int wit_outlinear = ADNN_WIT_OUTLINEAR;

		// control map
		input_transform_ctlmap_ = malloc(WIT_GRP_SZ0 * 2 * sizeof(int) * 4); // control map
		memset(input_transform_ctlmap_, 0, WIT_GRP_SZ0 * 2 * sizeof(int) * 4);
		adnn_data_parameters in_ctlmap_params;
		memset(&in_ctlmap_params, 0, sizeof(adnn_data_parameters));
		in_ctlmap_params.data_format = ADNN_DF_I32;
		in_ctlmap_params.batch_format = ADNN_BF_NW;
		in_ctlmap_params.dims[0] = WIT_GRP_SZ0 * 2;
		in_ctlmap_params.dims[1] = 4;
		aDNNTensor & inp_ctlmap = createSlot(getBotNm() + ADNN_WIN_TRANSFORM_NM + ADNN_WIN_CTLMAP_NM, in_ctlmap_params);


		// inverse trasnform
		int WMIT_GRP_SZ0 = 256;
		int WMIT_LCL_N_IN_CHNLS = 1;         // n of input channels - the  same id but from different batchs
		int WMIT_LCL_N_OUT_CHNLS = WMIT_GRP_SZ0 / 64;

		if (WIT_BIG)
		{
		  // number of read processors 
		  WIT_IN_PROC_SZ = WIT_GRP_SZ0;

		  // how much to read in a specific direction
		  WIT_READ_SZ1 = WIT_LCL_HEIGHT;
		  WIT_READ_SZ0 = WIT_LCL_WIDTH;

		  // local memory resolution
		  WIT_LCL_HEIGHT = (WIT_IN_TILE_SZ1 + pad1 * 2);
		  WIT_LCL_WIDTH = (WIT_IN_TILE_SZ0 + pad0 * 2);

		  // filling input tranform control map
		  for (int i = 0; i < WIT_GRP_SZ0; ++i)
		    {
		      // lcl 0
		      ((int*)input_transform_ctlmap_)[i * 4 + 2] = i % WIT_N_TILEPROCS0;
		      // lcl 1
		      ((int*)input_transform_ctlmap_)[i * 4 + 3] = i / WIT_N_TILEPROCS0;

		      // 8x8 tile
		      ((int*)input_transform_ctlmap_)[(WIT_GRP_SZ0 + i) * 4 + 0] = i % WIT_N_TILEPROCS0;
		      ((int*)input_transform_ctlmap_)[(WIT_GRP_SZ0 + i) * 4 + 1] = i / WIT_N_TILEPROCS0;
		      // 1 processing tile
		      ((int*)input_transform_ctlmap_)[(WIT_GRP_SZ0 + i) * 4 + 2] = 0;
		      ((int*)input_transform_ctlmap_)[(WIT_GRP_SZ0 + i) * 4 + 3] = 0;
		    }
		}
		else
		{

			// how much to read in a specific direction
			WIT_READ_SZ1 = height;
			WIT_READ_SZ0 = width;

			// local memory resolution
			WIT_LCL_HEIGHT = (n_proc_tiles1 * tile_sz1 + pad1 * 2);
			WIT_LCL_WIDTH = (n_proc_tiles0 * tile_sz0 + pad0 * 2);

			// number processors per output data tile
			WIT_N_TILEPROCS1 = (n_proc_tiles1 > 4) ? 8 : (n_proc_tiles1 > 2) ? 4 : (n_proc_tiles1 > 1) ? 2 : 1;
			WIT_N_TILEPROCS0 = (n_proc_tiles0 > 4) ? 8 : (n_proc_tiles0 > 2) ? 4 : (n_proc_tiles0 > 1) ? 2 : 1;

			WIT_N_TILEPROCS1 = (win2x2_ && n_proc_tiles1 > 8) ? 16 : WIT_N_TILEPROCS1;
			WIT_N_TILEPROCS0 = (win2x2_ && n_proc_tiles0 > 8) ? 16 : WIT_N_TILEPROCS0;

			// n of tiles (channels) processed per dimension
			wit_n_proctiles1 = grp_dim1 / WIT_N_TILEPROCS1;
			wit_n_proctiles0 = grp_dim0 / WIT_N_TILEPROCS0;

			// total number of input channels processed
			WMIT_LCL_N_IN_CHNLS = WIT_LCL_N_IN_CHNLS = wit_n_proctiles1 * wit_n_proctiles0;

			WIT_IN_PROC_SZ = (WIT_GRP_SZ0 / WIT_LCL_N_IN_CHNLS);

			// filling input tranform control map
			for (int i = 0; i < WIT_GRP_SZ0; ++i)
			{
				// input proc id
				((int*)input_transform_ctlmap_)[i * 4 + 0] = i % WIT_IN_PROC_SZ;
				// input channel
				((int*)input_transform_ctlmap_)[i * 4 + 1] = i / WIT_IN_PROC_SZ;

				// WIT_N_TILEPROCS1xWIT_N_TILEPROCS0 tile
				((int*)input_transform_ctlmap_)[(WIT_GRP_SZ0 + i) * 4 + 0] = (i % grp_dim0) % WIT_N_TILEPROCS0;
				((int*)input_transform_ctlmap_)[(WIT_GRP_SZ0 + i) * 4 + 1] = (i / grp_dim0) % WIT_N_TILEPROCS1;
				// processing tile id
				((int*)input_transform_ctlmap_)[(WIT_GRP_SZ0 + i) * 4 + 2] = (i % grp_dim0) / WIT_N_TILEPROCS0;
				((int*)input_transform_ctlmap_)[(WIT_GRP_SZ0 + i) * 4 + 3] = (i / grp_dim0) / WIT_N_TILEPROCS1;
			}


			WMIT_LCL_N_OUT_CHNLS = WMIT_LCL_N_IN_CHNLS * 4;
			WMIT_LCL_N_OUT_CHNLS = (outputs > (WMIT_LCL_N_OUT_CHNLS / 2)) ? WMIT_LCL_N_OUT_CHNLS : (outputs > WMIT_LCL_N_OUT_CHNLS / 4) ? WMIT_LCL_N_OUT_CHNLS / 2 : WMIT_LCL_N_OUT_CHNLS / 4;
			WMIT_GRP_SZ0 = std::min(4, (WMIT_LCL_N_OUT_CHNLS + WIT_LCL_N_IN_CHNLS - 1) / WIT_LCL_N_IN_CHNLS) * 64;

		}

		int wit_lcl_sz1 = (WMIT_GRP_SZ0 == 256) ? 16 : (WMIT_GRP_SZ0 == 192) ? 12 : 8;
		int wit_lcl_sz0 = (WMIT_GRP_SZ0 >= 128) ? 16 : 8;

		// inverse control map
		inverse_transform_ctlmap_ = malloc(WMIT_GRP_SZ0 * 2 * sizeof(int) * 4); // control map
		memset(inverse_transform_ctlmap_, 0, WMIT_GRP_SZ0 * 2 * sizeof(int) * 4);
		adnn_data_parameters inv_ctlmap_params;
		memset(&inv_ctlmap_params, 0, sizeof(adnn_data_parameters));
		inv_ctlmap_params.data_format = ADNN_DF_I32;
		inv_ctlmap_params.batch_format = ADNN_BF_NW;
		inv_ctlmap_params.dims[0] = WMIT_GRP_SZ0 * 2;
		inv_ctlmap_params.dims[1] = 4;
		aDNNTensor & inv_ctlmap = createSlot(getTopNm() + ADNN_WIN_TRANSFORM_NM + ADNN_WIN_CTLMAP_NM, inv_ctlmap_params);

		if (WIT_BIG)
		{
			// filling inverse tranform control map

			for (int i = 0; i < WMIT_GRP_SZ0; ++i)
			{
				// lcl 0
				((int*)inverse_transform_ctlmap_)[i * 4 + 2] = (i % (wit_lcl_sz0 * wit_lcl_sz1)) % wit_lcl_sz0;
				// lcl 1
				((int*)inverse_transform_ctlmap_)[i * 4 + 3] = (i % (wit_lcl_sz0 * wit_lcl_sz1)) / wit_lcl_sz0;

				// out of 4 tiles
				((int*)inverse_transform_ctlmap_)[(WIT_GRP_SZ0 + i) * 4 + 0] = (i % (WMIT_GRP_SZ0 / WMIT_LCL_N_OUT_CHNLS)) % 8;
				((int*)inverse_transform_ctlmap_)[(WIT_GRP_SZ0 + i) * 4 + 1] = (i % (WMIT_GRP_SZ0 / WMIT_LCL_N_OUT_CHNLS)) / 8;
				// 1 processing tile
				((int*)inverse_transform_ctlmap_)[(WIT_GRP_SZ0 + i) * 4 + 2] = (i / (WMIT_GRP_SZ0 / WMIT_LCL_N_OUT_CHNLS));
				// 1 input tile
				((int*)inverse_transform_ctlmap_)[(WIT_GRP_SZ0 + i) * 4 + 3] = 0;
			}
		}
		else
		{

			// filling inverse tranform control map

			for (int i = 0; i < WMIT_GRP_SZ0; ++i)
			{
				// input proc id
				((int*)inverse_transform_ctlmap_)[i * 4 + 0] = i % (WMIT_GRP_SZ0 / WMIT_LCL_N_IN_CHNLS);
				// input channel
				((int*)inverse_transform_ctlmap_)[i * 4 + 1] = i / (WMIT_GRP_SZ0 / WMIT_LCL_N_IN_CHNLS);

				// out of 4 tiles
				((int*)inverse_transform_ctlmap_)[(WIT_GRP_SZ0 + i) * 4 + 0] = (i % (WMIT_GRP_SZ0 / WMIT_LCL_N_OUT_CHNLS)) % wit_n_proctiles0;
				((int*)inverse_transform_ctlmap_)[(WIT_GRP_SZ0 + i) * 4 + 1] = (i % (WMIT_GRP_SZ0 / WMIT_LCL_N_OUT_CHNLS)) / wit_n_proctiles0;
				// processing tile
				((int*)inverse_transform_ctlmap_)[(WIT_GRP_SZ0 + i) * 4 + 2] = i / (WMIT_GRP_SZ0 / WMIT_LCL_N_OUT_CHNLS);
				// input tile
				((int*)inverse_transform_ctlmap_)[(WIT_GRP_SZ0 + i) * 4 + 3] = i / (WMIT_GRP_SZ0 / WMIT_LCL_N_IN_CHNLS);
			}
		}


		int WIT_LCL_LOG2N_IN_CHNLS = (int)ceil(log((double)WIT_LCL_N_IN_CHNLS) / log(2.));
		int WIT_LCL_STRIDE = WIT_LCL_WIDTH;			// local memory stride ADNN_LCL_WIDTH
		int WIT_LCL_SZ = WIT_LCL_STRIDE * WIT_LCL_HEIGHT;						// local memory size (ADNN_LCL_STRIDE * ADNN_LCL_HEIGHT)
		int WMIT_IN_PROC_SZ = WMIT_GRP_SZ0 / WMIT_LCL_N_IN_CHNLS;
		int WIT_LCL_TFM_WIDTH = n_tiles_TILE0 * block_sz0;    // input transformed size
		int WIT_LCL_TFM_HEIGHT = n_tiles_TILE1 * block_sz1;



		// multiply Hard Wired for now
		int WMT_LCL_N_OUT_CHNLS = block_sz1;
		int WMT_PIXPER_WKITM = block_sz0;
		int WMT_WKITMSPER_BLOCK = WMT_LCL_N_OUT_CHNLS;
		int WMT_GRP_SZ0 = (WMT_PIXPER_WKITM == 6) ? 192 : 256;
		// has to be even to be able to traspose 2 halfs
		//
		int WMT_BLOCKSPER_GROUP = WMT_GRP_SZ0 / WMT_WKITMSPER_BLOCK;

		// stand-alone inverse
		int WSIT_GRP_SZ0 = 8;
		int WSIT_GRP_SZ1 = 8;
		int WSIT_GRP_SZ2 = 1;


		std::string comp_options =
			std::string("-D ADNN_WWT_GRP_SZ0=") + std::to_string((long long)WWT_GRP_SZ0)
			+ std::string(" -D ADNN_WWT_GRP_SZ1=") + std::to_string((long long)WWT_GRP_SZ1)
			+ std::string(" -D ADNN_WWT_GRP_SZ2=") + std::to_string((long long)WWT_GRP_SZ2)
			+ std::string(" -D ADNN_WWT_PEROUTPUT=") + std::to_string((long long)wwt_peroutput)
			+ std::string(" -D ADNN_W_TILE1=") + std::to_string((long long)tile_sz1)
			+ std::string(" -D ADNN_W_TILE0=") + std::to_string((long long)tile_sz0)
			+ std::string(" -D ADNN_W_BLOCK1=") + std::to_string((long long)block_sz1)
			+ std::string(" -D ADNN_W_BLOCK0=") + std::to_string((long long)block_sz0)
			+ std::string(" -D ADNN_W_N_ICHNLS=") + std::to_string((long long)inputs)   // total number of input channels
			+ std::string(" -D ADNN_W_N_OCHNLS=") + std::to_string((long long)outputs)              // total number of output channels
			+ std::string(" -D ADNN_W_FLTR1=") + std::to_string((long long)(kernel_size1))
			+ std::string(" -D ADNN_W_FLTR0=") + std::to_string((long long)(kernel_size0))
			+ std::string(" -D ADNN_W_PAD1=") + std::to_string((long long)(pad1))
			+ std::string(" -D ADNN_W_PAD0=") + std::to_string((long long)(pad0))
			+ std::string(" -D ADNN_WEI_SZ=") + std::to_string((long long)weights_sz)
			+ std::string(" -D ADNN_WIT_GRP_SZ0=") + std::to_string((long long)WIT_GRP_SZ0)				//input transform group size in dim 0
			+ std::string(" -D ADNN_WIT_GRP_LG2SZ0=") + std::to_string((long long)WIT_GRP_LG2SZ0)			//log
			+ std::string(" -D ADNN_WIT_GRP_SZ1=") + std::to_string((long long)WIT_GRP_SZ1)				//input transform group size in dim 1
			+ std::string(" -D ADNN_WIT_GRP_SZ2=") + std::to_string((long long)WIT_GRP_SZ2)				//input transform group size in dim 2
			+ std::string(" -D ADNN_WIT_OUTLINEAR=") + std::to_string((long long)wit_outlinear)
			+ std::string(" -D ADNN_W_BATCH_SZ=") + std::to_string((long long)batch_sz)				//batch size
			+ std::string(" -D ADNN_W_HEIGH=") + std::to_string((long long)height)                  //input height
			+ std::string(" -D ADNN_W_WIDTH=") + std::to_string((long long)width)					//input width
			+ std::string(" -D ADNN_W_ISTRIDE=") + std::to_string((long long)bot_stride)				//input stride
			+ std::string(" -D ADNN_W_ICHNL_STRIDE=") + std::to_string((long long)bot_channel_stride)			//input channel stride
			+ std::string(" -D ADNN_W_IBATCH_STRIDE=") + std::to_string((long long)bot_batch_stride)          //input batch stride
			+ std::string(" -D ADNN_W_BSTRIDE=") + std::to_string((long long)tbot_stride)				//blocked(trasformed input) stride
			+ std::string(" -D ADNN_W_BCHNL_STRIDE=") + std::to_string((long long)tbot_channel_stride)			//blocked(trasformed input) channel stride
			+ std::string(" -D ADNN_W_BBATCH_STRIDE=") + std::to_string((long long)tbot_batch_stride)			//blocked(trasformed input) batch stride
			+ std::string(" -D ADNN_WIT_LCL_N_IN_CHNLS=") + std::to_string((long long)WIT_LCL_N_IN_CHNLS)			//n of localy kept input channels
			+ std::string(" -D ADNN_WIT_LCL_LOG2N_IN_CHNLS=") + std::to_string((long long)WIT_LCL_LOG2N_IN_CHNLS)		//log2 of n of localy kept input channels
			+ std::string(" -D ADNN_WIT_N_IN_TILES1=") + std::to_string((long long)wit_n_tiles1)			//	n total input tiles dim 1 for > 32x32
			+ std::string(" -D ADNN_WIT_N_IN_TILES0=") + std::to_string((long long)wit_n_tiles1)			//	n total input tiles dim 0 for > 32x32
			+ std::string(" -D ADNN_WIT_IN_TILE_LOG2SZ1=") + std::to_string((long long)tile_log2sz1)         // log size of input tile for > 32x32
			+ std::string(" -D ADNN_WIT_IN_TILE_LOG2SZ0=") + std::to_string((long long)tile_log2sz0)         // log size of input tile for > 32x32
			+ std::string(" -D ADNN_WIT_IN_PROC_SZ=") + std::to_string((long long)WIT_IN_PROC_SZ)				//n of processors reading 1 input channel for < 32x32 :  (1 << (ADNN_WIT_GRP_LG2SZ0 - ADNN_LCL_LOG2N_IN_CHNLS))
			+ std::string(" -D ADNN_WIT_READ_SZ1=") + std::to_string((long long)WIT_READ_SZ1)					//length to read dim 1
			+ std::string(" -D ADNN_WIT_READ_SZ0=") + std::to_string((long long)WIT_READ_SZ0)				//lenght to read dim 0
			+ std::string(" -D ADNN_WIT_N_TILEPROCS1=") + std::to_string((long long)WIT_N_TILEPROCS1)          // n procs in proc tile dim 1
			+ std::string(" -D ADNN_WIT_N_TILEPROCS0=") + std::to_string((long long)WIT_N_TILEPROCS0)          //  n procs in proc tile dim 0
			+ std::string(" -D ADNN_WIT_BLKD_TILE_SZ1=") + std::to_string((long long)WIT_BLKD_TILE_SZ1)				//size of output tiles dim 1 for > 32x32
			+ std::string(" -D ADNN_WIT_BLKD_TILE_SZ0=") + std::to_string((long long)WIT_BLKD_TILE_SZ0)			//	size of output tiles dim 0 for > 32x32
			+ std::string(" -D ADNN_WIT_BIG=") + std::to_string((long long)WIT_BIG)                //  input > 32x32
			+ std::string(" -D ADNN_WIT_IN_SIZE=") + std::to_string((long long)WIT_IN_SIZE)            // input block size
			+ std::string(" -D ADNN_WIT_LCL_HEIGHT=") + std::to_string((long long)WIT_LCL_HEIGHT)				// local memory height(ADNN_IN_HEIGHT + ADNN_W_PAD1 * 2)
			+ std::string(" -D ADNN_WIT_LCL_WIDTH=") + std::to_string((long long)WIT_LCL_WIDTH)				// local memory width(ADNN_IN_WIDTH + ADNN_W_PAD0 * 2)
			+ std::string(" -D ADNN_WIT_LCL_STRIDE=") + std::to_string((long long)WIT_LCL_STRIDE)				// local memory stride ADNN_LCL_WIDTH
			+ std::string(" -D ADNN_WIT_LCL_SZ=") + std::to_string((long long)WIT_LCL_SZ)					// local memory size(ADNN_LCL_STRIDE * ADNN_LCL_HEIGHT)
			+ std::string(" -D ADNN_WMIT_GRP_SZ0=") + std::to_string((long long)WMIT_GRP_SZ0)
			+ std::string(" -D ADNN_WMIT_GRP_SZ1=") + std::to_string((long long)1)
			+ std::string(" -D ADNN_WMIT_GRP_SZ2=") + std::to_string((long long)1)
			+ std::string(" -D ADNN_WMIT_LCL_N_IN_CHNLS=") + std::to_string((long long)WMIT_LCL_N_IN_CHNLS)    //      n of local input channels - the  same id but from different batchs
			+ std::string(" -D ADNN_WMIT_LCL_N_OUT_CHNLS=") + std::to_string((long long)WMIT_LCL_N_OUT_CHNLS)    //      n of local output channels
			+ std::string(" -D ADNN_WMIT_IN_PROC_SZ=") + std::to_string((long long)WMIT_IN_PROC_SZ)           //  n of processors reading 1 input channel
			+ std::string(" -D ADNN_W_BHEIGH=") + std::to_string((long long)theight)                  // height of the transformed buffer
			+ std::string(" -D ADNN_W_BWIDTH=") + std::to_string((long long)twidth)                 // width of tarnsformed data buffer in pixels
			+ std::string(" -D ADNN_W_OBBATCH_STRIDE=") + std::to_string((long long)ttop_batch_stride)
			+ std::string(" -D ADNN_W_OBCHNL_STRIDE=") + std::to_string((long long)ttop_channel_stride)
			+ std::string(" -D ADNN_W_OBSTRIDE=") + std::to_string((long long)ttop_stride)
			+ std::string(" -D ADNN_W_OBATCH_STRIDE=") + std::to_string((long long)top_batch_stride)
			+ std::string(" -D ADNN_W_OCHNL_STRIDE=") + std::to_string((long long)top_channel_stride)
			+ std::string(" -D ADNN_W_OSTRIDE=") + std::to_string((long long)top_stride)
			+ std::string(" -D ADNN_WMT_GRP_SZ0=") + std::to_string((long long)WMT_GRP_SZ0)			// group size
			+ std::string(" -D ADNN_WMT_GRP_SZ1=") + std::to_string((long long)1)
			+ std::string(" -D ADNN_WMT_GRP_SZ2=") + std::to_string((long long)1)
			+ std::string(" -D ADNN_WMT_LCL_N_OUT_CHNLS=") + std::to_string((long long)WMT_LCL_N_OUT_CHNLS) // n of local output channels
			+ std::string(" -D ADNN_WMT_PIXPER_WKITM=") + std::to_string((long long)WMT_PIXPER_WKITM)    // n input pixels per wk-item
			+ std::string(" -D ADNN_WMT_WKITMSPER_BLOCK=") + std::to_string((long long)WMT_WKITMSPER_BLOCK) // n wk items per block
			+ std::string(" -D ADNN_WMT_BLOCKSPER_GROUP=") + std::to_string((long long)WMT_BLOCKSPER_GROUP) // n blocks per group
			+ std::string(" -D ADNN_N_BLOCKS0=") + std::to_string((long long)n_proc_tiles0)
			+ std::string(" -D ADNN_WSIT_GRP_SZ0=") + std::to_string((long long)WSIT_GRP_SZ0)
			+ std::string(" -D ADNN_WSIT_GRP_SZ1=") + std::to_string((long long)WSIT_GRP_SZ1)
			+ std::string(" -D ADNN_WSIT_GRP_SZ2=") + std::to_string((long long)WSIT_GRP_SZ2)
			+ std::string(" -D ADNN_WIT_LCL_TFM_WIDTH=") + std::to_string((long long)WIT_LCL_TFM_WIDTH)    // input transformed size
			+ std::string(" -D ADNN_WIT_LCL_TFM_HEIGHT=") + std::to_string((long long)WIT_LCL_TFM_HEIGHT)    // input transformed size


			+ std::string(" -D ADNN_ALG_WIN2x2_3x3=") + std::to_string((long long)(ALG_WIN2x2_3x3))
			+ std::string(" -D ADNN_ELEMENT_WISE_TRANSOFRMS=") + std::to_string((long long)(ELEMENT_WISE_TRANSOFRMS))
			+ getGenericCompOptions();

		std::string kernel_file = "aDNNConv_Win4x4_NCHW.cl";

		if (win2x2_)
		{
			kernel_file = "aDNNConv_Win2x2_NCHW.cl";
		}

		std::string kernel_name = "Weights_Tansform_Win";

		// execution setup


		std::vector<size_t> l_wk;
		l_wk.push_back(WWT_GRP_SZ0);
		l_wk.push_back(WWT_GRP_SZ1);
		l_wk.push_back(WWT_GRP_SZ2);


		std::vector<size_t> g_wk;
		g_wk.push_back(inputs);
		g_wk.push_back(outputs);
		g_wk.push_back(1);



		CDNN_OCL_kern_exe kern_exe0(this, kernel_name, kernel_file, comp_options, 0, &g_wk, &l_wk, 0);

		kern_exe0.Construct();

		ocl_fwd_execs_.push_back(kern_exe0);

		if (!win2x2_)
		{

			kernel_name = "Input_Tansform_Win";
		}
		else
		{
			kernel_name = "Conv_Win2x2_3x3";

		}
		// execution setup


		l_wk.clear();
		l_wk.push_back(WIT_GRP_SZ0);
		l_wk.push_back(WIT_GRP_SZ1);
		l_wk.push_back(WIT_GRP_SZ2);

		int glbl2 = 0;
		if (win2x2_)
		{
			glbl2 = (outputs + WMIT_LCL_N_OUT_CHNLS - 1) / WMIT_LCL_N_OUT_CHNLS;
		}
		else
		{
			glbl2 = (inputs + WIT_LCL_N_IN_CHNLS - 1) / WIT_LCL_N_IN_CHNLS;
		}
		g_wk.clear();
		g_wk.push_back(wit_n_tiles1 * wit_n_tiles0 * WIT_GRP_SZ0);
		g_wk.push_back(glbl2);
		g_wk.push_back(batch_sz);



		CDNN_OCL_kern_exe kern_exe1(this, kernel_name, kernel_file, comp_options, 0, &g_wk, &l_wk, 0);

		kern_exe1.Construct();

		ocl_fwd_execs_.push_back(kern_exe1);

		if (!win2x2_)
		{

			//		kernel_name = "Mult_Win2";
			kernel_name = "MultInv_Win2";

			// execution setup


			l_wk.clear();
			l_wk.push_back(WMT_GRP_SZ0);
			l_wk.push_back(1);
			l_wk.push_back(1);

			int n_mt_output_blocks = (outputs + WMT_LCL_N_OUT_CHNLS - 1) / WMT_LCL_N_OUT_CHNLS;
			int n_mt_batch_blocks = (batch_sz + WMT_LCL_N_OUT_CHNLS - 1) / WMT_LCL_N_OUT_CHNLS;
			g_wk.clear();
			g_wk.push_back(n_proc_tiles1 * n_proc_tiles0 * WMT_WKITMSPER_BLOCK);
			g_wk.push_back(n_mt_output_blocks);
			g_wk.push_back(batch_sz);



			CDNN_OCL_kern_exe kern_exe2(this, kernel_name, kernel_file, comp_options, 0, &g_wk, &l_wk, 0);

			kern_exe2.Construct();

			ocl_fwd_execs_.push_back(kern_exe2);
		}

#if 0
		kernel_name = "Inv_Win";

		// execution setup


		l_wk.clear();
		l_wk.push_back(WSIT_GRP_SZ0);
		l_wk.push_back(WSIT_GRP_SZ1);
		l_wk.push_back(WSIT_GRP_SZ2);

		g_wk.clear();
		g_wk.push_back(n_proc_tiles1);
		g_wk.push_back(n_proc_tiles0);
		g_wk.push_back(batch_sz * outputs);



		CDNN_OCL_kern_exe kern_exe3(this, kernel_name, kernel_file, comp_options, 0, &g_wk, &l_wk, 0);

		kern_exe3.Construct();

		ocl_fwd_execs_.push_back(kern_exe3);
#endif

		return(ret);
	}

	int aDNNodeConv::BuildFwdWin_NCHW(void)
	{
		int ret = 0;

		// weight transform
		aDNNTensor & tweights = getSlot(getWeightsNm() + ADNN_WIN_TWEIGHTS_NM);
		tweights.allocTensor();

		if (getDebugLevel() & 1)
		{
			aDNNTensor & tweights_vf = getSlot(getWeightsNm() + ADNN_WIN_TWEIGHTS_NM + ADNN_VERIFY_NM);
			tweights_vf.allocTensor(_CBUF_MEM_SYS_ONLY);
			aDNNTensor & tbot_vf = getSlot(getBotNm() + ADNN_WIN_TRANSFORM_NM + ADNN_VERIFY_NM);
			tbot_vf.allocTensor(_CBUF_MEM_SYS_ONLY);
			aDNNTensor & ttop_vf = getSlot(getTopNm() + ADNN_WIN_TRANSFORM_NM + ADNN_VERIFY_NM);
			ttop_vf.allocTensor(_CBUF_MEM_SYS_ONLY);
		}

		// memory has to bealocated utside of the pipeline by user
		const aDNNTensor & wei = getBotWeightsFwd();

		cl_mem weights_mem = wei.getOCLBuffer();
		cl_mem tweights_mem = tweights.getOCLBuffer();

		// pass all arguments once
		int n_arg = 0;
		ocl_args kern_args;
		if (tweights_mem)
		{
			kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &tweights_mem);
		}
		n_arg++;
		if (weights_mem)
		{
			kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &weights_mem);
		}
		n_arg++;



		ocl_fwd_execs_[0].Build(kern_args);


			// input transform

			const aDNNTensor & bot = getBotFwd();

			aDNNTensor & tinput = getSlot(getBotNm() + ADNN_WIN_TRANSFORM_NM);
			tinput.allocTensor();
			// fill comtrol map
			aDNNTensor & inp_ctlmap = getSlot(getBotNm() + ADNN_WIN_TRANSFORM_NM + ADNN_WIN_CTLMAP_NM);
			inp_ctlmap.allocTensor();
			void * inp_ctlmap_ptr = inp_ctlmap.accessTensor(ADNN_MEM_ACCESS_WRITE);
			memcpy(inp_ctlmap_ptr, input_transform_ctlmap_, inp_ctlmap.getSizeInBytes());
			inp_ctlmap.commitTensor();

			aDNNTensor & ttop = getSlot(getTopNm() + ADNN_WIN_TRANSFORM_NM);
			ttop.allocTensor();
			cl_mem ttop_mem = ttop.getOCLBuffer();

			const aDNNTensor & top = getTopFwd();

			cl_mem top_mem = top.getOCLBuffer();

			cl_mem bot_mem = bot.getOCLBuffer();
			cl_mem tbot_mem = tinput.getOCLBuffer();
			cl_mem ctlmap_mem = inp_ctlmap.getOCLBuffer();

			int inputs = (int)bot.getDim(aDNN_TENSOR_DEPTH);

			aDType padding_val = 0;

			n_arg = 0;
			kern_args.clear();
			if (win2x2_)
			{
				kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &top_mem);

				n_arg++;
			}
			if (tbot_mem)
			{
				kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &tbot_mem);
			}
			n_arg++;
			if (win2x2_)
			{
				kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &ttop_mem);

				n_arg++;
			}

			if (bot_mem)
			{
				kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &bot_mem);
			}
			n_arg++;

			if (win2x2_)
			{
				kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &tweights_mem);

				n_arg++;
			}
			kern_args[n_arg] = std::make_pair(sizeof(aDType), &padding_val);
			n_arg++;
			if (ctlmap_mem)
			{
				kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &ctlmap_mem);
			}
			n_arg++;

			if (win2x2_)
			{
				kern_args[n_arg] = std::make_pair(sizeof(int), &inputs);
				n_arg++;
			}
			ocl_fwd_execs_[1].Build(kern_args);

//			exit(0);


			if (!win2x2_)
			{




			// mult
			//		aDNNTensor & tbot = getSlot(getBotNm() + ADNN_WIN_TRANSFORM_NM);
			//		cl_mem tbot_mem = tbot.getOCLBuffer();

			n_arg = 0;
			kern_args.clear();
			if (ttop_mem)
			{
				//			kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &ttop_mem);
				kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &top_mem);
			}
			n_arg++;
			if (tbot_mem)
			{
				kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &tbot_mem);
			}
			n_arg++;
			if (tweights_mem)
			{
				kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &tweights_mem);
			}
			n_arg++;
			kern_args[n_arg] = std::make_pair(sizeof(int), &inputs);
			n_arg++;


			ocl_fwd_execs_[2].Build(kern_args);

		}
#if 0
		// inverse transform

//		const aDNNTensor & top = getTopFwd();

		// fill comtrol map
//		aDNNTensor & inv_ctlmap = getSlot(getTopNm() + ADNN_WIN_TRANSFORM_NM + ADNN_WIN_CTLMAP_NM);
//		inv_ctlmap.allocTensor();
//		void * inv_ctlmap_ptr = inv_ctlmap.accessTensor(ADNN_MEM_ACCESS_WRITE);
//		memcpy(inv_ctlmap_ptr, inverse_transform_ctlmap_, inv_ctlmap.getSizeInBytes());
//		inv_ctlmap.commitTensor();

//		cl_mem top_mem = top.getOCLBuffer();
//		ctlmap_mem = inv_ctlmap.getOCLBuffer();

		n_arg = 0;
		kern_args.clear();
		if (top_mem)
		{
			kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &top_mem);
		}
		n_arg++;
		if (tbot_mem)
		{
			kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &ttop_mem);
		}
		n_arg++;


		ocl_fwd_execs_[3].Build(kern_args);
#endif


		return(ret);
	}


	int aDNNodeConv::RunFwdWin_NCHW(const adnn_node_parameters * running_params)
	{
		int ret = 0;



		// execute through specialized object
		ocl_args additional_args;

		if (running_params)
		{
			update(*running_params);

			int n_arg = 0;

			if (getInputEdge().isWeightsUpdated())
			{
				cl_mem weights_mem = ((aDNNTensor &)getInputEdge().getWeightsData()).getOCLBuffer();
				getInputEdge().setWeightsUpdated(false);
				additional_args[1] = std::make_pair(sizeof(cl_mem), &weights_mem);
			}

		}
		// forward
		int iter = getNTimingIter();


		double s = 0, e = 0;
		if (isPerLayerTiming())
		{
			s = mach_absolute_time();
		}

		for (int i = 0; i < iter; i++)
		{

			ocl_fwd_execs_[0].ExecuteNoWait(&additional_args);
			ocl_fwd_execs_[1].ExecuteNoWait();

			if (!win2x2_)
			{
				ocl_fwd_execs_[2].ExecuteNoWait();
			}
//			ocl_fwd_execs_[3].ExecuteNoWait();
		}

		if (isPerLayerTiming())
		{
			clFinish(ocl_fwd_execs_[0].getOclQueue());
			e = mach_absolute_time();
		}
		// verify

		if (getDebugLevel() == 1)
		{
			ret = VerifyFwdWin();
		}

		if (isPerLayerMessaging())
		{
			const aDNNTensor & bot = getBotFwd();
			const aDNNTensor & top = getTopFwd();
			const aDNNTensor & wei = getBotWeightsFwd();

			int width = (int)top.getDim(aDNN_TENSOR_WIDTH);
			int height = (int)top.getDim(aDNN_TENSOR_HEIGHT);
			int outputs = (int)top.getDim(aDNN_TENSOR_DEPTH);

			// TO DO: check top, bot dim are equal
			int kernel_size = getKernelSz();
			int pad = getPad();
			int stride = getKernelStride();
			int batch_sz = (int)bot.getDim(aDNN_TENSOR_BATCH);
			int inputs = (int)bot.getDim(aDNN_TENSOR_DEPTH);
			iter = (iter <= 0) ? 1 : iter;
			processing_time_ = subtractTimes(e, s);
			int ident = 4;
			printf("Passed layer: convolution: \"%s\"\n", getName().c_str());
			printf("%*s" "Arguments: CxWxHxKxKxOxB: %dx%dx%dx%dx%dx%dx%d\n", ident, " ", inputs, width, height, kernel_size, kernel_size, outputs, batch_sz);
			if (isPerLayerTiming())
			{
				printf("%*s" "Performance: %6.2f ms, %6.3f TFLOPs\n", ident, " ", processing_time_ / iter, ((double)2 * inputs*width*height*kernel_size*kernel_size*outputs*batch_sz * iter) / (processing_time_ * 1000000000));
			}
		}

		return(ret);
	}

	/*******************************************************************************************************************************************
	**
	** c-emulator
	**
	*******************************************************************************************************************************************/

	// filter transform 1D
	static void funcGxg_2x2_3x3(int *n_mult, int *n_add, float * Gxg, const float * g)
	{

		Gxg[0] = g[0];
		float tmp = (g[0] + g[2]);
		Gxg[1] = (tmp + g[1]) * 0.5f;
		Gxg[2] = (tmp - g[1]) * 0.5f;
		Gxg[3] = g[2];

		*n_mult = 2;
		*n_add = 3;
	}

	// data transform 1D
	static void funcC_Txd_2x2_3x3(int *n_mult, int *n_add, float * C_Txd, const float * d)
	{

		C_Txd[0] = d[0] - d[2];
		C_Txd[1] = d[1] + d[2];
		C_Txd[2] = d[2] - d[1];
		C_Txd[3] = d[1] - d[3];

		*n_mult = 0;
		*n_add = 4;
	}

	// inverse transform 1D
	static void funcConv_2x2_3x3(int *n_mult, int *n_add, float * Conv, const float * ewm)
	{

		Conv[0] = ewm[0] + ewm[1] + ewm[2];
		Conv[1] = ewm[1] - ewm[2] - ewm[3];

		*n_mult = 0;
		*n_add = 4;
	}



	void Weights_Block_Tansform_2x2_3x3Win(
		int ADNN_W_BLOCK1,
		int ADNN_W_BLOCK0,
		int ADNN_W_FLTR1,
		int ADNN_W_FLTR0,
	_FLOAT *GxgxG_T, const _FLOAT *g, const _FLOAT *G,_FLOAT * Gxg, int *n_mul, int *n_add, int *n_flin)
	{

		// Gxg : ((ADNN_W_TILE1 + ADNN_W_PAD1 *2) x ADNN_W_FLTR0) * (ADNN_W_FLTR1 x ADNN_W_FLTR0).T
		// trasnposed
		 Gxg[0] = g[0];
		 Gxg[1] = g[3];
		 Gxg[2] = g[6];
	//	 _FLOAT tg0 = g[0] + g[2];
	//	 _FLOAT tg3 = g[3] + g[5];
	//	 _FLOAT tg6 = g[6] + g[8];
		 Gxg[3] = (g[0] * 0.5f + g[1] * 0.5f + g[2] * 0.5f);
		 Gxg[4] = (g[3] * 0.5f + g[4] * 0.5f + g[5] * 0.5f);
		 Gxg[5] = (g[6] * 0.5f + g[7] * 0.5f + g[8] * 0.5f);
		 Gxg[6] = (g[0] * 0.5f - g[1] * 0.5f + g[2] * 0.5f);
		 Gxg[7] = (g[3] * 0.5f - g[4] * 0.5f + g[5] * 0.5f);
		 Gxg[8] = (g[6] * 0.5f - g[7] * 0.5f + g[8] * 0.5f);
		 Gxg[9] = g[2];
		 Gxg[10] = g[5];
		 Gxg[11] = g[8];

		 *n_mul += 6;
		 *n_add += 9;
		 *n_flin += 3 + 12;

		 GxgxG_T[0] = Gxg[0];
//		 _FLOAT tGxg0 = Gxg[0] + Gxg[2];
//		 _FLOAT tGxg3 = Gxg[3] + Gxg[5];
//		 _FLOAT tGxg6 = Gxg[6] + Gxg[8];
//		 _FLOAT tGxg9 = Gxg[9] + Gxg[11];
		 GxgxG_T[1] = (Gxg[0] * 0.5f + Gxg[1] * 0.5f + Gxg[2] * 0.5f);
		 GxgxG_T[2] = (Gxg[0] * 0.5f + Gxg[2] * 0.5f - Gxg[1] * 0.5f);
		 GxgxG_T[3] = Gxg[2];
		 GxgxG_T[4] = Gxg[3];
		 GxgxG_T[5] = (Gxg[3] * 0.5f + Gxg[4] * 0.5f + Gxg[5] * 0.5f);
		 GxgxG_T[6] = (Gxg[3] * 0.5f - Gxg[4] * 0.5f + Gxg[5] * 0.5f);
		 GxgxG_T[7] = Gxg[5];
		 GxgxG_T[8] = Gxg[6];
		 GxgxG_T[9] = (Gxg[6] * 0.5f + Gxg[7] * 0.5f + Gxg[8] * 0.5f);
		 GxgxG_T[10] = (Gxg[6] * 0.5f - Gxg[7] * 0.5f + Gxg[8] * 0.5f);
		 GxgxG_T[11] = Gxg[8];
		 GxgxG_T[12] = Gxg[9];
		 GxgxG_T[13] = (Gxg[9] * 0.5f + Gxg[10] * 0.5f + Gxg[11] * 0.5f);
		 GxgxG_T[14] = (Gxg[9] * 0.5f - Gxg[10] * 0.5f + Gxg[11] * 0.5f);
		 GxgxG_T[15] = Gxg[11];

		 *n_mul += 8;
		 *n_add += 12;
		 *n_flin += 4 + 16;

	}

	void Weights_Block_Tansform_4x4_3x3Win(
		int ADNN_W_BLOCK1,
		int ADNN_W_BLOCK0,
		int ADNN_W_FLTR1,
		int ADNN_W_FLTR0,
	_FLOAT *GxgxG_T, const _FLOAT *g, const _FLOAT *G, _FLOAT * Gxg, int *n_mul, int *n_add, int *n_flin)
	{
		// Gxg : ((ADNN_W_TILE1 + ADNN_W_PAD1 *2) x ADNN_W_FLTR0) * (ADNN_W_FLTR1 x ADNN_W_FLTR0).T
		// trasnposed

#if 1
		Gxg[0] = (_FLOAT)(g[0] * 0.25);
		Gxg[1] = (_FLOAT)(g[3] * 0.25);
		Gxg[2] = (_FLOAT)(g[6] * 0.25);

		for (int j = 1; j < ADNN_W_BLOCK1 - 1; ++j)
		{
			for (int i = 0; i < ADNN_W_FLTR0; ++i)
			{
				Gxg[j*ADNN_W_FLTR0 + i] = 0;
				for (int k = 0; k < ADNN_W_FLTR0; ++k)
				{
					// g_transposes
					Gxg[j*ADNN_W_FLTR0 + i] += G[j*ADNN_W_FLTR0 + k] * g[i * ADNN_W_FLTR1 + k];
					//printf("Tw: GxgI=%d, G_v=%f gi=%d\n", j*ADNN_W_FLTR0 + i, G[j*ADNN_W_FLTR0 + k], i * ADNN_W_FLTR1 + k);
				}
			}
		}

		Gxg[15] = g[2];
		Gxg[16] = g[5];
		Gxg[17] = g[8];


#else
		for (int j = 0; j < ADNN_W_BLOCK1; ++j)
		{
			for (int i = 0; i < ADNN_W_FLTR0; ++i)
			{
				Gxg[j*ADNN_W_FLTR0 + i] = 0;
				for (int k = 0; k < ADNN_W_FLTR0; ++k)
				{
					// g_transposes
					Gxg[j*ADNN_W_FLTR0 + i] += G[j*ADNN_W_FLTR0 + k] * g[i * ADNN_W_FLTR1 + k];
					//printf("Tw: GxgI=%d, G_v=%f gi=%d\n", j*ADNN_W_FLTR0 + i, G[j*ADNN_W_FLTR0 + k], i * ADNN_W_FLTR1 + k);
				}
			}
		}

#endif

		*n_mul += 3 + (ADNN_W_BLOCK1 - 2) * ADNN_W_FLTR0 * ADNN_W_FLTR0;
		*n_add += (ADNN_W_BLOCK1 - 2) * ADNN_W_FLTR0 * ADNN_W_FLTR0;
		*n_flin = *n_mul;

		// mult on transpose G

#if 1
		GxgxG_T[0] = (_FLOAT)(Gxg[0] * 0.25);
//		GxgxG_T[1] = (_FLOAT)(Gxg[0] * (-1. / 6.) + Gxg[1] * (-1. / 6.) + Gxg[2] * (-1. / 6.));
//		GxgxG_T[2] = (_FLOAT)(Gxg[0] * (-1. / 6.) + Gxg[1] * (1. / 6.) + Gxg[2] * (-1. / 6.));
//		GxgxG_T[3] = (_FLOAT)(Gxg[0] * (1. / 24.) + Gxg[1] * (1. / 12.) + Gxg[2] * (1. / 6.));
//		GxgxG_T[4] = (_FLOAT)(Gxg[0] * (1. / 24.) + Gxg[1] * (-1. / 12.) + Gxg[2] * (1. / 6.));
		GxgxG_T[5] = Gxg[2];
		GxgxG_T[6] = (_FLOAT)(Gxg[3] * 0.25);
//		GxgxG_T[7] = (_FLOAT)(Gxg[3] * (-1. / 6.) + Gxg[4] * (-1. / 6.) + Gxg[5] * (-1. / 6.));
//		GxgxG_T[8] = (_FLOAT)(Gxg[3] * (-1. / 6.) + Gxg[4] * (1. / 6.) + Gxg[5] * (-1. / 6.));
//		GxgxG_T[9] = (_FLOAT)(Gxg[3] * (1. / 24.) + Gxg[4] * (1. / 12.) + Gxg[5] * (1. / 6.));
//		GxgxG_T[10] = (_FLOAT)(Gxg[3] * (1. / 24.) + Gxg[4] * (-1. / 12.) + Gxg[5] * (1. / 6.));
		GxgxG_T[11] = Gxg[5];
		GxgxG_T[12] = (_FLOAT)(Gxg[6] * 0.25);
//		GxgxG_T[13] = (_FLOAT)(Gxg[6] * (-1. / 6.) + Gxg[7] * (-1. / 6.) + Gxg[8] * (-1. / 6.));
//		GxgxG_T[14] = (_FLOAT)(Gxg[6] * (-1. / 6.) + Gxg[7] * (1. / 6.) + Gxg[8] * (-1. / 6.));
//		GxgxG_T[15] = (_FLOAT)(Gxg[6] * (1. / 24.) + Gxg[7] * (1. / 12.) + Gxg[8] * (1. / 6.));
//		GxgxG_T[16] = (_FLOAT)(Gxg[6] * (1. / 24.) + Gxg[7] * (-1. / 12.) + Gxg[8] * (1. / 6.));
		GxgxG_T[17] = Gxg[8];
		GxgxG_T[18] = (_FLOAT)(Gxg[9] * 0.25);
//		GxgxG_T[19] = (_FLOAT)(Gxg[9] * (-1. / 6.) + Gxg[10] * (-1. / 6.) + Gxg[11] * (-1. / 6.));
//		GxgxG_T[20] = (_FLOAT)(Gxg[9] * (-1. / 6.) + Gxg[10] * (1. / 6.) + Gxg[11] * (-1. / 6.));
//		GxgxG_T[21] = (_FLOAT)(Gxg[9] * (1. / 24.) + Gxg[10] * (1. / 12.) + Gxg[11] * (1. / 6.));
//		GxgxG_T[22] = (_FLOAT)(Gxg[9] * (1. / 24.) + Gxg[10] * (-1. / 12.) + Gxg[11] * (1. / 6.));
		GxgxG_T[23] = Gxg[11];
		GxgxG_T[24] = (_FLOAT)(Gxg[12] * 0.25);
//		GxgxG_T[25] = (_FLOAT)(Gxg[12] * (-1. / 6.) + Gxg[13] * (-1. / 6.) + Gxg[14] * (-1. / 6.));
//		GxgxG_T[26] = (_FLOAT)(Gxg[12] * (-1. / 6.) + Gxg[13] * (1. / 6.) + Gxg[14] * (-1. / 6.));
//		GxgxG_T[27] = (_FLOAT)(Gxg[12] * (1. / 24.) + Gxg[13] * (1. / 12.) + Gxg[14] * (1. / 6.));
//		GxgxG_T[28] = (_FLOAT)(Gxg[12] * (1. / 24.) + Gxg[13] * (-1. / 12.) + Gxg[14] * (1. / 6.));
		GxgxG_T[29] = Gxg[14];
		GxgxG_T[30] = (_FLOAT)(Gxg[15] * 0.25);
//		GxgxG_T[31] = (_FLOAT)(Gxg[15] * (-1. / 6.) + Gxg[16] * (-1. / 6.) + Gxg[17] * (-1. / 6.));
//		GxgxG_T[32] = (_FLOAT)(Gxg[15] * (-1. / 6.) + Gxg[16] * (1. / 6.) + Gxg[17] * (-1. / 6.));
//		GxgxG_T[33] = (_FLOAT)(Gxg[15] * (1. / 24.) + Gxg[16] * (1. / 12.) + Gxg[17] * (1. / 6.));
//		GxgxG_T[34] = (_FLOAT)(Gxg[15] * (1. / 24.) + Gxg[16] * (-1. / 12.) + Gxg[17] * (1. / 6.));
		GxgxG_T[35] = Gxg[17];

		for (int j = 0; j < ADNN_W_BLOCK1; ++j)
		{
			for (int i = 1; i < ADNN_W_BLOCK0 - 1; ++i)
			{
				GxgxG_T[j * ADNN_W_BLOCK0 + i] = 0;
				for (int k = 0; k < ADNN_W_FLTR0; ++k)
				{
					// G transposed
					GxgxG_T[j * ADNN_W_BLOCK0 + i] += Gxg[j * ADNN_W_FLTR0 + k] * G[i * ADNN_W_FLTR0 + k];
//					printf("wT: oi =%d ii =%d G_v=%f\n", j * ADNN_W_BLOCK0 + i, j * ADNN_W_FLTR0 + k, G[i * ADNN_W_FLTR0 + k]);
				}
			}
		}

#else
		for (int j = 0; j < ADNN_W_BLOCK1; ++j)
		{
			for (int i = 0; i < ADNN_W_BLOCK0; ++i)
			{
				GxgxG_T[j * ADNN_W_BLOCK0 + i] = 0;
				for (int k = 0; k < ADNN_W_FLTR0; ++k)
				{
					// G transposed
					GxgxG_T[j * ADNN_W_BLOCK0 + i] += Gxg[j * ADNN_W_FLTR0 + k] * G[i * ADNN_W_FLTR0 + k];
					printf("wT: oi =%d ii =%d G_v=%f\n", j * ADNN_W_BLOCK0 + i, j * ADNN_W_FLTR0 + k, G[i * ADNN_W_FLTR0 + k]);
				}
			}
		}
#endif
		*n_mul += 6 + ADNN_W_BLOCK1 * (ADNN_W_BLOCK0 - 2) * ADNN_W_FLTR0;
		*n_add += ADNN_W_BLOCK1 * (ADNN_W_BLOCK0 - 2) * ADNN_W_FLTR0;
		*n_flin = *n_mul;


	}



	void Weights_Block_Tansform_Win(
		int ADNN_W_BLOCK1,
		int ADNN_W_BLOCK0,
		int ADNN_W_FLTR1,
		int ADNN_W_FLTR0,
		_FLOAT *GxgxG_T, const _FLOAT *g, const _FLOAT *G, _FLOAT * Gxg, int *n_mul, int *n_add, int *n_flin)
	{

		// Gxg : ((ADNN_W_TILE1 + ADNN_W_PAD1 *2) x ADNN_W_FLTR0) * (ADNN_W_FLTR1 x ADNN_W_FLTR0).T
		// trasnposed
#if ADNN_ELEMENT_WISE_TRANSOFRMS


#if ADNN_ALG_WIN2x2_3x3
	Weights_Block_Tansform_2x2_3x3Win
#else
	Weights_Block_Tansform_4x4_3x3Win
#endif
		(
		ADNN_W_BLOCK1,
		ADNN_W_BLOCK0,
		ADNN_W_FLTR1,
		ADNN_W_FLTR0,
		GxgxG_T, g, G, Gxg, n_mul, n_add, n_flin);


#else


		for (int j = 0; j < ADNN_W_BLOCK1; ++j)
		{
			for (int i = 0; i < ADNN_W_FLTR0; ++i)
			{
				Gxg[j*ADNN_W_FLTR0 + i] = 0;
				for (int k = 0; k < ADNN_W_FLTR0; ++k)
				{
					// g_transposes
					Gxg[j*ADNN_W_FLTR0 + i] += G[j*ADNN_W_FLTR0 + k] * g[i * ADNN_W_FLTR1 + k];
//					printf("Tw: bj=%d, bi=%d GxgI =%d Gi=%d G_v=%f gi=%d\n", j, i, j*ADNN_W_FLTR0 + i, j*ADNN_W_FLTR0 + k, G[j*ADNN_W_FLTR0 + k], i * ADNN_W_FLTR1 + k);
				}
			}
		}

		// mult on transpose G
		for (int j = 0; j < ADNN_W_BLOCK1; ++j)
		{
			for (int i = 0; i < ADNN_W_BLOCK0; ++i)
			{
				GxgxG_T[j * ADNN_W_BLOCK0 + i] = 0;
				for (int k = 0; k < ADNN_W_FLTR0; ++k)
				{
					// G transposed
					GxgxG_T[j * ADNN_W_BLOCK0 + i] += Gxg[j * ADNN_W_FLTR0 + k] * G[i * ADNN_W_FLTR0 + k];
//					printf("wT: Gxg_Ti =%d GxgI =%d Gi=%d G_v=%f\n", j * ADNN_W_BLOCK0 + i, j * ADNN_W_FLTR0 + k, i * ADNN_W_FLTR0 + k, G[i * ADNN_W_FLTR0 + k]);
				}
			}
		}

#endif
	}


	// transforms weights
	void Weights_Tansform_Win(
		int ADNN_W_N_ICHNLS,
		int ADNN_W_N_OCHNLS,
		int ADNN_W_BLOCK1,
		int ADNN_W_BLOCK0,
		int ADNN_W_FLTR1,
		int ADNN_W_FLTR0,
	_FLOAT *t_w, const _FLOAT *w, const _FLOAT *G, int *n_mul, int *n_add, int *n_flin)
	{
		_FLOAT * Gxg = (_FLOAT *)malloc(ADNN_W_BLOCK1 * ADNN_W_FLTR0 * sizeof(_FLOAT));
		for (int o = 0; o < ADNN_W_N_OCHNLS; ++o)
		{
			for (int c = 0; c < ADNN_W_N_ICHNLS; ++c)
			{

				float *GxgxG_T = &t_w[(o * ADNN_W_N_ICHNLS + c) * ADNN_W_BLOCK1 * ADNN_W_BLOCK0];
				const float *my_g = &w[(o * ADNN_W_N_ICHNLS + c) * ADNN_W_FLTR1 * ADNN_W_FLTR0];
				Weights_Block_Tansform_Win(ADNN_W_BLOCK1, ADNN_W_BLOCK0, ADNN_W_FLTR1, ADNN_W_FLTR0, GxgxG_T, my_g, G, Gxg, n_mul, n_add, n_flin);

			}
		}
		free(Gxg);
	}


	void Input_Block_Tansform_2x2_3x3Win(
		int ADNN_W_BLOCK1,
		int ADNN_W_BLOCK0,
		int ADNN_W_DSTRIDE,
		int ADNN_W_TSTRIDE,
		_FLOAT *C_TxdxC, _FLOAT *run_d, const _FLOAT *C, _FLOAT * C_Txd, int *n_mul, int *n_add, int *n_flin)
	{

		// data transform

		C_Txd[0] = run_d[0 * ADNN_W_DSTRIDE + 0] - run_d[0 * ADNN_W_DSTRIDE + 2];
		C_Txd[1] = run_d[1 * ADNN_W_DSTRIDE + 0] - run_d[1 * ADNN_W_DSTRIDE + 2];
		C_Txd[2] = run_d[2 * ADNN_W_DSTRIDE + 0] - run_d[2 * ADNN_W_DSTRIDE + 2];
		C_Txd[3] = run_d[3 * ADNN_W_DSTRIDE + 0] - run_d[3 * ADNN_W_DSTRIDE + 2];
		C_Txd[4] = run_d[0 * ADNN_W_DSTRIDE + 1] + run_d[0 * ADNN_W_DSTRIDE + 2];
		C_Txd[5] = run_d[1 * ADNN_W_DSTRIDE + 1] + run_d[1 * ADNN_W_DSTRIDE + 2];
		C_Txd[6] = run_d[2 * ADNN_W_DSTRIDE + 1] + run_d[2 * ADNN_W_DSTRIDE + 2];
		C_Txd[7] = run_d[3 * ADNN_W_DSTRIDE + 1] + run_d[3 * ADNN_W_DSTRIDE + 2];
		C_Txd[8] = -run_d[0 * ADNN_W_DSTRIDE + 1] + run_d[0 * ADNN_W_DSTRIDE + 2];
		C_Txd[9] = -run_d[1 * ADNN_W_DSTRIDE + 1] + run_d[1 * ADNN_W_DSTRIDE + 2];
		C_Txd[10] = -run_d[2 * ADNN_W_DSTRIDE + 1] + run_d[2 * ADNN_W_DSTRIDE + 2];
		C_Txd[11] = -run_d[3 * ADNN_W_DSTRIDE + 1] + run_d[3 * ADNN_W_DSTRIDE + 2];
		C_Txd[12] = run_d[0 * ADNN_W_DSTRIDE + 1] - run_d[0 * ADNN_W_DSTRIDE + 3];
		C_Txd[13] = run_d[1 * ADNN_W_DSTRIDE + 1] - run_d[1 * ADNN_W_DSTRIDE + 3];
		C_Txd[14] = run_d[2 * ADNN_W_DSTRIDE + 1] - run_d[2 * ADNN_W_DSTRIDE + 3];
		C_Txd[15] = run_d[3 * ADNN_W_DSTRIDE + 1] - run_d[3 * ADNN_W_DSTRIDE + 3];


		*n_add += 16;
		*n_flin += 16;

		C_TxdxC[0 * ADNN_W_TSTRIDE + 0] = C_Txd[0] - C_Txd[2];
		C_TxdxC[0 * ADNN_W_TSTRIDE + 1] = C_Txd[1] + C_Txd[2];
		C_TxdxC[0 * ADNN_W_TSTRIDE + 2] = -C_Txd[1] + C_Txd[2];
		C_TxdxC[0 * ADNN_W_TSTRIDE + 3] = C_Txd[1] - C_Txd[3];
		C_TxdxC[1 * ADNN_W_TSTRIDE + 0] = C_Txd[4] - C_Txd[6];
		C_TxdxC[1 * ADNN_W_TSTRIDE + 1] = C_Txd[5] + C_Txd[6];
		C_TxdxC[1 * ADNN_W_TSTRIDE + 2] = -C_Txd[5] + C_Txd[6];
		C_TxdxC[1 * ADNN_W_TSTRIDE + 3] = C_Txd[5] - C_Txd[7];
		C_TxdxC[2 * ADNN_W_TSTRIDE + 0] = C_Txd[8] - C_Txd[10];
		C_TxdxC[2 * ADNN_W_TSTRIDE + 1] = C_Txd[9] + C_Txd[10];
		C_TxdxC[2 * ADNN_W_TSTRIDE + 2] = -C_Txd[9] + C_Txd[10];
		C_TxdxC[2 * ADNN_W_TSTRIDE + 3] = C_Txd[9] - C_Txd[11];
		C_TxdxC[3 * ADNN_W_TSTRIDE + 0] = C_Txd[12] - C_Txd[14];
		C_TxdxC[3 * ADNN_W_TSTRIDE + 1] = C_Txd[13] + C_Txd[14];
		C_TxdxC[3 * ADNN_W_TSTRIDE + 2] = -C_Txd[13] + C_Txd[14];
		C_TxdxC[3 * ADNN_W_TSTRIDE + 3] = C_Txd[13] - C_Txd[15];

		*n_add += 16;
		*n_flin += 16;


	}



	void Input_Block_Tansform_4x4_3x3Win(
		int ADNN_W_BLOCK1,
		int ADNN_W_BLOCK0,
		int ADNN_W_DSTRIDE,
		int ADNN_W_TSTRIDE,
		_FLOAT *C_TxdxC, _FLOAT *run_d, const _FLOAT *C, _FLOAT * C_Txd, int *n_mul, int *n_add, int *n_flin)
	{

		// data transform
#if 1
		C_Txd[0] = (_FLOAT)(run_d[0 * ADNN_W_DSTRIDE + 0] * 4. + run_d[0 * ADNN_W_DSTRIDE + 2] * -5. + run_d[0 * ADNN_W_DSTRIDE + 4]);
		C_Txd[1] = (_FLOAT)(run_d[1 * ADNN_W_DSTRIDE + 0] * 4. + run_d[1 * ADNN_W_DSTRIDE + 2] * -5. + run_d[1 * ADNN_W_DSTRIDE + 4]);
		C_Txd[2] = (_FLOAT)(run_d[2 * ADNN_W_DSTRIDE + 0] * 4. + run_d[2 * ADNN_W_DSTRIDE + 2] * -5. + run_d[2 * ADNN_W_DSTRIDE + 4]);
		C_Txd[3] = (_FLOAT)(run_d[3 * ADNN_W_DSTRIDE + 0] * 4. + run_d[3 * ADNN_W_DSTRIDE + 2] * -5. + run_d[3 * ADNN_W_DSTRIDE + 4]);
		C_Txd[4] = (_FLOAT)(run_d[4 * ADNN_W_DSTRIDE + 0] * 4. + run_d[4 * ADNN_W_DSTRIDE + 2] * -5. + run_d[4 * ADNN_W_DSTRIDE + 4]);
		C_Txd[5] = (_FLOAT)(run_d[5 * ADNN_W_DSTRIDE + 0] * 4. + run_d[5 * ADNN_W_DSTRIDE + 2] * -5. + run_d[5 * ADNN_W_DSTRIDE + 4]);
		C_Txd[6] = (_FLOAT)(run_d[0 * ADNN_W_DSTRIDE + 1] * -4. + run_d[0 * ADNN_W_DSTRIDE + 2] * -4. + run_d[0 * ADNN_W_DSTRIDE + 3] + run_d[0 * ADNN_W_DSTRIDE + 4]);
		C_Txd[7] = (_FLOAT)(run_d[1 * ADNN_W_DSTRIDE + 1] * -4. + run_d[1 * ADNN_W_DSTRIDE + 2] * -4. + run_d[1 * ADNN_W_DSTRIDE + 3] + run_d[1 * ADNN_W_DSTRIDE + 4]);
		C_Txd[8] = (_FLOAT)(run_d[2 * ADNN_W_DSTRIDE + 1] * -4. + run_d[2 * ADNN_W_DSTRIDE + 2] * -4. + run_d[2 * ADNN_W_DSTRIDE + 3] + run_d[2 * ADNN_W_DSTRIDE + 4]);
		C_Txd[9] = (_FLOAT)(run_d[3 * ADNN_W_DSTRIDE + 1] * -4. + run_d[3 * ADNN_W_DSTRIDE + 2] * -4. + run_d[3 * ADNN_W_DSTRIDE + 3] + run_d[3 * ADNN_W_DSTRIDE + 4]);
		C_Txd[10] = (_FLOAT)(run_d[4 * ADNN_W_DSTRIDE + 1] * -4. + run_d[4 * ADNN_W_DSTRIDE + 2] * -4. + run_d[4 * ADNN_W_DSTRIDE + 3] + run_d[4 * ADNN_W_DSTRIDE + 4]);
		C_Txd[11] = (_FLOAT)(run_d[5 * ADNN_W_DSTRIDE + 1] * -4. + run_d[5 * ADNN_W_DSTRIDE + 2] * -4. + run_d[5 * ADNN_W_DSTRIDE + 3] + run_d[5 * ADNN_W_DSTRIDE + 4]);
		C_Txd[12] = (_FLOAT)(run_d[0 * ADNN_W_DSTRIDE + 1] * 4. + run_d[0 * ADNN_W_DSTRIDE + 2] * -4. - run_d[0 * ADNN_W_DSTRIDE + 3] + run_d[0 * ADNN_W_DSTRIDE + 4]);
		C_Txd[13] = (_FLOAT)(run_d[1 * ADNN_W_DSTRIDE + 1] * 4. + run_d[1 * ADNN_W_DSTRIDE + 2] * -4. - run_d[1 * ADNN_W_DSTRIDE + 3] + run_d[1 * ADNN_W_DSTRIDE + 4]);
		C_Txd[14] = (_FLOAT)(run_d[2 * ADNN_W_DSTRIDE + 1] * 4. + run_d[2 * ADNN_W_DSTRIDE + 2] * -4. - run_d[2 * ADNN_W_DSTRIDE + 3] + run_d[2 * ADNN_W_DSTRIDE + 4]);
		C_Txd[15] = (_FLOAT)(run_d[3 * ADNN_W_DSTRIDE + 1] * 4. + run_d[3 * ADNN_W_DSTRIDE + 2] * -4. - run_d[3 * ADNN_W_DSTRIDE + 3] + run_d[3 * ADNN_W_DSTRIDE + 4]);
		C_Txd[16] = (_FLOAT)(run_d[4 * ADNN_W_DSTRIDE + 1] * 4. + run_d[4 * ADNN_W_DSTRIDE + 2] * -4. - run_d[4 * ADNN_W_DSTRIDE + 3] + run_d[4 * ADNN_W_DSTRIDE + 4]);
		C_Txd[17] = (_FLOAT)(run_d[5 * ADNN_W_DSTRIDE + 1] * 4. + run_d[5 * ADNN_W_DSTRIDE + 2] * -4. - run_d[5 * ADNN_W_DSTRIDE + 3] + run_d[5 * ADNN_W_DSTRIDE + 4]);
		C_Txd[18] = (_FLOAT)(run_d[0 * ADNN_W_DSTRIDE + 1] * -2. - run_d[0 * ADNN_W_DSTRIDE + 2] + run_d[0 * ADNN_W_DSTRIDE + 3] * 2. + run_d[0 * ADNN_W_DSTRIDE + 4]);
		C_Txd[19] = (_FLOAT)(run_d[1 * ADNN_W_DSTRIDE + 1] * -2. - run_d[1 * ADNN_W_DSTRIDE + 2] + run_d[1 * ADNN_W_DSTRIDE + 3] * 2. + run_d[1 * ADNN_W_DSTRIDE + 4]);
		C_Txd[20] = (_FLOAT)(run_d[2 * ADNN_W_DSTRIDE + 1] * -2. - run_d[2 * ADNN_W_DSTRIDE + 2] + run_d[2 * ADNN_W_DSTRIDE + 3] * 2. + run_d[2 * ADNN_W_DSTRIDE + 4]);
		C_Txd[21] = (_FLOAT)(run_d[3 * ADNN_W_DSTRIDE + 1] * -2. - run_d[3 * ADNN_W_DSTRIDE + 2] + run_d[3 * ADNN_W_DSTRIDE + 3] * 2. + run_d[3 * ADNN_W_DSTRIDE + 4]);
		C_Txd[22] = (_FLOAT)(run_d[4 * ADNN_W_DSTRIDE + 1] * -2. - run_d[4 * ADNN_W_DSTRIDE + 2] + run_d[4 * ADNN_W_DSTRIDE + 3] * 2. + run_d[4 * ADNN_W_DSTRIDE + 4]);
		C_Txd[23] = (_FLOAT)(run_d[5 * ADNN_W_DSTRIDE + 1] * -2. - run_d[5 * ADNN_W_DSTRIDE + 2] + run_d[5 * ADNN_W_DSTRIDE + 3] * 2. + run_d[5 * ADNN_W_DSTRIDE + 4]);
		C_Txd[24] = (_FLOAT)(run_d[0 * ADNN_W_DSTRIDE + 1] * 2. - run_d[0 * ADNN_W_DSTRIDE + 2] + run_d[0 * ADNN_W_DSTRIDE + 3] * -2. + run_d[0 * ADNN_W_DSTRIDE + 4]);
		C_Txd[25] = (_FLOAT)(run_d[1 * ADNN_W_DSTRIDE + 1] * 2. - run_d[1 * ADNN_W_DSTRIDE + 2] + run_d[1 * ADNN_W_DSTRIDE + 3] * -2. + run_d[1 * ADNN_W_DSTRIDE + 4]);
		C_Txd[26] = (_FLOAT)(run_d[2 * ADNN_W_DSTRIDE + 1] * 2. - run_d[2 * ADNN_W_DSTRIDE + 2] + run_d[2 * ADNN_W_DSTRIDE + 3] * -2. + run_d[2 * ADNN_W_DSTRIDE + 4]);
		C_Txd[27] = (_FLOAT)(run_d[3 * ADNN_W_DSTRIDE + 1] * 2. - run_d[3 * ADNN_W_DSTRIDE + 2] + run_d[3 * ADNN_W_DSTRIDE + 3] * -2. + run_d[3 * ADNN_W_DSTRIDE + 4]);
		C_Txd[28] = (_FLOAT)(run_d[4 * ADNN_W_DSTRIDE + 1] * 2. - run_d[4 * ADNN_W_DSTRIDE + 2] + run_d[4 * ADNN_W_DSTRIDE + 3] * -2. + run_d[4 * ADNN_W_DSTRIDE + 4]);
		C_Txd[29] = (_FLOAT)(run_d[5 * ADNN_W_DSTRIDE + 1] * 2. - run_d[5 * ADNN_W_DSTRIDE + 2] + run_d[5 * ADNN_W_DSTRIDE + 3] * -2. + run_d[5 * ADNN_W_DSTRIDE + 4]);
		C_Txd[30] = (_FLOAT)(run_d[0 * ADNN_W_DSTRIDE + 1] * 4. + run_d[0 * ADNN_W_DSTRIDE + 3] * -5. + run_d[0 * ADNN_W_DSTRIDE + 5]);
		C_Txd[31] = (_FLOAT)(run_d[1 * ADNN_W_DSTRIDE + 1] * 4. + run_d[1 * ADNN_W_DSTRIDE + 3] * -5. + run_d[1 * ADNN_W_DSTRIDE + 5]);
		C_Txd[32] = (_FLOAT)(run_d[2 * ADNN_W_DSTRIDE + 1] * 4. + run_d[2 * ADNN_W_DSTRIDE + 3] * -5. + run_d[2 * ADNN_W_DSTRIDE + 5]);
		C_Txd[33] = (_FLOAT)(run_d[3 * ADNN_W_DSTRIDE + 1] * 4. + run_d[3 * ADNN_W_DSTRIDE + 3] * -5. + run_d[3 * ADNN_W_DSTRIDE + 5]);
		C_Txd[34] = (_FLOAT)(run_d[4 * ADNN_W_DSTRIDE + 1] * 4. + run_d[4 * ADNN_W_DSTRIDE + 3] * -5. + run_d[4 * ADNN_W_DSTRIDE + 5]);
		C_Txd[35] = (_FLOAT)(run_d[5 * ADNN_W_DSTRIDE + 1] * 4. + run_d[5 * ADNN_W_DSTRIDE + 3] * -5. + run_d[5 * ADNN_W_DSTRIDE + 5]);

#else
		// C_T x d
		for (int j = 0; j < ADNN_W_BLOCK1; ++j)
		{
			for (int i = 0; i < ADNN_W_BLOCK0; ++i)
			{
				C_Txd[j * ADNN_W_BLOCK0 + i] = 0;
				for (int k = 0; k < ADNN_W_BLOCK0; ++k)
				{
					// C transposed + run_d transposed
					C_Txd[j * ADNN_W_BLOCK0 + i] += C[k * ADNN_W_BLOCK0 + j] * run_d[i * ADNN_W_DSTRIDE + k];
//					printf("C_Td: oi=%d i=%d k=%d cv=%f\n", j * ADNN_W_BLOCK0 + i, i, k, C[k * ADNN_W_BLOCK0 + j]);
				}
			}
		}
#endif
		*n_mul += 2 * 36;
		*n_add += 2 * 12 + 3*24;
		*n_flin = *n_add;

#if 1
		C_TxdxC[0 * ADNN_W_TSTRIDE + 0] = (_FLOAT)(C_Txd[0] * 4. + C_Txd[2] * -5. + C_Txd[4]);
		C_TxdxC[0 * ADNN_W_TSTRIDE + 1] = (_FLOAT)(C_Txd[1] * -4. + C_Txd[2] * -4. + C_Txd[3] + C_Txd[4]);
		C_TxdxC[0 * ADNN_W_TSTRIDE + 2] = (_FLOAT)(C_Txd[1] * 4. + C_Txd[2] * -4. - C_Txd[3] + C_Txd[4]);
		C_TxdxC[0 * ADNN_W_TSTRIDE + 3] = (_FLOAT)(C_Txd[1] * -2. - C_Txd[2] + C_Txd[3] * 2. + C_Txd[4]);
		C_TxdxC[0 * ADNN_W_TSTRIDE + 4] = (_FLOAT)(C_Txd[1] * 2. - C_Txd[2] + C_Txd[3] * -2. + C_Txd[4]);
		C_TxdxC[0 * ADNN_W_TSTRIDE + 5] = (_FLOAT)(C_Txd[1] * 4. + C_Txd[3] * -5. + C_Txd[5]);
		C_TxdxC[1 * ADNN_W_TSTRIDE + 0] = (_FLOAT)(C_Txd[6] * 4. + C_Txd[8] * -5. + C_Txd[10]);
		C_TxdxC[1 * ADNN_W_TSTRIDE + 1] = (_FLOAT)(C_Txd[7] * -4. + C_Txd[8] * -4. + C_Txd[9] + C_Txd[10]);
		C_TxdxC[1 * ADNN_W_TSTRIDE + 2] = (_FLOAT)(C_Txd[7] * 4. + C_Txd[8] * -4. - C_Txd[9] + C_Txd[10]);
		C_TxdxC[1 * ADNN_W_TSTRIDE + 3] = (_FLOAT)(C_Txd[7] * -2. - C_Txd[8] + C_Txd[9] * 2. + C_Txd[10]);
		C_TxdxC[1 * ADNN_W_TSTRIDE + 4] = (_FLOAT)(C_Txd[7] * 2. - C_Txd[8] + C_Txd[9] * -2. + C_Txd[10]);
		C_TxdxC[1 * ADNN_W_TSTRIDE + 5] = (_FLOAT)(C_Txd[7] * 4. + C_Txd[9] * -5. + C_Txd[11]);
		C_TxdxC[2 * ADNN_W_TSTRIDE + 0] = (_FLOAT)(C_Txd[12] * 4. + C_Txd[14] * -5. + C_Txd[16]);
		C_TxdxC[2 * ADNN_W_TSTRIDE + 1] = (_FLOAT)(C_Txd[13] * -4. + C_Txd[14] * -4. + C_Txd[15] + C_Txd[16]);
		C_TxdxC[2 * ADNN_W_TSTRIDE + 2] = (_FLOAT)(C_Txd[13] * 4. + C_Txd[14] * -4. - C_Txd[15] + C_Txd[16]);
		C_TxdxC[2 * ADNN_W_TSTRIDE + 3] = (_FLOAT)(C_Txd[13] * -2. - C_Txd[14] + C_Txd[15] * 2. + C_Txd[16]);
		C_TxdxC[2 * ADNN_W_TSTRIDE + 4] = (_FLOAT)(C_Txd[13] * 2. - C_Txd[14] + C_Txd[15] * -2. + C_Txd[16]);
		C_TxdxC[2 * ADNN_W_TSTRIDE + 5] = (_FLOAT)(C_Txd[13] * 4. + C_Txd[15] * -5. + C_Txd[17]);
		C_TxdxC[3 * ADNN_W_TSTRIDE + 0] = (_FLOAT)(C_Txd[18] * 4. + C_Txd[20] * -5. + C_Txd[22]);
		C_TxdxC[3 * ADNN_W_TSTRIDE + 1] = (_FLOAT)(C_Txd[19] * -4. + C_Txd[20] * -4. + C_Txd[21] + C_Txd[22]);
		C_TxdxC[3 * ADNN_W_TSTRIDE + 2] = (_FLOAT)(C_Txd[19] * 4. + C_Txd[20] * -4. - C_Txd[21] + C_Txd[22]);
		C_TxdxC[3 * ADNN_W_TSTRIDE + 3] = (_FLOAT)(C_Txd[19] * -2. - C_Txd[20] + C_Txd[21] * 2. + C_Txd[22]);
		C_TxdxC[3 * ADNN_W_TSTRIDE + 4] = (_FLOAT)(C_Txd[19] * 2. - C_Txd[20] + C_Txd[21] * -2. + C_Txd[22]);
		C_TxdxC[3 * ADNN_W_TSTRIDE + 5] = (_FLOAT)(C_Txd[19] * 4. + C_Txd[21] * -5. + C_Txd[23]);
		C_TxdxC[4 * ADNN_W_TSTRIDE + 0] = (_FLOAT)(C_Txd[24] * 4. + C_Txd[26] * -5. + C_Txd[28]);
		C_TxdxC[4 * ADNN_W_TSTRIDE + 1] = (_FLOAT)(C_Txd[25] * -4. + C_Txd[26] * -4. + C_Txd[27] + C_Txd[28]);
		C_TxdxC[4 * ADNN_W_TSTRIDE + 2] = (_FLOAT)(C_Txd[25] * 4. + C_Txd[26] * -4. - C_Txd[27] + C_Txd[28]);
		C_TxdxC[4 * ADNN_W_TSTRIDE + 3] = (_FLOAT)(C_Txd[25] * -2. - C_Txd[26] + C_Txd[27] * 2. + C_Txd[28]);
		C_TxdxC[4 * ADNN_W_TSTRIDE + 4] = (_FLOAT)(C_Txd[25] * 2. - C_Txd[26] + C_Txd[27] * -2. + C_Txd[28]);
		C_TxdxC[4 * ADNN_W_TSTRIDE + 5] = (_FLOAT)(C_Txd[25] * 4. + C_Txd[27] * -5. + C_Txd[29]);
		C_TxdxC[5 * ADNN_W_TSTRIDE + 0] = (_FLOAT)(C_Txd[30] * 4. + C_Txd[32] * -5. + C_Txd[34]);
		C_TxdxC[5 * ADNN_W_TSTRIDE + 1] = (_FLOAT)(C_Txd[31] * -4. + C_Txd[32] * -4. + C_Txd[33] + C_Txd[34]);
		C_TxdxC[5 * ADNN_W_TSTRIDE + 2] = (_FLOAT)(C_Txd[31] * 4. + C_Txd[32] * -4. - C_Txd[33] + C_Txd[34]);
		C_TxdxC[5 * ADNN_W_TSTRIDE + 3] = (_FLOAT)(C_Txd[31] * -2. - C_Txd[32] + C_Txd[33] * 2. + C_Txd[34]);
		C_TxdxC[5 * ADNN_W_TSTRIDE + 4] = (_FLOAT)(C_Txd[31] * 2. - C_Txd[32] + C_Txd[33] * -2. + C_Txd[34]);
		C_TxdxC[5 * ADNN_W_TSTRIDE + 5] = (_FLOAT)(C_Txd[31] * 4. + C_Txd[33] * -5. + C_Txd[35]);


#else

		// C_Txd x C
		for (int j = 0; j < ADNN_W_BLOCK1; ++j)
		{
			for (int i = 0; i < ADNN_W_BLOCK0; ++i)
			{
				C_TxdxC[j * ADNN_W_TSTRIDE + i] = 0;
				for (int k = 0; k < ADNN_W_BLOCK0; ++k)
				{
					C_TxdxC[j * ADNN_W_TSTRIDE + i] += C_Txd[j * ADNN_W_BLOCK0 + k] * C[k * ADNN_W_BLOCK0 + i];
//					printf("dC: j=%d i=%d ii=%d cv=%f\n", j, i, j * ADNN_W_BLOCK0 + k, C[k * ADNN_W_BLOCK0 + i]);
				}
			}
		}
#endif

		*n_mul += 2 * 36;
		*n_add += 2 * 2*6 + 4*3 * 6;
		*n_flin = *n_add;

	}



	void Input_Block_Tansform_Win(
		int ADNN_W_BLOCK1,
		int ADNN_W_BLOCK0,
		int ADNN_W_DSTRIDE,
		int ADNN_W_TSTRIDE,
		_FLOAT *C_TxdxC, _FLOAT *run_d, const _FLOAT *C, _FLOAT * C_Txd, int *n_mul, int *n_add, int *n_flin)
	{

		// data transform
#if ADNN_ELEMENT_WISE_TRANSOFRMS

#if ADNN_ALG_WIN2x2_3x3
		Input_Block_Tansform_2x2_3x3Win
#else
		Input_Block_Tansform_4x4_3x3Win
#endif
			(
			ADNN_W_BLOCK1,
			ADNN_W_BLOCK0,
			ADNN_W_DSTRIDE,
			ADNN_W_TSTRIDE,
			C_TxdxC, run_d, C, C_Txd, n_mul, n_add, n_flin);

#else

		// C_T x d
		for (int j = 0; j < ADNN_W_BLOCK1; ++j)
		{
			for (int i = 0; i < ADNN_W_BLOCK0; ++i)
			{
				C_Txd[j * ADNN_W_BLOCK0 + i] = 0;
				for (int k = 0; k < ADNN_W_BLOCK0; ++k)
				{
					// C transposed + run_d transposed
					C_Txd[j * ADNN_W_BLOCK0 + i] += C[k * ADNN_W_BLOCK0 + j] * run_d[i * ADNN_W_DSTRIDE + k];
//					printf("C_Td: C_TdI=%d i=%d k=%d cv=%f\n", j * ADNN_W_BLOCK0 + i, i, k, C[k * ADNN_W_BLOCK0 + j]);
				}
//				printf("C_Td: C_TdI=%d cv=%f\n", j * ADNN_W_BLOCK0 + i, C_Txd[j * ADNN_W_BLOCK0 + i]);
			}
		}

//		printf("\n\n");
		// C_Txd x C
		for (int j = 0; j < ADNN_W_BLOCK1; ++j)
		{
			for (int i = 0; i < ADNN_W_BLOCK0; ++i)
			{
				C_TxdxC[j * ADNN_W_TSTRIDE + i] = 0;
				for (int k = 0; k < ADNN_W_BLOCK0; ++k)
				{
					C_TxdxC[j * ADNN_W_TSTRIDE + i] += C_Txd[j * ADNN_W_BLOCK0 + k] * C[k * ADNN_W_BLOCK0 + i];
//					printf("dC: j=%d i=%d C_TdI=%d ctv=%f cv=%f\n", j, i, j * ADNN_W_BLOCK0 + k, C_Txd[j * ADNN_W_BLOCK0 + k], C[k * ADNN_W_BLOCK0 + i]);
				}
//				printf("dC: dCI=%d cv=%f\n", j * ADNN_W_TSTRIDE + i, C_TxdxC[j * ADNN_W_TSTRIDE + i]);
			}
		}
#endif
	}


	void Input_Tansform_Win(
		int ADNN_W_BATCH_SZ,
		int ADNN_W_N_ICHNLS,
		int ADNN_W_HEIGH, int ADNN_W_WIDTH,
		int ADNN_W_ISTRIDE, int ADNN_W_ICHNL_STRIDE, int ADNN_W_IBATCH_STRIDE,
		int ADNN_W_BSTRIDE, int ADNN_W_BCHNL_STRIDE, int ADNN_W_BBATCH_STRIDE,
		int ADNN_W_BLOCK1,
		int ADNN_W_BLOCK0,
		int ADNN_W_TILE1,
		int ADNN_W_TILE0,
		int ADNN_W_PAD1,
		int ADNN_W_PAD0,
	_FLOAT *blocked_d, const _FLOAT *d, const _FLOAT *C, int *n_mul, int *n_add, int *n_flin)
	{
		_FLOAT * C_Txd = (_FLOAT *)malloc(ADNN_W_BLOCK1 * ADNN_W_BLOCK0 * sizeof(_FLOAT));
		for (int b = 0; b < ADNN_W_BATCH_SZ; ++b)
		{
			for (int c = 0; c < ADNN_W_N_ICHNLS; ++c)
			{
				for (int j = 0; j < (ADNN_W_HEIGH / ADNN_W_TILE1); ++j)
				{
					for (int i = 0; i < (ADNN_W_WIDTH / ADNN_W_TILE0); ++i)
					{
						float * block = &blocked_d[b*ADNN_W_BBATCH_STRIDE + c*ADNN_W_BCHNL_STRIDE + j*ADNN_W_BLOCK1 * ADNN_W_BSTRIDE + i* ADNN_W_BLOCK0];
						for (int k = 0; k < ADNN_W_BLOCK1; ++k)
						{
							int act_j = j * ADNN_W_TILE1 + k - ADNN_W_PAD1;
							bool invY = (act_j < 0 || act_j >= ADNN_W_HEIGH);
							act_j = (invY) ? 0 : act_j;

							for (int l = 0; l < ADNN_W_BLOCK0; ++l)
							{
								int act_i = i * ADNN_W_TILE0 + l - ADNN_W_PAD0;
								bool invX = (act_i < 0 || act_i >= ADNN_W_WIDTH);
								act_i = (invX) ? 0 : act_i;
								block[k*ADNN_W_BSTRIDE + l] = (invY || invX) ? 0 : d[b*ADNN_W_IBATCH_STRIDE + c* ADNN_W_ICHNL_STRIDE + act_j * ADNN_W_ISTRIDE + act_i];
							}
						}
#if 0
						if (b == 0 && i==0 && j == 0 && c == 1)
						{
							printf("c:it: %f %f %f %f\n",
								block[0],
								block[1],
								block[2],
								block[3]
								);
						}
#endif


						Input_Block_Tansform_Win(ADNN_W_BLOCK1, ADNN_W_BLOCK0, ADNN_W_BSTRIDE, ADNN_W_BSTRIDE, block, block, C, C_Txd, n_mul, n_add, n_flin);
					}
				}
			}
		}
		free(C_Txd);
	}


	void ElemWise_Multiply_Win(
		int ADNN_W_BATCH_SZ,
		int ADNN_W_N_ICHNLS, int ADNN_W_N_OCHNLS,
		int ADNN_W_HEIGH, int ADNN_W_WIDTH,
		int ADNN_W_BSTRIDE, int ADNN_W_BCHNL_STRIDE,
		int ADNN_W_IBBATCH_STRIDE, int ADNN_W_OBBATCH_STRIDE,
		int ADNN_W_BLOCK1,
		int ADNN_W_BLOCK0,
		int ADNN_W_TILE1,
		int ADNN_W_TILE0,
	_FLOAT * EwM, const _FLOAT *d_transformed, const _FLOAT *w_t, int *n_mul, int *n_add, int *n_flin)
	{

		// Winodrad algorithms
		int y_loop = ADNN_W_HEIGH / ADNN_W_TILE1;
		int x_loop = ADNN_W_WIDTH / ADNN_W_TILE0;
		for (int b = 0; b < ADNN_W_BATCH_SZ; ++b)
		{
			for (int o = 0; o < ADNN_W_N_OCHNLS; ++o)
			{
				const _FLOAT *GxgxG_T = &w_t[o * ADNN_W_BLOCK1 * ADNN_W_BLOCK0 * ADNN_W_N_ICHNLS];
				for (int tile_y = 0; tile_y < y_loop; ++tile_y)
				{
					for (int tile_x = 0; tile_x < x_loop; ++tile_x)
					{

						_FLOAT * Elem_wise_Mult = &EwM[b * ADNN_W_OBBATCH_STRIDE + o * ADNN_W_BCHNL_STRIDE + tile_y * ADNN_W_BLOCK1 * ADNN_W_BSTRIDE + tile_x * ADNN_W_BLOCK0];

						// element-wise multiply
						for (int j = 0; j < ADNN_W_BLOCK1; ++j)
						{
							for (int i = 0; i < ADNN_W_BLOCK0; ++i)
							{
								Elem_wise_Mult[j* ADNN_W_BSTRIDE + i] = 0;
								for (int c = 0; c < ADNN_W_N_ICHNLS; ++c)
								{
									const _FLOAT * C_TxdxC = &d_transformed[b * ADNN_W_IBBATCH_STRIDE + c * ADNN_W_BCHNL_STRIDE + tile_y * ADNN_W_BLOCK1 * ADNN_W_BSTRIDE + tile_x * ADNN_W_BLOCK0];

									Elem_wise_Mult[j*ADNN_W_BSTRIDE + i] += GxgxG_T[c*ADNN_W_BLOCK1 * ADNN_W_BLOCK0 + j* ADNN_W_BLOCK0 + i] * C_TxdxC[j*ADNN_W_BSTRIDE + i];
#if 0
									if (b==0 && o == 0 && tile_y == 0 && tile_x == 0 && j == 0 && i == 0)
									{
										printf("c:ewm: b=%d o=%d c=%d by=%d bx=%d j=%d j=%d s=%f d=%f w=%f\n",
											b,
											o,
											c,
											tile_y,
											tile_x,
											j,
											i,
											Elem_wise_Mult[j*ADNN_W_BSTRIDE + i],
											C_TxdxC[j*ADNN_W_BSTRIDE + i],
											GxgxG_T[c*ADNN_W_BLOCK1 * ADNN_W_BLOCK0 + j* ADNN_W_BLOCK0 + i]
											);
									}

#endif


								}
							}
						}
					}
				}
			}
		}
		*n_mul += ADNN_W_BATCH_SZ * ADNN_W_N_OCHNLS * y_loop * ADNN_W_BLOCK1 * x_loop * ADNN_W_BLOCK0 * ADNN_W_N_ICHNLS;
		*n_add += ADNN_W_BATCH_SZ * ADNN_W_N_OCHNLS * y_loop * ADNN_W_BLOCK1 * x_loop * ADNN_W_BLOCK0 * ADNN_W_N_ICHNLS;
		*n_flin += ADNN_W_BATCH_SZ * ADNN_W_N_OCHNLS * y_loop * ADNN_W_BLOCK1 * x_loop * ADNN_W_BLOCK0 * ADNN_W_N_ICHNLS;
	}


	// inverse transform

	void Inverse_BlockTransform_2x2_3x3Win(
		int ADNN_W_DSTRIDE,
		int ADNN_W_FSTRIDE,
		int ADNN_W_BLOCK1,
		int ADNN_W_BLOCK0,
		int ADNN_W_TILE1,
		int ADNN_W_TILE0,
		_FLOAT *run_f, const _FLOAT * Elem_wise_Mult, const _FLOAT *A, _FLOAT * A_TxEWM, int *n_mul, int *n_add, int *n_flin)
	{

		A_TxEWM[0] = Elem_wise_Mult[0 * ADNN_W_DSTRIDE + 0] + Elem_wise_Mult[1 * ADNN_W_DSTRIDE + 0] + Elem_wise_Mult[2 * ADNN_W_DSTRIDE + 0];
		A_TxEWM[1] = Elem_wise_Mult[0 * ADNN_W_DSTRIDE + 1] + Elem_wise_Mult[1 * ADNN_W_DSTRIDE + 1] + Elem_wise_Mult[2 * ADNN_W_DSTRIDE + 1];
		A_TxEWM[2] = Elem_wise_Mult[0 * ADNN_W_DSTRIDE + 2] + Elem_wise_Mult[1 * ADNN_W_DSTRIDE + 2] + Elem_wise_Mult[2 * ADNN_W_DSTRIDE + 2];
		A_TxEWM[3] = Elem_wise_Mult[0 * ADNN_W_DSTRIDE + 3] + Elem_wise_Mult[1 * ADNN_W_DSTRIDE + 3] + Elem_wise_Mult[2 * ADNN_W_DSTRIDE + 3];
		A_TxEWM[4] = Elem_wise_Mult[1 * ADNN_W_DSTRIDE + 0] - Elem_wise_Mult[2 * ADNN_W_DSTRIDE + 0] - Elem_wise_Mult[3 * ADNN_W_DSTRIDE + 0];
		A_TxEWM[5] = Elem_wise_Mult[1 * ADNN_W_DSTRIDE + 1] - Elem_wise_Mult[2 * ADNN_W_DSTRIDE + 1] - Elem_wise_Mult[3 * ADNN_W_DSTRIDE + 1];
		A_TxEWM[6] = Elem_wise_Mult[1 * ADNN_W_DSTRIDE + 2] - Elem_wise_Mult[2 * ADNN_W_DSTRIDE + 2] - Elem_wise_Mult[3 * ADNN_W_DSTRIDE + 2];
		A_TxEWM[7] = Elem_wise_Mult[1 * ADNN_W_DSTRIDE + 3] - Elem_wise_Mult[2 * ADNN_W_DSTRIDE + 3] - Elem_wise_Mult[3 * ADNN_W_DSTRIDE + 3];

		*n_add += 16;
		*n_flin += 16;

		// x A
		// transpose output
		run_f[0 * ADNN_W_FSTRIDE + 0] = A_TxEWM[0] + A_TxEWM[1] + A_TxEWM[2];
		run_f[1 * ADNN_W_FSTRIDE + 0] = A_TxEWM[1] - A_TxEWM[2] - A_TxEWM[3];
		run_f[0 * ADNN_W_FSTRIDE + 1] = A_TxEWM[4] + A_TxEWM[5] + A_TxEWM[6];
		run_f[1 * ADNN_W_FSTRIDE + 1] = A_TxEWM[5] - A_TxEWM[6] - A_TxEWM[7];

		*n_add += 8;
		*n_flin += 8;
	}


	void Inverse_BlockTransform_4x4_3x3Win(
		int ADNN_W_BSTRIDE,
		int ADNN_W_OSTRIDE,
		int ADNN_W_BLOCK1,
		int ADNN_W_BLOCK0,
		int ADNN_W_TILE1,
		int ADNN_W_TILE0,
		_FLOAT *run_f, const _FLOAT * Elem_wise_Mult, const _FLOAT *A, _FLOAT * A_TxEWM, int *n_mul, int *n_add, int *n_flin)
	{


#if 1
		A_TxEWM[0] = Elem_wise_Mult[0 * ADNN_W_BSTRIDE + 0] + Elem_wise_Mult[1 * ADNN_W_BSTRIDE + 0] + Elem_wise_Mult[2 * ADNN_W_BSTRIDE + 0] + Elem_wise_Mult[3 * ADNN_W_BSTRIDE + 0] + Elem_wise_Mult[4 * ADNN_W_BSTRIDE + 0];
		A_TxEWM[1] = Elem_wise_Mult[0 * ADNN_W_BSTRIDE + 1] + Elem_wise_Mult[1 * ADNN_W_BSTRIDE + 1] + Elem_wise_Mult[2 * ADNN_W_BSTRIDE + 1] + Elem_wise_Mult[3 * ADNN_W_BSTRIDE + 1] + Elem_wise_Mult[4 * ADNN_W_BSTRIDE + 1];
		A_TxEWM[2] = Elem_wise_Mult[0 * ADNN_W_BSTRIDE + 2] + Elem_wise_Mult[1 * ADNN_W_BSTRIDE + 2] + Elem_wise_Mult[2 * ADNN_W_BSTRIDE + 2] + Elem_wise_Mult[3 * ADNN_W_BSTRIDE + 2] + Elem_wise_Mult[4 * ADNN_W_BSTRIDE + 2];
		A_TxEWM[3] = Elem_wise_Mult[0 * ADNN_W_BSTRIDE + 3] + Elem_wise_Mult[1 * ADNN_W_BSTRIDE + 3] + Elem_wise_Mult[2 * ADNN_W_BSTRIDE + 3] + Elem_wise_Mult[3 * ADNN_W_BSTRIDE + 3] + Elem_wise_Mult[4 * ADNN_W_BSTRIDE + 3];
		A_TxEWM[4] = Elem_wise_Mult[0 * ADNN_W_BSTRIDE + 4] + Elem_wise_Mult[1 * ADNN_W_BSTRIDE + 4] + Elem_wise_Mult[2 * ADNN_W_BSTRIDE + 4] + Elem_wise_Mult[3 * ADNN_W_BSTRIDE + 4] + Elem_wise_Mult[4 * ADNN_W_BSTRIDE + 4];
		A_TxEWM[5] = Elem_wise_Mult[0 * ADNN_W_BSTRIDE + 5] + Elem_wise_Mult[1 * ADNN_W_BSTRIDE + 5] + Elem_wise_Mult[2 * ADNN_W_BSTRIDE + 5] + Elem_wise_Mult[3 * ADNN_W_BSTRIDE + 5] + Elem_wise_Mult[4 * ADNN_W_BSTRIDE + 5];
		A_TxEWM[6] = Elem_wise_Mult[1 * ADNN_W_BSTRIDE + 0] - Elem_wise_Mult[2 * ADNN_W_BSTRIDE + 0] + Elem_wise_Mult[3 * ADNN_W_BSTRIDE + 0] * (_FLOAT)2. + Elem_wise_Mult[4 * ADNN_W_BSTRIDE + 0] * (_FLOAT)(-2.);
		A_TxEWM[7] = Elem_wise_Mult[1 * ADNN_W_BSTRIDE + 1] - Elem_wise_Mult[2 * ADNN_W_BSTRIDE + 1] + Elem_wise_Mult[3 * ADNN_W_BSTRIDE + 1] * (_FLOAT)2. + Elem_wise_Mult[4 * ADNN_W_BSTRIDE + 1] * (_FLOAT)(-2.);
		A_TxEWM[8] = Elem_wise_Mult[1 * ADNN_W_BSTRIDE + 2] - Elem_wise_Mult[2 * ADNN_W_BSTRIDE + 2] + Elem_wise_Mult[3 * ADNN_W_BSTRIDE + 2] * (_FLOAT)2. + Elem_wise_Mult[4 * ADNN_W_BSTRIDE + 2] * (_FLOAT)(-2.);
		A_TxEWM[9] = Elem_wise_Mult[1 * ADNN_W_BSTRIDE + 3] - Elem_wise_Mult[2 * ADNN_W_BSTRIDE + 3] + Elem_wise_Mult[3 * ADNN_W_BSTRIDE + 3] * (_FLOAT)2. + Elem_wise_Mult[4 * ADNN_W_BSTRIDE + 3] * (_FLOAT)(-2.);
		A_TxEWM[10] = Elem_wise_Mult[1 * ADNN_W_BSTRIDE + 4] - Elem_wise_Mult[2 * ADNN_W_BSTRIDE + 4] + Elem_wise_Mult[3 * ADNN_W_BSTRIDE + 4] * (_FLOAT)2. + Elem_wise_Mult[4 * ADNN_W_BSTRIDE + 4] * (_FLOAT)(-2.);
		A_TxEWM[11] = Elem_wise_Mult[1 * ADNN_W_BSTRIDE + 5] - Elem_wise_Mult[2 * ADNN_W_BSTRIDE + 5] + Elem_wise_Mult[3 * ADNN_W_BSTRIDE + 5] * (_FLOAT)2. + Elem_wise_Mult[4 * ADNN_W_BSTRIDE + 5] * (_FLOAT)(-2.);
		A_TxEWM[12] = Elem_wise_Mult[1 * ADNN_W_BSTRIDE + 0] + Elem_wise_Mult[2 * ADNN_W_BSTRIDE + 0] + Elem_wise_Mult[3 * ADNN_W_BSTRIDE + 0] * (_FLOAT)4. + Elem_wise_Mult[4 * ADNN_W_BSTRIDE + 0] * (_FLOAT)4.;
		A_TxEWM[13] = Elem_wise_Mult[1 * ADNN_W_BSTRIDE + 1] + Elem_wise_Mult[2 * ADNN_W_BSTRIDE + 1] + Elem_wise_Mult[3 * ADNN_W_BSTRIDE + 1] * (_FLOAT)4. + Elem_wise_Mult[4 * ADNN_W_BSTRIDE + 1] * (_FLOAT)4.;
		A_TxEWM[14] = Elem_wise_Mult[1 * ADNN_W_BSTRIDE + 2] + Elem_wise_Mult[2 * ADNN_W_BSTRIDE + 2] + Elem_wise_Mult[3 * ADNN_W_BSTRIDE + 2] * (_FLOAT)4. + Elem_wise_Mult[4 * ADNN_W_BSTRIDE + 2] * (_FLOAT)4.;
		A_TxEWM[15] = Elem_wise_Mult[1 * ADNN_W_BSTRIDE + 3] + Elem_wise_Mult[2 * ADNN_W_BSTRIDE + 3] + Elem_wise_Mult[3 * ADNN_W_BSTRIDE + 3] * (_FLOAT)4. + Elem_wise_Mult[4 * ADNN_W_BSTRIDE + 3] * (_FLOAT)4.;
		A_TxEWM[16] = Elem_wise_Mult[1 * ADNN_W_BSTRIDE + 4] + Elem_wise_Mult[2 * ADNN_W_BSTRIDE + 4] + Elem_wise_Mult[3 * ADNN_W_BSTRIDE + 4] * (_FLOAT)4. + Elem_wise_Mult[4 * ADNN_W_BSTRIDE + 4] * (_FLOAT)4.;
		A_TxEWM[17] = Elem_wise_Mult[1 * ADNN_W_BSTRIDE + 5] + Elem_wise_Mult[2 * ADNN_W_BSTRIDE + 5] + Elem_wise_Mult[3 * ADNN_W_BSTRIDE + 5] * (_FLOAT)4. + Elem_wise_Mult[4 * ADNN_W_BSTRIDE + 5] * (_FLOAT)4.;
		A_TxEWM[18] = Elem_wise_Mult[1 * ADNN_W_BSTRIDE + 0] - Elem_wise_Mult[2 * ADNN_W_BSTRIDE + 0] + Elem_wise_Mult[3 * ADNN_W_BSTRIDE + 0] * (_FLOAT)8. + Elem_wise_Mult[4 * ADNN_W_BSTRIDE + 0] * (_FLOAT)(-8.) + Elem_wise_Mult[5 * ADNN_W_BSTRIDE + 0];
		A_TxEWM[19] = Elem_wise_Mult[1 * ADNN_W_BSTRIDE + 1] - Elem_wise_Mult[2 * ADNN_W_BSTRIDE + 1] + Elem_wise_Mult[3 * ADNN_W_BSTRIDE + 1] * (_FLOAT)8. + Elem_wise_Mult[4 * ADNN_W_BSTRIDE + 1] * (_FLOAT)(-8.) + Elem_wise_Mult[5 * ADNN_W_BSTRIDE + 1];
		A_TxEWM[20] = Elem_wise_Mult[1 * ADNN_W_BSTRIDE + 2] - Elem_wise_Mult[2 * ADNN_W_BSTRIDE + 2] + Elem_wise_Mult[3 * ADNN_W_BSTRIDE + 2] * (_FLOAT)8. + Elem_wise_Mult[4 * ADNN_W_BSTRIDE + 2] * (_FLOAT)(-8.) + Elem_wise_Mult[5 * ADNN_W_BSTRIDE + 2];
		A_TxEWM[21] = Elem_wise_Mult[1 * ADNN_W_BSTRIDE + 3] - Elem_wise_Mult[2 * ADNN_W_BSTRIDE + 3] + Elem_wise_Mult[3 * ADNN_W_BSTRIDE + 3] * (_FLOAT)8. + Elem_wise_Mult[4 * ADNN_W_BSTRIDE + 3] * (_FLOAT)(-8.) + Elem_wise_Mult[5 * ADNN_W_BSTRIDE + 3];
		A_TxEWM[22] = Elem_wise_Mult[1 * ADNN_W_BSTRIDE + 4] - Elem_wise_Mult[2 * ADNN_W_BSTRIDE + 4] + Elem_wise_Mult[3 * ADNN_W_BSTRIDE + 4] * (_FLOAT)8. + Elem_wise_Mult[4 * ADNN_W_BSTRIDE + 4] * (_FLOAT)(-8.) + Elem_wise_Mult[5 * ADNN_W_BSTRIDE + 4];
		A_TxEWM[23] = Elem_wise_Mult[1 * ADNN_W_BSTRIDE + 5] - Elem_wise_Mult[2 * ADNN_W_BSTRIDE + 5] + Elem_wise_Mult[3 * ADNN_W_BSTRIDE + 5] * (_FLOAT)8. + Elem_wise_Mult[4 * ADNN_W_BSTRIDE + 5] * (_FLOAT)(-8.) + Elem_wise_Mult[5 * ADNN_W_BSTRIDE + 5];

#else
		for (int j = 0; j < ADNN_W_TILE1; ++j)
		{
			for (int i = 0; i < ADNN_W_BLOCK1; ++i)
			{
				A_TxEWM[j * ADNN_W_BLOCK1 + i] = 0;
				for (int k = 0; k < ADNN_W_BLOCK0; ++k)
				{
					// A transposed
					A_TxEWM[j * ADNN_W_BLOCK1 + i] += A[k*ADNN_W_TILE1 + j] * Elem_wise_Mult[k*ADNN_W_BSTRIDE + i];
//					printf("A_TxEwm: oi=%d Av=%f k=%d i=%d\n", j * ADNN_W_BLOCK1 + i, A[k*ADNN_W_TILE1 + j], k, i);
				}

			}
		}
#endif

		*n_mul += 3 * 6 * 2;
		*n_add += 2 * 6 * 4 + 2 * 6 * 3;
		*n_flin = *n_add;

		// x A
		// transpose output

#if 1
		run_f[0 * ADNN_W_OSTRIDE + 0] = (A_TxEWM[0] + A_TxEWM[1] + A_TxEWM[2] + A_TxEWM[3] + A_TxEWM[4]);
		run_f[1 * ADNN_W_OSTRIDE + 0] = (A_TxEWM[1] - A_TxEWM[2] + A_TxEWM[3] * (_FLOAT)2. + A_TxEWM[4] * (_FLOAT) (-2.));
		run_f[2 * ADNN_W_OSTRIDE + 0] = (A_TxEWM[1] + A_TxEWM[2] + A_TxEWM[3] * (_FLOAT)4. + A_TxEWM[4] * (_FLOAT) 4.);
		run_f[3 * ADNN_W_OSTRIDE + 0] = (A_TxEWM[1] - A_TxEWM[2] + A_TxEWM[3] * (_FLOAT)8. + A_TxEWM[4] * (_FLOAT)(-8.) + A_TxEWM[5]);
		run_f[0 * ADNN_W_OSTRIDE + 1] = (A_TxEWM[6] + A_TxEWM[7] + A_TxEWM[8] + A_TxEWM[9] + A_TxEWM[10]);
		run_f[1 * ADNN_W_OSTRIDE + 1] = (A_TxEWM[7] - A_TxEWM[8] + A_TxEWM[9] * (_FLOAT)2. + A_TxEWM[10] * (_FLOAT) (-2.));
		run_f[2 * ADNN_W_OSTRIDE + 1] = (A_TxEWM[7] + A_TxEWM[8] + A_TxEWM[9] * (_FLOAT)4. + A_TxEWM[10] * (_FLOAT) 4.);
		run_f[3 * ADNN_W_OSTRIDE + 1] = (A_TxEWM[7] - A_TxEWM[8] + A_TxEWM[9] * (_FLOAT)8. + A_TxEWM[10] * (_FLOAT)(-8.) + A_TxEWM[11]);
		run_f[0 * ADNN_W_OSTRIDE + 2] = (A_TxEWM[12] + A_TxEWM[13] + A_TxEWM[14] + A_TxEWM[15] + A_TxEWM[16]);
		run_f[1 * ADNN_W_OSTRIDE + 2] = (A_TxEWM[13] - A_TxEWM[14] + A_TxEWM[15] * (_FLOAT)2. + A_TxEWM[16] * (_FLOAT) (-2.));
		run_f[2 * ADNN_W_OSTRIDE + 2] = (A_TxEWM[13] + A_TxEWM[14] + A_TxEWM[15] * (_FLOAT)4. + A_TxEWM[16] * (_FLOAT) 4.);
		run_f[3 * ADNN_W_OSTRIDE + 2] = (A_TxEWM[13] - A_TxEWM[14] + A_TxEWM[15] * (_FLOAT)8. + A_TxEWM[16] * (_FLOAT)(-8.) + A_TxEWM[17]);
		run_f[0 * ADNN_W_OSTRIDE + 3] = (A_TxEWM[18] + A_TxEWM[19] + A_TxEWM[20] + A_TxEWM[21] + A_TxEWM[22]);
		run_f[1 * ADNN_W_OSTRIDE + 3] = (A_TxEWM[19] - A_TxEWM[20] + A_TxEWM[21] * (_FLOAT)2. + A_TxEWM[22] * (_FLOAT) (-2.));
		run_f[2 * ADNN_W_OSTRIDE + 3] = (A_TxEWM[19] + A_TxEWM[20] + A_TxEWM[21] * (_FLOAT)4. + A_TxEWM[22] * (_FLOAT) 4.);
		run_f[3 * ADNN_W_OSTRIDE + 3] = (A_TxEWM[19] - A_TxEWM[20] + A_TxEWM[21] * (_FLOAT)8. + A_TxEWM[22] * (_FLOAT)(-8.) + A_TxEWM[23]);

#else
		for (int j = 0; j < ADNN_W_TILE1; ++j)
		{
			for (int i = 0; i < ADNN_W_TILE0; ++i)
			{

				run_f[i * ADNN_W_OSTRIDE + j] = 0;
				for (int k = 0; k < ADNN_W_BLOCK1; ++k)
				{
					run_f[i * ADNN_W_OSTRIDE + j] += A_TxEWM[j * ADNN_W_BLOCK0 + k] * A[k*ADNN_W_TILE1 + i];
//					printf("EwMxA: i=%d j=%d ii=%d Av=%f\n", i, j, j * ADNN_W_BLOCK0 + k, A[k*ADNN_W_TILE1 + i]);
				}

			}
		}
#endif

		*n_mul += 3*2*4;
		*n_add += (4*2 + 3*2) * 4;
		*n_flin = *n_add;

	}


		void Inverse_BlockTransform_Win(
		int ADNN_W_BSTRIDE,
		int ADNN_W_OSTRIDE,
		int ADNN_W_BLOCK1,
		int ADNN_W_BLOCK0,
		int ADNN_W_TILE1,
		int ADNN_W_TILE0,
		_FLOAT *run_f, const _FLOAT * Elem_wise_Mult, const _FLOAT *A, _FLOAT * A_TxEWM, int *n_mul, int *n_add, int *n_flin)
	{


#if ADNN_ELEMENT_WISE_TRANSOFRMS

#if ADNN_ALG_WIN2x2_3x3
		Inverse_BlockTransform_2x2_3x3Win
#else
		Inverse_BlockTransform_4x4_3x3Win
#endif
			(
			ADNN_W_BSTRIDE, ADNN_W_OSTRIDE,
			ADNN_W_BLOCK1, ADNN_W_BLOCK0,
			ADNN_W_TILE1, ADNN_W_TILE0,
			run_f, Elem_wise_Mult, A, A_TxEWM, n_mul, n_add, n_flin);
#else

		for (int j = 0; j < ADNN_W_TILE1; ++j)
		{
			for (int i = 0; i < ADNN_W_BLOCK1; ++i)
			{
				A_TxEWM[j * ADNN_W_BLOCK1 + i] = 0;
				for (int k = 0; k < ADNN_W_BLOCK0; ++k)
				{
					// A transposed
					A_TxEWM[j * ADNN_W_BLOCK1 + i] += A[k*ADNN_W_TILE1 + j] * Elem_wise_Mult[k*ADNN_W_BSTRIDE + i];
//					printf("A_TxEwm: oi=%d Ai=%d Av=%f k=%d i=%d\n", j * ADNN_W_BLOCK1 + i, k*ADNN_W_TILE1 + j, A[k*ADNN_W_TILE1 + j], k, i);
				}

			}
		}

//		printf("\n");

		// x A
		// transpose output
		for (int j = 0; j < ADNN_W_TILE1; ++j)
		{
			for (int i = 0; i < ADNN_W_TILE0; ++i)
			{

				run_f[i * ADNN_W_OSTRIDE + j] = 0;
				for (int k = 0; k < ADNN_W_BLOCK1; ++k)
				{
					run_f[i * ADNN_W_OSTRIDE + j] += A_TxEWM[j * ADNN_W_BLOCK0 + k] * A[k*ADNN_W_TILE1 + i];
//					printf("EwMxA: i=%d j=%d A_Ti=%d Ai=%d Av=%f\n", i, j, j * ADNN_W_BLOCK0 + k, k*ADNN_W_TILE1 + i, A[k*ADNN_W_TILE1 + i]);
				}

			}
		}

#endif
	}

		void InverseTransform_Win(
			int ADNN_W_BATCH_SZ,
			int ADNN_W_N_ICHNLS,
			int ADNN_W_N_OCHNLS,
			int ADNN_W_HEIGH,
			int ADNN_W_WIDTH,
			int ADNN_W_BSTRIDE,
			int ADNN_W_BCHNL_STRIDE,
			int ADNN_W_BBATCH_STRIDE,
			int ADNN_W_OSTRIDE,
			int ADNN_W_OCHNL_STRIDE,
			int ADNN_W_OBATCH_STRIDE,
			int ADNN_W_BLOCK1,
			int ADNN_W_BLOCK0,
			int ADNN_W_TILE1,
			int ADNN_W_TILE0,
			_FLOAT *w_filtered, const _FLOAT * EwM, const _FLOAT *A, int *n_mul, int *n_add, int *n_flin)
		{

		// Winodrad algorithms

		int y_loop = ADNN_W_HEIGH / ADNN_W_TILE1;
		int x_loop = ADNN_W_WIDTH / ADNN_W_TILE0;
		_FLOAT * A_TxEWM = (_FLOAT *)malloc(ADNN_W_TILE1 * ADNN_W_BLOCK1 * sizeof(_FLOAT));

		for (int b = 0; b < ADNN_W_BATCH_SZ; ++b)
		{
			for (int o = 0; o < ADNN_W_N_OCHNLS; ++o)
			{
				for (int tile_y = 0; tile_y < y_loop; ++tile_y)
				{
					for (int tile_x = 0; tile_x < x_loop; ++tile_x)
					{

						const _FLOAT * Elem_wise_Mult = &EwM[b * ADNN_W_BBATCH_STRIDE + o * ADNN_W_BCHNL_STRIDE + tile_y * ADNN_W_BLOCK1 * ADNN_W_BSTRIDE + tile_x * ADNN_W_BLOCK0];
						_FLOAT * run_f = &w_filtered[b * ADNN_W_OBATCH_STRIDE + o * ADNN_W_OCHNL_STRIDE + tile_y * ADNN_W_TILE1 * ADNN_W_OSTRIDE + tile_x * ADNN_W_TILE0];

						// inverse transform


						Inverse_BlockTransform_Win
							(
							ADNN_W_BSTRIDE, ADNN_W_OSTRIDE,
							ADNN_W_BLOCK1, ADNN_W_BLOCK0,
							ADNN_W_TILE1, ADNN_W_TILE0,
							run_f, Elem_wise_Mult, A, A_TxEWM, n_mul, n_add, n_flin);


					}
				}
			}
		}
		free(A_TxEWM);
	}



	void W2x2_3x3Direct(
		int ADNN_W_BATCH_SZ,
		int ADNN_W_N_ICHNLS, int ADNN_W_ICHNL_STRIDE, int ADNN_W_IBATCH_STRIDE,
		int ADNN_W_N_OCHNLS, int ADNN_W_OCHNL_STRIDE, int ADNN_W_OBATCH_STRIDE,
		int ADNN_W_HEIGH,
		int ADNN_W_WIDTH,
		int ADNN_W_ISTRIDE,
		int ADNN_W_OSTRIDE,
		int ADNN_W_WSTRIDE,
		int ADNN_W_TILE1,
		int ADNN_W_TILE0,
		int ADNN_W_FLTR1,
		int ADNN_W_FLTR0,
		int ADNN_W_PAD1,
		int ADNN_W_PAD0,
	_FLOAT *d_filtered, const _FLOAT *d, const _FLOAT *g)
	{
		// direct filter
		for (int b = 0; b < ADNN_W_BATCH_SZ; ++b)
		{
			for (int o = 0; o < ADNN_W_N_OCHNLS; ++o)
			{
				_FLOAT *run_filtered = &d_filtered[b*ADNN_W_OBATCH_STRIDE + o*ADNN_W_OCHNL_STRIDE];
				for (int j = 0; j < ADNN_W_HEIGH; ++j)
				{
					for (int i = 0; i < ADNN_W_WIDTH; ++i)
					{
						run_filtered[j * ADNN_W_OSTRIDE + i] = 0;
						double accum = 0;
						for (int c = 0; c < ADNN_W_N_ICHNLS; ++c)
						{

							for (int k = 0; k < ADNN_W_FLTR1; ++k)
							{

								int act_j = j + k - ADNN_W_PAD1;

								bool invY = (act_j < 0 || act_j >= ADNN_W_HEIGH);
								act_j = (invY) ? 0 : act_j;

								for (int l = 0; l < ADNN_W_FLTR0; ++l)
								{
									int act_i = (i + l - ADNN_W_PAD0);

									bool invX = (act_i < 0 || act_i >= ADNN_W_WIDTH);
									act_i = (invX) ? 0 : act_i;


									_FLOAT val = d[b*ADNN_W_IBATCH_STRIDE + c *ADNN_W_ICHNL_STRIDE + act_j * ADNN_W_ISTRIDE + act_i];
									_FLOAT we = g[o* ADNN_W_WSTRIDE + c*ADNN_W_FLTR1*ADNN_W_FLTR0 + k * ADNN_W_FLTR0 + l];
									accum += (double)((invX || invY) ? 0 : val) * (double)we;
//									run_filtered[j * ADNN_W_OSTRIDE + i] += ((invX || invY) ? 0 : val) * we;

								}
							}
						}
						run_filtered[j * ADNN_W_OSTRIDE + i] = (_FLOAT)accum;

					}
				}
			}
		}

	}


	// weights transform matrix
	static _FLOAT C_4x4[4*4] =
	{

		1, 0, 0, 0,
		0, 1, -1, 1,
		-1, 1, 1, 0,
		0, 0, 0, -1
	};

	static _FLOAT C_6x6[6*6] =
	{
		4, 0, 0, 0, 0, 0,
		0, -4, 4, -2, 2, 4,
		-5, -4, -4, -1, -1, 0,
		0, 1, -1, 2, -2, -5,
		1, 1, 1, 1, 1, 0,
		0, 0, 0, 0, 0, 1
	};

	// input tarsnform matrix
	static _FLOAT G_4x3[4 * 3] =
	{

		1, 0, 0,
		0.5f, 0.5f, 0.5f,
		0.5f, -0.5f, 0.5f,
		0, 0, 1
	};

	static _FLOAT G_6x3[6 * 3] =
	{
		(float)(1. / 4), 0, 0,
		-(float)(1. / 6.), -(float)(1. / 6.), -(float)(1. / 6.),
		-(float)(1. / 6.), (float)(1. / 6.), -(float)(1. / 6.),
		(float)(1. / 24), (float)(1. / 12), (float)(1. / 6),
		(float)(1. / 24), -(float)(1. / 12), (float)(1. / 6),
		0, 0, 1

	};

	// inverse transform matrix
	static _FLOAT A_4x2[4 * 2] =
	{
		1, 0,
		1, 1,
		1, -1,
		0, -1
	};

	static _FLOAT A_6x4[6*4] =
	{
		1, 0, 0, 0,
		1, 1, 1, 1,
		1, -1, 1, -1,
		1, 2, 4, 8,
		1, -2, 4, -8,
		0, 0, 0, 1
	};



	int aDNNodeConv::RunHostFwdWin(void)
	{
		int ret = 0;

		int pad0 = getPad();
		int kernel_size0 = getKernelSz();
		int pad1 = getPad(0, 1);
		int kernel_size1 = getKernelSz(0, 1);

		aDNNTensor & bot = (aDNNTensor & )getBotFwd();
		const aDNNTensor & top = getTopFwd();
		aDNNTensor & weights = (aDNNTensor & )getBotWeightsFwd();

		int width = (int)bot.getDim(aDNN_TENSOR_WIDTH);
		int height = (int)bot.getDim(aDNN_TENSOR_HEIGHT);
		int inputs = (int)bot.getDim(aDNN_TENSOR_DEPTH);
		int batch_sz = (int)bot.getDim(aDNN_TENSOR_BATCH);
		int bot_stride = (int)bot.getStride(aDNN_TENSOR_WIDTH);
		int bot_channel_stride = (int)bot.getStride(aDNN_TENSOR_HEIGHT);
		int bot_batch_stride = (int)bot.getStride(aDNN_TENSOR_DEPTH);

		int top_stride = (int)top.getStride(aDNN_TENSOR_WIDTH);
		int top_channel_stride = (int)top.getStride(aDNN_TENSOR_HEIGHT);
		int top_height_stride = top_channel_stride / top_stride;
		int	top_batch_stride = (int)top.getStride(aDNN_TENSOR_DEPTH);

		int outputs = (int)top.getDim(aDNN_TENSOR_DEPTH);
		int height_out = (int)top.getDim(aDNN_TENSOR_HEIGHT);
		int width_out = (int)top.getDim(aDNN_TENSOR_WIDTH);


		int weights_hight = (int)weights.getDim(aDNN_TENSOR_HEIGHT);
		int weights_stride = (int)weights.getStride(aDNN_TENSOR_WIDTH);



		int ADNN_WI_TILE1 = tile_sz1_;
		int ADNN_WI_TILE0 = tile_sz0_;

		int ADNN_WI_WIDTH = width;
		int ADNN_WI_HEIGH = height;
		int ADNN_WI_BATCH_SZ = batch_sz;
		int ADNN_WI_N_ICHNLS = inputs;
		int ADNN_WI_N_OCHNLS = outputs;
		int ADNN_WI_FLTR0 = kernel_size0;
		int ADNN_WI_FLTR1 = kernel_size1;

		int ADNN_WI_PAD0 = pad0;
		int ADNN_WI_PAD1 = pad1;

		int ADNN_WI_BLOCK1 = (ADNN_WI_TILE1 + ADNN_WI_PAD1 * 2);
		int ADNN_WI_BLOCK0 = (ADNN_WI_TILE0 + ADNN_WI_PAD0 * 2);

		int ADNN_WI_ISTRIDE = bot_stride;
		int ADNN_WI_OSTRIDE = top_stride;
		int ADNN_WI_ICHNL_STRIDE = bot_channel_stride;
		int ADNN_WI_OCHNL_STRIDE = top_channel_stride;
		int ADNN_WI_IBATCH_STRIDE = bot_batch_stride;
		int ADNN_WI_OBATCH_STRIDE = top_batch_stride;
		int ADNN_WI_BSTRIDE = ((ADNN_WI_WIDTH / ADNN_WI_TILE0) * ADNN_WI_BLOCK0);
		int ADNN_WI_BCHNL_STRIDE = (ADNN_WI_BSTRIDE * (ADNN_WI_HEIGH / ADNN_WI_TILE1) * ADNN_WI_BLOCK1);
		int ADNN_WI_IBBATCH_STRIDE = (ADNN_WI_BCHNL_STRIDE * ADNN_WI_N_ICHNLS);
		int ADNN_WI_OBBATCH_STRIDE = (ADNN_WI_BCHNL_STRIDE * ADNN_WI_N_OCHNLS);
		int ADNN_WI_WSTRIDE = weights_stride; // (ADNN_WI_N_ICHNLS * ADNN_WI_FLTR1 * ADNN_WI_FLTR0)


// constants
		const _FLOAT * C;  // weights transform matrix
		const _FLOAT * G;  // input tarsnform matrix
		const _FLOAT * A;  // inverse transform matrix
// 2x2
		if ( ADNN_WI_TILE1 == 2)
		{
			C = C_4x4;
			G = G_4x3;
			A = A_4x2;
		}
// 4x4
		else
		{
			C = C_6x6;
			G = G_6x3;
			A = A_6x4;
		}





		// weights
		aDType *w = NULL;
		aDType* t_w = NULL;
		// input
		aDType *d = NULL;
		// transformed input
		aDType *blocked_d = NULL;

		// element-wise multiply
		aDType *EwM = NULL;

		// filtered output
		aDType *w_filtered = NULL;

#if 1

// init weights
		aDNNTensor & tweights_vf = getSlot(getWeightsNm() + ADNN_WIN_TRANSFORM_NM + ADNN_VERIFY_NM);

		w = (aDType *)weights.accessTensor(ADNN_MEM_ACCESS_READ);

		t_w = (aDType*)tweights_vf.accessTensor(ADNN_MEM_ACCESS_WRITE);

// data
// init data
		d = (aDType*) bot.accessTensor(ADNN_MEM_ACCESS_READ);
		aDNNTensor & tbot_vf = getSlot(getBotNm() + ADNN_WIN_TRANSFORM_NM + ADNN_VERIFY_NM);
		blocked_d = (aDType *)tbot_vf.accessTensor(ADNN_MEM_ACCESS_WRITE);

		aDNNTensor & ttop_vf = getSlot(getTopNm() + ADNN_WIN_TRANSFORM_NM + ADNN_VERIFY_NM);

		EwM = (aDType *)ttop_vf.accessTensor(ADNN_MEM_ACCESS_WRITE);
		// get from the list of tensors referred by this node
		aDNNTensor & top_vf = getSlot(getTopNm() + ADNN_VERIFY_NM);;

		w_filtered = (aDType*)top_vf.accessTensor(ADNN_MEM_ACCESS_WRITE);
#else
		w = (aDType*)malloc(ADNN_WI_WSTRIDE * ADNN_WI_N_OCHNLS * sizeof(aDType));
		for (int i = 0; i < ADNN_WI_WSTRIDE * ADNN_WI_N_OCHNLS; ++i)
		{
			w[i] = (aDType)(((double)rand() / RAND_MAX) - 0.5) * 0.001;
		}
		// transformed weights
		t_w = (aDType*)malloc(ADNN_WI_N_ICHNLS * ADNN_WI_BLOCK1 * ADNN_WI_BLOCK0 * ADNN_WI_N_OCHNLS * sizeof(aDType));
		// input
		d = (aDType*)malloc(ADNN_WI_BATCH_SZ *ADNN_WI_N_ICHNLS * ADNN_WI_ICHNL_STRIDE * sizeof(aDType));
		// transformed input
		blocked_d = (aDType*)malloc(ADNN_WI_BATCH_SZ * ADNN_WI_IBBATCH_STRIDE * sizeof(aDType));
		// data
		// init data
#if 1
		for (int i = 0; i < ADNN_WI_BATCH_SZ * ADNN_WI_IBATCH_STRIDE; ++i)
		{
			d[i] = (float)((double)rand() / RAND_MAX)/* * ((i & 1) ? 0 : 1)*/;

		}
#else
		d[2] = d[0] = (float)(((double)rand() / RAND_MAX) - 0.5);
		d[3] = d[1] = (float)(((double)rand() / RAND_MAX) - 0.5);
		d[6] = d[4] = (float)(((double)rand() / RAND_MAX) - 0.5);
		d[7] = d[5] = (float)(((double)rand() / RAND_MAX) - 0.5);

#endif
		EwM = (aDType*)malloc(ADNN_WI_BATCH_SZ * ADNN_WI_OBBATCH_STRIDE * sizeof(aDType));

		// filtered output
		w_filtered = (aDType*)malloc(ADNN_WI_BATCH_SZ * ADNN_WI_OBATCH_STRIDE * sizeof(aDType));

#endif

		// direct filtering
		float *d_filtered = (float*)malloc(ADNN_WI_BATCH_SZ * ADNN_WI_OBATCH_STRIDE *sizeof(float));

		W2x2_3x3Direct(
			ADNN_WI_BATCH_SZ,
			ADNN_WI_N_ICHNLS, ADNN_WI_ICHNL_STRIDE, ADNN_WI_IBATCH_STRIDE,
			ADNN_WI_N_OCHNLS, ADNN_WI_OCHNL_STRIDE, ADNN_WI_OBATCH_STRIDE,
			ADNN_WI_HEIGH, ADNN_WI_WIDTH, ADNN_WI_ISTRIDE, ADNN_WI_OSTRIDE,
			ADNN_WI_WSTRIDE,
			ADNN_WI_TILE1, ADNN_WI_TILE0,
			ADNN_WI_FLTR1, ADNN_WI_FLTR0, ADNN_WI_PAD1, ADNN_WI_PAD0,
			(float*)d_filtered, (const float*)d, (const float*)w);

// winograd algorithm
		int n_mul = 0, n_add = 0, n_flin = 0;



// transforms weights
// once per batch
		Weights_Tansform_Win(
			ADNN_WI_N_ICHNLS,
			ADNN_WI_N_OCHNLS,
			ADNN_WI_BLOCK1,
			ADNN_WI_BLOCK0,
			ADNN_WI_FLTR1,
			ADNN_WI_FLTR0,
		t_w, w, G, &n_mul, &n_add, &n_flin);

// arrange and transform input tiles
		Input_Tansform_Win(
			ADNN_WI_BATCH_SZ,
			ADNN_WI_N_ICHNLS,
			ADNN_WI_HEIGH,ADNN_WI_WIDTH,
			ADNN_WI_ISTRIDE, ADNN_WI_ICHNL_STRIDE, ADNN_WI_IBATCH_STRIDE,
			ADNN_WI_BSTRIDE, ADNN_WI_BCHNL_STRIDE, ADNN_WI_IBBATCH_STRIDE,
			ADNN_WI_BLOCK1,ADNN_WI_BLOCK0,
			ADNN_WI_TILE1,ADNN_WI_TILE0,
			ADNN_WI_PAD1,ADNN_WI_PAD0,
		blocked_d, d, C, &n_mul, &n_add, &n_flin);

// element-wise multiply

		ElemWise_Multiply_Win(
			ADNN_WI_BATCH_SZ,
			ADNN_WI_N_ICHNLS, ADNN_WI_N_OCHNLS,
			ADNN_WI_HEIGH,ADNN_WI_WIDTH,
			ADNN_WI_BSTRIDE, ADNN_WI_BCHNL_STRIDE,
			ADNN_WI_IBBATCH_STRIDE, ADNN_WI_OBBATCH_STRIDE,
			ADNN_WI_BLOCK1,ADNN_WI_BLOCK0,
			ADNN_WI_TILE1,ADNN_WI_TILE0,
			EwM, blocked_d, t_w, &n_mul, &n_add, &n_flin);


		InverseTransform_Win(
					ADNN_WI_BATCH_SZ,
					ADNN_WI_N_ICHNLS, ADNN_WI_N_OCHNLS,
					ADNN_WI_HEIGH, ADNN_WI_WIDTH,
					ADNN_WI_BSTRIDE, ADNN_WI_BCHNL_STRIDE, ADNN_WI_OBBATCH_STRIDE,
					ADNN_WI_OSTRIDE, ADNN_WI_OCHNL_STRIDE, ADNN_WI_OBATCH_STRIDE,
					ADNN_WI_BLOCK1, ADNN_WI_BLOCK0, ADNN_WI_TILE1, ADNN_WI_TILE0,
					w_filtered, EwM, A, &n_mul, &n_add, &n_flin);




		int t_n_add = ADNN_WI_BATCH_SZ * ADNN_WI_WIDTH * ADNN_WI_HEIGH * ADNN_WI_FLTR1 * ADNN_WI_FLTR0 * ADNN_WI_N_ICHNLS * ADNN_WI_N_OCHNLS;
		int t_n_mul = ADNN_WI_BATCH_SZ * ADNN_WI_WIDTH * ADNN_WI_HEIGH * ADNN_WI_FLTR1 * ADNN_WI_FLTR0 * ADNN_WI_N_ICHNLS * ADNN_WI_N_OCHNLS;
		int t_n_flin = ADNN_WI_BATCH_SZ * ADNN_WI_WIDTH * ADNN_WI_HEIGH * ADNN_WI_FLTR1 * ADNN_WI_FLTR0 * ADNN_WI_N_ICHNLS * ADNN_WI_N_OCHNLS;
		printf(
#if ADNN_ALG_WIN2x2_3x3
			"2x2,3x3: "
#else
			"4x4,3x3: "
#endif
			"bz=%d ic=%d oc=%d wi=%d hi=%d\ntheory: adds=%d mults=%d flins=%d\nwinograd: adds=%d mults=%d flins=%d\nratio: adds=%3.1f mults=%3.1f flins=%3.1f\n",
			ADNN_WI_BATCH_SZ,
			ADNN_WI_N_ICHNLS,
			ADNN_WI_N_OCHNLS,
			ADNN_WI_WIDTH,
			ADNN_WI_HEIGH,
			t_n_add,
			t_n_mul,
			t_n_flin,
			n_add,
			n_mul,
			n_flin,
			(float)n_add / t_n_add,
			(float)n_mul / t_n_mul,
			(float)n_flin/ t_n_flin
			);



// trnspose output
		int match = 1;
		double max_err = 0, cmax_err = 0;
		int mb = 0, mo = 0, mi = 0, mj = 0;
		double eps = (1 << 8);
		size_t err_cnt = 0;
		for (int b = 0; b < ADNN_WI_BATCH_SZ && match; ++b)
		{
			for (int o = 0; o < ADNN_WI_N_OCHNLS && match; ++o)
			{
				for (int j = 0; j < ADNN_WI_HEIGH && match; ++j)
				{
					for (int i = 0; i < ADNN_WI_WIDTH && match; ++i)
					{
						float d_v = d_filtered[b * ADNN_WI_OBATCH_STRIDE + o * ADNN_WI_OCHNL_STRIDE + j*ADNN_WI_OSTRIDE + i];
						float w_v = w_filtered[b * ADNN_WI_OBATCH_STRIDE + o * ADNN_WI_OCHNL_STRIDE + j*ADNN_WI_OSTRIDE + i];
						double err = CalculateErr(d_v, w_v);
						cmax_err = fmax(err, max_err);
						if (max_err != cmax_err/* && (abs(d_v) >= 0.0000001 || abs(w_v) >= 0.0000001)*/)
						{
							max_err = cmax_err;
							mb = b; mo = o; mi = i; mj = j;
						}
						if (err > eps /*&& (abs(d_v) >= 0.0000001 || abs(w_v) >= 0.0000001)*/)
						{
//							printf("ERROR %d at b=%d o=%d i=%d j=%d d_v=%f w_v=%f d=%12.11f\n", (int)err, b, o, i, j, d_v, w_v,
//								fabs(w_v - d_v));
							err_cnt++;
//							match = 0;
						}
					}
				}
			}
		}
		float d_v = d_filtered[mb * ADNN_WI_OBATCH_STRIDE + mo * ADNN_WI_OCHNL_STRIDE + mj*ADNN_WI_OSTRIDE + mi];
		float w_v = w_filtered[mb * ADNN_WI_OBATCH_STRIDE + mo * ADNN_WI_OCHNL_STRIDE + mj*ADNN_WI_OSTRIDE + mi];
		printf("N ERRORs exeeding %d = %d, %5.4f%%\nMAX ERROR %d at b=%d o=%d i=%d j=%d d_v=%12.11f w_v=%12.11f dif=%12.11f\n", (int)eps, (int)err_cnt, ((double)err_cnt * 100) / (double)n_flin, (int)max_err, mb, mo, mi, mj, d_v, w_v, fabs(w_v - d_v));

		if (d_filtered)
		{
			free(d_filtered);
		}




#if 1
		if (w_filtered)
		{
			top_vf.commitTensor();
		}

		if (EwM)
		{
			ttop_vf.commitTensor();
		}
		if (blocked_d)
		{
			tbot_vf.commitTensor();
		}

		if (d)
		{
			bot.commitTensor();
		}

		if (t_w)
		{
			tweights_vf.commitTensor();
		}

		if (w)
		{

			weights.commitTensor();

		}

#else
		if ( w_filtered)
		{
			free(w_filtered);
		}
		if (EwM)
		{
			free(EwM);
		}

		if (blocked_d)
		{
			free(blocked_d);
		}

		if (d)
		{
			free(d);
		}

		if (t_w)
		{
			free(t_w);
		}

		if (w)
		{
			free(w);
		}
#endif
		return(ret);
	}



	int aDNNodeConv::VerifyFwdWin(void)
	{
		int ret = 0;
		ret = RunHostFwdWin();
		int pad0 = getPad();
		int pad1 = getPad(0, 1);

		const aDNNTensor & bot = getBotFwd();
		aDNNTensor & top = (aDNNTensor &)getTopFwd();
		int inputs = (int)bot.getDim(aDNN_TENSOR_DEPTH);
		int outputs = (int)top.getDim(aDNN_TENSOR_DEPTH);
		int batch_sz = (int)bot.getDim(aDNN_TENSOR_BATCH);
		int block_sz0 = tile_sz0_ + pad0 * 2;
		int block_sz1 = tile_sz1_ + pad1 * 2;
		int tweight_stride_overout = inputs * block_sz0 * block_sz1;
		int tweight_stride_overinp = outputs * block_sz0 * block_sz1;

		// weights trasnform
		aDNNTensor & tweights_vf = getSlot(getWeightsNm() + ADNN_WIN_TRANSFORM_NM + ADNN_VERIFY_NM);
		aDNNTensor & tweights = getSlot(getWeightsNm() + ADNN_WIN_TRANSFORM_NM);

		aDType* twi_vf_ptr = (aDType*)tweights_vf.accessTensor(ADNN_MEM_ACCESS_READ);
		aDType* twi_ptr = (aDType*)tweights.accessTensor(ADNN_MEM_ACCESS_READ);
		size_t size = tweights.getSize();
		int eps = 16;
		int match_we = 1;
		for (size_t i = 0; i < size && match_we; ++i)
		{
			int o;
			int r;
			int c;
			int j;
			//s tandar layout generated by verification tool
			o = i / tweight_stride_overout;
			r = i % tweight_stride_overout;
			c = r / (block_sz0 * block_sz1);
			int k = r % (block_sz0 * block_sz1);

#if ADNN_WWT_PEROUTPUT
			j = i;
#else
			j = c * tweight_stride_overinp + o * (block_sz0 * block_sz1) + k;
#endif
			int kj = k / tile_sz0_;
			int ki = k % tile_sz0_;

			aDType c_v = twi_vf_ptr[i];
			aDType g_v = twi_ptr[j];
			double err = CalculateErr(c_v, g_v);

			if (err > eps)
			{

				printf("weights transform error at o=%d c=%d kj=%d ki=%d c_v=%f g_v=%f\n", o, c, kj, ki, c_v, g_v);
				match_we = 0;
			}
		}

		tweights_vf.commitTensor();
		tweights.commitTensor();
		if (match_we)
		{
			printf("passed weights transform\n");
		}

		// input transform
		aDNNTensor & tbot_vf = getSlot(getBotNm() + ADNN_WIN_TRANSFORM_NM + ADNN_VERIFY_NM);
		aDNNTensor & tbot = getSlot(getBotNm() + ADNN_WIN_TRANSFORM_NM);
		int tbot_vf_stride = (int)tbot_vf.getStride(aDNN_TENSOR_WIDTH);
		int tbot_vf_channel_stride = (int)tbot_vf.getStride(aDNN_TENSOR_HEIGHT);
		int tbot_vf_batch_stride = (int)tbot_vf.getStride(aDNN_TENSOR_DEPTH);
		int tbot_stride = (int)tbot.getStride(aDNN_TENSOR_WIDTH);
		int tbot_channel_stride = (int)tbot.getStride(aDNN_TENSOR_HEIGHT);
		int tbot_batch_stride = (int)tbot.getStride(aDNN_TENSOR_DEPTH);
		int twidth = (int)tbot.getDim(aDNN_TENSOR_WIDTH);
		int theight = (int)tbot.getDim(aDNN_TENSOR_HEIGHT);
		int n_blocks1 = theight / block_sz1;
		int n_blocks0 = twidth / block_sz0;
		int n_blocks = (theight*twidth) / (block_sz1 * block_sz0);

		if (!win2x2_)
		{

			aDType* tbot_vf_ptr = (aDType*)tbot_vf.accessTensor(ADNN_MEM_ACCESS_READ);
			aDType* tbot_ptr = (aDType*)tbot.accessTensor(ADNN_MEM_ACCESS_READ);
			eps = 256;
			int match_tbot = 1;
			size_t iinv_err_cnt = 0;
			for (int b = 0; b < batch_sz && match_tbot; ++b)
			{
				for (int c = 0; c < inputs && match_tbot; ++c)
				{
					int block_off = b*tbot_batch_stride + c*tbot_channel_stride;
					int block_vf_off = b*tbot_vf_batch_stride + c*tbot_vf_channel_stride;
#if ADNN_WIT_OUTLINEAR
					for (int by = 0; by < n_blocks1 && match_tbot; ++by)
					{
						for (int bx = 0; bx < n_blocks0 && match_tbot; ++bx)
						{

							for (int j = 0; j < block_sz1 && match_tbot; ++j)
							{
								for (int i = 0; i < block_sz0 && match_tbot; ++i)
								{
									aDType c_v = tbot_vf_ptr[block_vf_off + by * block_sz1 * tbot_vf_stride + bx * block_sz0 + j * tbot_vf_stride + i];
									aDType g_v = tbot_ptr[block_off + by * block_sz1 * tbot_stride + bx * block_sz0 + j * tbot_stride + i];
									double err = CalculateErr(c_v, g_v);

									if (err > eps)
									{

										//									printf("input transform error %d at b=%d c=%d by=%d bx=%d j=%d i=%d c_v=%f g_v=%f\n", (int)err, b, c, by, bx, j, i, c_v, g_v);
										//									match_tbot = 0;
										iinv_err_cnt++;
									}
								}
							}
						}
					}
#else
					for (int bk = 0; bk < n_blocks && match_tbot; ++bk)
					{
						int by = bk / n_blocks0;
						int bx = bk % n_blocks0;

						for (int j = 0; j < block_sz1 && match_tbot; ++j)
						{
							for (int i = 0; i < block_sz0 && match_tbot; ++i)
							{
								aDType c_v = tbot_vf_ptr[block_vf_off + by * block_sz1 * tbot_vf_stride + bx * block_sz0 + j * tbot_vf_stride + i];
								aDType g_v = tbot_ptr[block_off + bk * block_sz1 * block_sz0 + j * block_sz0 + i];
								double err = CalculateErr(c_v, g_v);

								if (err > eps)
								{

									//								printf("input transform error %d at b=%d c=%d by=%d bx=%d j=%d i=%d c_v=%f g_v=%f\n", (int)err, b, c, by, bx, j, i, c_v, g_v);
									//								match_tbot = 0;
									iinv_err_cnt++;
								}
							}
						}

					}
#endif

				}
			}

			tbot_vf.commitTensor();
			tbot.commitTensor();
			if (match_tbot)
			{
				printf("passed input transform with n of errs exeed %d = %d\n", eps, (int)iinv_err_cnt);
			}

		}

#if 0
		// element-wise multiplication
		aDNNTensor & ttop_vf = getSlot(getTopNm() + ADNN_WIN_TRANSFORM_NM + ADNN_VERIFY_NM);
		aDNNTensor & ttop = getSlot(getTopNm() + ADNN_WIN_TRANSFORM_NM);
		int ttop_vf_stride = (int)ttop_vf.getStride(aDNN_TENSOR_WIDTH);
		int ttop_vf_channel_stride = (int)ttop_vf.getStride(aDNN_TENSOR_HEIGHT);
		int ttop_vf_batch_stride = (int)ttop_vf.getStride(aDNN_TENSOR_DEPTH);
		int ttop_stride = (int)ttop.getStride(aDNN_TENSOR_WIDTH);
		int ttop_channel_stride = (int)ttop.getStride(aDNN_TENSOR_HEIGHT);
		int ttop_batch_stride = (int)ttop.getStride(aDNN_TENSOR_DEPTH);
#if 0
		int twidth = (int)ttop.getDim(aDNN_TENSOR_WIDTH);
		int theight = (int)ttop.getDim(aDNN_TENSOR_HEIGHT);
		int n_blocks1 = theight / block_sz1;
		int n_blocks0 = twidth / block_sz0;
		int n_blocks = (theight*twidth) / (block_sz1 * block_sz0);
#endif
		aDType* ttop_vf_ptr = (aDType*)ttop_vf.accessTensor(ADNN_MEM_ACCESS_READ);
		aDType* ttop_ptr = (aDType*)ttop.accessTensor(ADNN_MEM_ACCESS_READ);
		eps = 256;
		int match_ttop = 1;
		size_t ewm_err_cnt = 0;

		for (int b = 0; b < batch_sz && match_ttop; ++b)
		{
			for (int o = 0; o < outputs && match_ttop; ++o)
			{
				int block_off = b*ttop_batch_stride + o*ttop_channel_stride;
				int block_vf_off = b*ttop_vf_batch_stride + o*ttop_vf_channel_stride;

#if ADNN_WEWM_OUTLINEAR
				for (int by = 0; by < n_blocks1 && match_ttop; ++by)
				{
					for (int bx = 0; bx < n_blocks0 && match_ttop; ++bx)
					{
#if 0
						if (b == 0 && o == 1 && by == 0 && bx == 0)
						{
							printf("c:mit:\n");

						}
#endif
						for (int j = 0; j < block_sz1 && match_ttop; ++j)
						{
							for (int i = 0; i < block_sz0 && match_ttop; ++i)
							{
#if 0
								if (b == 0 && o == 1 && by == 0 && bx == 0)
								{
									aDType c_v = ttop_vf_ptr[block_vf_off + by * block_sz1 * ttop_vf_stride + bx * block_sz0 + j * ttop_vf_stride + i];
									printf("%f ", c_v);

								}
#endif
								aDType c_v = ttop_vf_ptr[block_vf_off + by * block_sz1 * ttop_vf_stride + bx * block_sz0 + j * ttop_vf_stride + i];
								aDType g_v = ttop_ptr[block_off + by * block_sz1 * ttop_stride + bx * block_sz0 + j * ttop_stride + i];
								double err = CalculateErr(c_v, g_v);

								if ((int)err > eps)
								{

																		printf("element-wise multiplication error %dulps at b=%d o=%d by=%d bx=%d j=%d i=%d c_v=%f g_v=%f del=%11.10f\n", (int)err, b, o, by, bx, j, i, c_v, g_v, fabs(c_v- g_v));
																		match_ttop = 0;
									ewm_err_cnt++;
								}
							}
#if 0
							if (b == 0 && o == 1 && by == 0 && bx == 0)
							{
								printf("\n");

							}
#endif
						}
					}
				}


#else
				for (int bk = 0; bk < n_blocks && match_tbot; ++bk)
				{
					int by = bk / n_blocks0;
					int bx = bk % n_blocks0;

					for (int j = 0; j < block_sz1 && match_tbot; ++j)
					{
						for (int i = 0; i < block_sz0 && match_tbot; ++i)
						{
							aDType c_v = ttop_vf_ptr[block_vf_off + by * block_sz1 * ttop_vf_stride + bx * block_sz0 + j * ttop_vf_stride + i];
							aDType g_v = ttop_ptr[block_off + bk * block_sz1 * block_sz0 +j * block_sz0 + i];
							double err = CalculateErr(c_v, g_v);

							if (err > eps)
							{

								printf("element-wise multiplication error %d at b=%d o=%d by=%d bx=%d j=%d i=%d c_v=%f g_v=%f\n", (int)err, b, o, by, bx, j, i, c_v, g_v);
								match_tbot = 0;
								ewm_err_cnt++;
							}
						}
					}

				}

#endif

			}
		}


		tbot_vf.commitTensor();
		tbot.commitTensor();
		if (match_ttop)
		{
			printf("passed element-wise multiplication with n of errs exeed %d = %d\n", eps, (int)ewm_err_cnt);
		}

#endif



			// inverse transform

			aDNNTensor & top_vf = getSlot(getTopNm() + ADNN_VERIFY_NM);
			int top_vf_stride = (int)top_vf.getStride(aDNN_TENSOR_WIDTH);
			int top_vf_channel_stride = (int)top_vf.getStride(aDNN_TENSOR_HEIGHT);
			int top_vf_batch_stride = (int)top_vf.getStride(aDNN_TENSOR_DEPTH);
			int top_stride = (int)top.getStride(aDNN_TENSOR_WIDTH);
			int top_channel_stride = (int)top.getStride(aDNN_TENSOR_HEIGHT);
			int top_batch_stride = (int)top.getStride(aDNN_TENSOR_DEPTH);
			int width_out = (int)top.getDim(aDNN_TENSOR_WIDTH);
			int height_out = (int)top.getDim(aDNN_TENSOR_HEIGHT);

			aDType* top_vf_ptr = (aDType*)top_vf.accessTensor(ADNN_MEM_ACCESS_READ);
			aDType* top_ptr = (aDType*)top.accessTensor(ADNN_MEM_ACCESS_READ);
			eps = 256;
			int match_inv = 1;
			size_t oinv_err_cnt = 0;

			for (int b = 0; b < batch_sz && match_inv; ++b)
			{
				for (int o = 0; o < outputs && match_inv; ++o)
				{
					int block_off = b*top_batch_stride + o*top_channel_stride;
					int block_vf_off = b*top_vf_batch_stride + o*top_vf_channel_stride;

					for (int j = 0; j < height_out && match_inv; ++j)
					{
						for (int i = 0; i < width_out && match_inv; ++i)
						{
							aDType c_v = top_vf_ptr[block_vf_off + j * top_vf_stride + i];
							aDType g_v = top_ptr[block_off + j * top_stride + i];
							double err = CalculateErr(c_v, g_v);

							if ((int)err > eps)
							{

															printf("inverse trasnform error %dulps at b=%d o=%d j=%d i=%d c_v=%f g_v=%f del=%11.10f\n", (int)err, b, o, j, i, c_v, g_v, fabs(c_v - g_v));
															match_inv = 0;
								oinv_err_cnt++;
							}
						}
					}

				}
			}


			top_vf.commitTensor();
			top.commitTensor();
			if (match_inv)
			{
				printf("passed inverse trasnform with n of errs exeed %d = %d\n", eps, (int)oinv_err_cnt);
			}


		return(ret);
	}

} // adnn



