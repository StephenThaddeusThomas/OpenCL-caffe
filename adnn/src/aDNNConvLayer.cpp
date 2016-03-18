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

#define ADNN_CONV_WIN_ALG 0
#define ADNN_GENERIC_CONV 0

namespace adnn
{




	/************************************************************************************************************************
	**
	**			CONVOLUTIONAL LAYER
	**
	************************************************************************************************************************/


	/************************************************************************************************************************
	**
	**			aDNNodeConv Class
	**
	************************************************************************************************************************/

	/**
	* Constructors
	*/
	aDNNodeConv::aDNNodeConv(const ADNNBase & lib, const adnn_node_parameters & node_params)
		:aDNNode(lib, node_params)
	{
	}


	aDNNodeConv::aDNNodeConv()
		: aDNNode()
	{
	}


	aDNNodeConv::aDNNodeConv(const aDNNodeConv & rh)
	{
		*this = rh;
	}

	const aDNNode & aDNNodeConv:: operator = (const aDNNodeConv & rh)
	{
		*(aDNNode*)this = *(aDNNode*)&rh;
		return *this;
	}

	/**
	* Destructor
	*/

	aDNNodeConv::~aDNNodeConv(void)
	{
	}


	int aDNNodeConv::Connect(void)
	{
		int ret = 0;
		return(ret);
	}




	int aDNNodeConv::Run(void)
	{
		int ret = 0;
		// forward


		return(ret);

	}


	/************************************************************************************************************************
	**
	**			FORWARD PROPAGATION
	**
	************************************************************************************************************************/

	int aDNNodeConv::Construct(void)
	{
		int ret = 0;


		// to create internal system memory tensor for verification
		ret = ConstructOutput();
		const aDNNTensor & bot = getBotFwd();
		const aDNNTensor & top = getTopFwd();
		int width = (int)bot.getDim(aDNN_TENSOR_WIDTH);
		int height = (int)bot.getDim(aDNN_TENSOR_HEIGHT);
		int inputs = (int)bot.getDim(aDNN_TENSOR_DEPTH);
		int stride = getKernelStride();
		int kernel_size = getKernelSz();
		int outputs = (int)top.getDim(aDNN_TENSOR_DEPTH);

		win_alg_ = false;
#if ADNN_GENERIC_CONV
		old_ = true;
#else
		old_ = ((kernel_size != 3 && kernel_size != 5) || stride > 1);
#endif
#if 1
#if ADNN_CONV_WIN_ALG
		// TO DO:: why 33????
		win_alg_ = true; // (kernel_size == 3 && inputs >= 4 && inputs < 33) ? true : false;
#endif

		if (old_)
		{
#if ADNN_GENERIC_CONV
			ret = ConstructGen_NCHW();

#else
			ret = ConstructGT32_NCHW();
#endif
		}


// TO DO:: why 33????
		else if (win_alg_)
		{
			ret = ConstructFwdWin_NCHW();
		}
		else if (width <= 16 && height <= 16)
		{
//			ret = Construct_NCHW_N3();
			ret = ConstructLE32_NCHW();
		}
		else
		{
			ret = Construct_NCHW_N3();
//			ret = ConstructLE32_NCHW();
		}
#else

		ret = ConstructGT32_NCHW();
#endif

		return(ret);
	}

	int aDNNodeConv::ConstructGen_NCHW(void)
	{

		//ADNN_GRP_SZ0              group size in dim 0
		//ADNN_GRP_SZ1				group size in dim 1
		//ADNN_GRP_LG2SZ0           log2 group size in dim 0
		//ADNN_GRP_LG2SZ1           log2 group size in dim 1
		//ADNN_GRP_LG2SZ2           log2 group size in dim 2
		//ADNN_GRP_SZ               n of wk-item in the group
		//ADNN_N_IN_CHNLS			total number of input channels
		//ADNN_LCL_N_IN_CHNLS		n of localy kept input channels
		//ADNN_IN_WIDTH				input width in NCHW layout
		//ADNN_IN_HEIGHT			input height stride in NCHW layout
		//ADNN_IN_STRIDE			input stride in NCHW layout
		//ADNN_IN_CHNL_STRIDE       input channel stride in NCHW layout
		//ADNN_IN_BATCH_STRIDE      input batch stride in NCHW layout
		//ADNN_BATCH_SZ		        batch szie
		//ADNN_N_IN_PIX_SZ0        n input pixels per wk item in 0 dim
		//ADNN_N_IN_PIX_SZ1		n input pexels per wk item in 1 dim
		//ADNN_FLTR_SZ0             filter 0 dim size
		//ADNN_FLTR_PAD_SZ0				filter 0 dim pad
		//ADNN_FLTR_STRIDE0			filter 0 dim stride
		//ADNN_FLTR_SZ1             filter 1 dim size
		//ADNN_FLTR_PAD_SZ1				filter 1 dim pad
		//ADNN_FLTR_STRIDE1			filter 1 dim stride
		//ADNN_IN_CHNL_LOOP         main input channel loop
		//ADNN_OUT_WIDTH			output width in NCHW layout
		//ADNN_OUT_HEIGHT			output height stride in NCHW layout
		//ADNN_OUT_STRIDE			output stride in NCHW layout
		//ADNN_N_OUT_PIX_SZ0        n output pixel per wk item in 0 dim
		//ADNN_N_OUT_PIX_SZ1		n output pexels per wk item in 1 dim
		//ADNN_N_STACKS           n of separate data stacks
		//ADNN_N_PROCS1           n of processors per stack 1 dim
		//ADNN_N_PROCS0           n of processors per stack 0 dim
		//ADNN_ALIGNED             dimesions aligned to 2
		//ADNN_BATCH_ALIGNED      batch is multiple of n_ins
		//ADNN_IN_SZ0			horizontal read dim 0
		//ADNN_IN_SZ1			vertical read dim 1

		int ret = 0;

		// edge 0, dim 0
		int pad0 = getPad();
		int stride0 = getKernelStride();
		int kernel_size0 = getKernelSz();
		// edge 0, dim 1
		int pad1 = getPad(0, 1);
		int stride1 = getKernelStride(0, 1);
		int kernel_size1 = getKernelSz(0, 1);


		const aDNNTensor & bot = getBotFwd();
		const aDNNTensor & top = getTopFwd();

		int width = (int)bot.getDim(aDNN_TENSOR_WIDTH);
		int height = (int)bot.getDim(aDNN_TENSOR_HEIGHT);
		int inputs = (int)bot.getDim(aDNN_TENSOR_DEPTH);
		int batch_sz = (int)bot.getDim(aDNN_TENSOR_BATCH);
		int bot_stride = (int)bot.getStride(aDNN_TENSOR_WIDTH);
		int bot_channel_stride = (int)bot.getStride(aDNN_TENSOR_HEIGHT);
		int bot_batch_stride = (int)bot.getStride(aDNN_TENSOR_DEPTH);

//		int outputs = (int)top.getDim(aDNN_TENSOR_DEPTH);
		int height_out = (int)top.getDim(aDNN_TENSOR_HEIGHT);
		int width_out = (int)top.getDim(aDNN_TENSOR_WIDTH);


		// cols for img2cols
		adnn_data_parameters cols_params;
		memset(&cols_params, 0, sizeof(adnn_data_parameters));
		cols_params.data_format = ADNN_DF_FP32;

		cols_params.batch_format = ADNN_BF_HW;

		cols_params.dims[0] = inputs * kernel_size0 * kernel_size1;
		cols_params.dims[1] = height_out * width_out * batch_sz; // + BIAS TO DO: SEPARATE Bias

		aDNNTensor & cols_slot = createSlot(getBotNm() + ADNN_WIN_COLS_FWD_NM, cols_params);

		if (getDebugLevel() & 1)
		{
			cloneSlot(getBotNm() + ADNN_WIN_COLS_FWD_NM + ADNN_VERIFY_NM, cols_slot);
		}
		int cols_stride = (int)cols_slot.getStride(aDNN_TENSOR_WIDTH);


		int n_ins0 = 1; // number of inputs each a from different stack along dim 0
		int n_ins1 = 1; // number of inputs each a from different stack along dim 1
		int n_ins = n_ins0 * n_ins1; // number of inputs each a from different stack
		int n_outs = 1; // n outputs per a single input: major parameter
		int n_out_pix_horiz = 2; // n of output px horix per wk-item: major parameter
		int n_out_pix_vert = 2; // n of output px horix per wk-item: major parameter
		int n_in_pix_horiz = n_out_pix_horiz; // n of input pix per wk_item
		int n_in_pix_vert = n_out_pix_vert; // n of input pix per wk_item
		int n_v_proc0 = (width_out + n_out_pix_horiz - 1) / n_out_pix_horiz;
		int n_v_proc1 = (height_out + n_out_pix_vert - 1) / n_out_pix_vert;
		int ocl_group_sz0 = 16;
		int ocl_group_sz1 = 16;
		int ocl_group_sz2 = 1;

		in_main_loop_ = inputs;



		for (int proc0 = ocl_group_sz0 / 2; n_v_proc0 <= proc0 && proc0 > 1; proc0 /= 2)
		{
			n_ins0 *= 2;
		}
		for (int proc1 = ocl_group_sz1 / 2; n_v_proc1 <= proc1 && proc1 > 1; proc1 /= 2)
		{
			n_ins1 *= 2;
		}

		n_ins = n_ins0 * n_ins1;
		if (n_ins > batch_sz)
		{
			ocl_group_sz1 /= 2;
			n_ins1 = 1;
			for (int proc1 = ocl_group_sz1 / 2; n_v_proc1 <= proc1 && proc1 > 1; proc1 /= 2)
			{
				n_ins1 *= 2;
			}
			n_ins = n_ins0 * n_ins1;
		}

		if (n_ins > batch_sz)
		{
			ocl_group_sz0 /= 2;
			n_ins0 = 1;
			for (int proc0 = ocl_group_sz0 / 2; n_v_proc0 <= proc0 && proc0 > 1; proc0 /= 2)
			{
				n_ins0 *= 2;
			}
			n_ins = n_ins0 * n_ins1;
		}


		int batch_aligned = 0;
		if ((batch_sz / n_ins) * n_ins == batch_sz)
		{
			batch_aligned = 1;
		}


		int big = 0;
		if (ocl_group_sz0 * n_in_pix_horiz < width || ocl_group_sz1 * n_in_pix_vert < height)
		{
			big = 1;
		}
		int n_procs0 = ocl_group_sz0 / n_ins0;
		int n_procs1 = ocl_group_sz1 / n_ins1;

		int n_stack_blocks = ((batch_sz + n_ins - 1) / n_ins);

		// global work size
		int gbl0 = n_ins0 * ((n_v_proc0 + n_procs0 - 1) / (n_procs0)) *n_procs0;
		int gbl1 = n_ins1 * ((n_v_proc1 + n_procs1 - 1) / (n_procs1)) *n_procs1;
		int gbl2 = inputs * n_stack_blocks;

		int in_sz0 = (n_procs0 * n_out_pix_horiz) * stride0 + kernel_size0 - 2 * pad0;
		int in_sz1 = (n_procs1 * n_out_pix_vert) * stride1 + kernel_size1 - 2 * pad1;

		int aligned_out = 1;

		if (gbl0 != n_ins0 * (width_out / n_out_pix_horiz) || gbl1 != n_ins1 * (height_out / n_out_pix_vert))
		{
			aligned_out = 0;
		}

		std::string comp_options =
			std::string("-D ADNN_GRP_SZ=") + std::to_string((long long)ocl_group_sz0 * ocl_group_sz1 * ocl_group_sz2)
			+ std::string(" -D ADNN_GRP_SZ0=") + std::to_string((long long)ocl_group_sz0)
			+ std::string(" -D ADNN_GRP_SZ1=") + std::to_string((long long)ocl_group_sz1)
			+ std::string(" -D ADNN_GRP_SZ2=") + std::to_string((long long)ocl_group_sz2)
			+ std::string(" -D ADNN_N_IN_CHNLS=") + std::to_string((long long)inputs)
			+ std::string(" -D ADNN_IN_WIDTH=") + std::to_string((long long)width)
			+ std::string(" -D ADNN_IN_HEIGHT=") + std::to_string((long long)height)
			+ std::string(" -D ADNN_IN_STRIDE=") + std::to_string((long long)bot_stride)
			+ std::string(" -D ADNN_IN_CHNL_STRIDE=") + std::to_string((long long)bot_channel_stride)
			+ std::string(" -D ADNN_IN_BATCH_STRIDE=") + std::to_string((long long)bot_batch_stride)
			+ std::string(" -D ADNN_LCL_N_IN_CHNLS=") + std::to_string((long long)(n_ins))
			+ std::string(" -D ADNN_LCL_N_OUT_CHNLS=") + std::to_string((long long)n_outs)
			+ std::string(" -D ADNN_BATCH_SZ=") + std::to_string((long long)batch_sz)
			+ std::string(" -D ADNN_FLTR_SZ0=") + std::to_string((long long)kernel_size0)
			+ std::string(" -D ADNN_FLTR_PAD_SZ0=") + std::to_string((long long)pad0)
			+ std::string(" -D ADNN_FLTR_STRIDE0=") + std::to_string((long long)stride0)
			+ std::string(" -D ADNN_FLTR_SZ1=") + std::to_string((long long)kernel_size1)
			+ std::string(" -D ADNN_FLTR_PAD_SZ1=") + std::to_string((long long)pad1)
			+ std::string(" -D ADNN_FLTR_STRIDE1=") + std::to_string((long long)stride1)
			+ std::string(" -D ADNN_OUT_WIDTH=") + std::to_string((long long)width_out)
			+ std::string(" -D ADNN_OUT_HEIGHT=") + std::to_string((long long)height_out)
			+ std::string(" -D ADNN_OUT_STRIDE=") + std::to_string((long long)cols_stride)
			+ std::string(" -D ADNN_N_OUT_PIX_SZ0=") + std::to_string((long long)n_out_pix_horiz)
			+ std::string(" -D ADNN_N_OUT_PIX_SZ1=") + std::to_string((long long)n_out_pix_vert)
			+ std::string(" -D ADNN_N_IN_PIX_SZ0=") + std::to_string((long long)n_in_pix_horiz)         // size of output processing group in 0 dim
			+ std::string(" -D ADNN_N_IN_PIX_SZ1=") + std::to_string((long long)n_in_pix_vert)         // size of output processing group in 1 dim
			+ std::string(" -D ADNN_N_STACKS=") + std::to_string((long long)n_stack_blocks)          // n of separate data stacks
			+ std::string(" -D ADNN_N_PROCS0=") + std::to_string((long long)n_procs0)         // n of processors per stack
			+ std::string(" -D ADNN_N_PROCS1=") + std::to_string((long long)n_procs1)         // n of processors per stack
			+ std::string(" -D ADNN_ALIGNED=") + std::to_string((long long)aligned_out)		//	dimesions aligned
			+ std::string(" -D ADNN_BATCH_ALIGNED=") + std::to_string((long long)batch_aligned)      // batch is multiple of n_ins
			+ std::string(" -D ADNN_IN_SZ0=") + std::to_string((long long)in_sz0)			// horizontal read dim 0
			+ std::string(" -D ADNN_IN_SZ1=") + std::to_string((long long)in_sz1)			// vertical read dim 1

			+ std::string(" -D ADNN_BIG=") + std::to_string((long long)big)		//	resolution > 32 x 32

			+ getGenericCompOptions()
			;

		std::string kernel_file = "aDNNConv_img2col.cl";
		std::string kernel_name = "aDNNConv_img2col";

		// execution setup


		std::vector<size_t> l_wk;
		l_wk.push_back(ocl_group_sz0);
		l_wk.push_back(ocl_group_sz1);
		l_wk.push_back(ocl_group_sz2);


		std::vector<size_t> g_wk;




		g_wk.push_back(gbl0);
		g_wk.push_back(gbl1);
		g_wk.push_back(gbl2);


		CDNN_OCL_kern_exe kern_exe(this, kernel_name, kernel_file, comp_options, 0, &g_wk, &l_wk, 0);

		kern_exe.Construct();

		ocl_fwd_execs_.push_back(kern_exe);


		return(ret);
	}


	int aDNNodeConv::ConstructGT32_NCHW(void)
	{
		int ret = 0;

		int pad = getPad();
		int stride = getKernelStride();
		int kernel_size = getKernelSz();

		const aDNNTensor & bot = getBotFwd();
		const aDNNTensor & top = getTopFwd();
		const aDNNTensor & wei = getBotWeightsFwd();

		int width = (int)bot.getDim(aDNN_TENSOR_WIDTH);
		int height = (int)bot.getDim(aDNN_TENSOR_HEIGHT);
		int inputs = (int)bot.getDim(aDNN_TENSOR_DEPTH);
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
		int vis_height = (int)height;
		int vis_width = (int)width;

		int weights_stride = (int)wei.getStride(aDNN_TENSOR_WIDTH);

		int ocl_group_sz0 = 8;
		int ocl_group_sz1 = 8;
		int ocl_group_lg2sz1 = (int)ceil(log((double)ocl_group_sz1) / log(2.));
		int ocl_group_lg2sz0 = (int)ceil(log((double)ocl_group_sz0) / log(2.));

		int n_out_pix_horiz = (width_out <= ocl_group_sz0) ? 1 : (width_out < 4 * ocl_group_sz0 || stride > 1) ? 2 : 4;
		int n_out_pix_vert = (height_out <= ocl_group_sz1) ? 1 : 2; // (height_out <= 192) ? 2 : 4;

		int aligned = 0;
		if ((top_stride / n_out_pix_horiz) * n_out_pix_horiz == top_stride && (top_height_stride / n_out_pix_vert) * n_out_pix_vert == top_height_stride)
		{
			aligned = 1;
		}



		int n_outs = ((outputs & 1) == 1) ? 1 : (kernel_size == 3) && ((outputs / 4) * 4 == outputs) ? 4 : 2; // (n_out_pix_horiz >= 4) ? 1 : 2;

		int n_outputs = (int)outputs;
		n_outputs /= n_outs;

		std::string comp_options =
			std::string("-D ADNN_CONV_KERNEL_SZ=") + std::to_string((long long)kernel_size)
			+ std::string(" -D ADNN_CONV_N_OUTPUTS=") + std::to_string((long long)n_outputs)
			+ std::string(" -D ADNN_CONV_N_CHANNELS=") + std::to_string((long long)inputs)
			+ std::string(" -D ADNN_CONV_PAD=") + std::to_string((long long)pad)
			+ std::string(" -D ADNN_CONV_STRIDE=") + std::to_string((long long)stride)
			+ std::string(" -D ADNN_CONV_N_HORIZ_OUT_PIX=") + std::to_string((long long)n_out_pix_horiz)
			+ std::string(" -D ADNN_CONV_N_VERT_OUT_PIX=") + std::to_string((long long)n_out_pix_vert)
			+ std::string(" -D ADNN_CONV_GROUP_SZ0=") + std::to_string((long long)ocl_group_sz0)
			+ std::string(" -D ADNN_CONV_GROUP_SZ1=") + std::to_string((long long)ocl_group_sz1)
			+ std::string(" -D ADNN_CONV_GROUP_LG2SZ0=") + std::to_string((long long)ocl_group_lg2sz0)
			+ std::string(" -D ADNN_CONV_GROUP_LG2SZ1=") + std::to_string((long long)ocl_group_lg2sz1)
			+ std::string(" -D ADNN_CONV_BOT_BATCH_STRIDE=") + std::to_string((long long)bot_batch_stride)
			+ std::string(" -D ADNN_CONV_BOT_CHANNEL_STRIDE=") + std::to_string((long long)bot_channel_stride)
			+ std::string(" -D ADNN_CONV_BOT_STRIDE=") + std::to_string((long long)bot_stride)
			+ std::string(" -D ADNN_CONV_TOP_BATCH_STRIDE=") + std::to_string((long long)top_batch_stride)
			+ std::string(" -D ADNN_CONV_TOP_CHANNEL_STRIDE=") + std::to_string((long long)top_channel_stride)
			+ std::string(" -D ADNN_CONV_TOP_STRIDE=") + std::to_string((long long)top_stride)
			+ std::string(" -D ADNN_CONV_BOT_VIS_WIDTH=") + std::to_string((long long)vis_width)
			+ std::string(" -D ADNN_CONV_BOT_VIS_HEIGHT=") + std::to_string((long long)vis_height)
			+ std::string(" -D ADNN_CONV_WEIGHTS_STRIDE=") + std::to_string((long long)weights_stride)
			+ std::string(" -D ADNN_CONV_TOP_WIDTH=") + std::to_string((long long)width_out)
			+ std::string(" -D ADNN_CONV_TOP_HEIGHT=") + std::to_string((long long)height_out)
			+ std::string(" -D ADNN_CONV_N_OUTS=") + std::to_string((long long)n_outs)
			+ std::string(" -D ADNN_ALIGNED=") + std::to_string((long long)aligned)		//	weights stride

			+ getGenericCompOptions()
			;

		int do_bias = 0;
		comp_options += std::string(" -D ADNN_CONV_BIAS=") + std::to_string((long long)do_bias);


		std::string kernel_file = "aDNNConv_GT32_NCHW.cl";
		std::string kernel_name = "aDNNConv_GT32_NCHW";

		// execution setup

		int i_n_group_horiz = (width_out + ocl_group_sz0 * n_out_pix_horiz - 1) / (ocl_group_sz0 * n_out_pix_horiz);
		int i_n_group_vert = (height_out + ocl_group_sz1 * n_out_pix_vert - 1) / (ocl_group_sz1 * n_out_pix_vert);


		std::vector<size_t> l_wk;
		l_wk.push_back(ocl_group_sz0);
		l_wk.push_back(ocl_group_sz1);
		l_wk.push_back(1);


		std::vector<size_t> g_wk;
		g_wk.push_back(i_n_group_horiz * l_wk[0]);
		g_wk.push_back(i_n_group_vert * l_wk[1]);
		g_wk.push_back(top.getDim(aDNN_TENSOR_BATCH) * n_outputs);



		CDNN_OCL_kern_exe kern_exe(this, kernel_name, kernel_file, comp_options, 0, &g_wk, &l_wk, 0);

		kern_exe.Construct();

		ocl_fwd_execs_.push_back(kern_exe);

		return (ret);
	}

	int aDNNodeConv::Construct_NCHW_N3(void)
	{
		//ADNN_GRP_SZ0              group size in dim 0
		//ADNN_GRP_SZ1				group size in dim 1
		//ADNN_GRP_LG2SZ0           log2 group size in dim 0
		//ADNN_GRP_LG2SZ1           log2 group size in dim 1
		//ADNN_GRP_LG2SZ2           log2 group size in dim 2
		//ADNN_GRP_SZ               n of wk-item in the group
		//ADNN_N_IN_CHNLS			total number of input channels
		//ADNN_LCL_N_IN_CHNLS		n of localy kept input channels
		//ADNN_IN_WIDTH				input width in NCHW layout
		//ADNN_IN_HEIGHT			input height stride in NCHW layout
		//ADNN_IN_STRIDE			input stride in NCHW layout
		//ADNN_IN_CHNL_STRIDE       input channel stride in NCHW layout
		//ADNN_IN_BATCH_STRIDE      input batch stride in NCHW layout
		//ADNN_BATCH_SZ		        batch szie
		//ADNN_N_IN_PIX_SZ0        n input pixels per wk item in 0 dim
		//ADNN_N_IN_PIX_SZ1		n input pexels per wk item in 1 dim
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
		//ADNN_WEIGHTS_STRIDE			weights stride
		//ADNN_WEI_SZ               size of weight buffer
		//ADNN_N_STACKS           n of separate data stacks
		//ADNN_N_PROCS1           n of processors per stack 1 dim
		//ADNN_N_PROCS0           n of processors per stack 0 dim
		//ADNN_ALIGNED             dimesions aligned to 2
		//ADNN_BATCH_ALIGNED      batch is multiple of n_ins
		//ADNN_OUT_ALINED         outputs is multiple of n_outs

		int ret = 0;

		// edge 0, dim 0
		int pad0 = getPad();
		int stride0 = getKernelStride();
		int kernel_size0 = getKernelSz();
		// edge 0, dim 1
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

		int n_ins0 = 1; // number of inputs each a from different stack along dim 0
		int n_ins1 = 1; // number of inputs each a from different stack along dim 1
		int n_ins = n_ins0 * n_ins1; // number of inputs each a from different stack
		int n_outs = (kernel_size0 == 3 && width_out >= 64 && height_out >= 64) ? 14 : (kernel_size0 == 3) ? 8 : 12; // n outputs per a single input: major parameter
		int n_out_pix_horiz = 2; // n of output px horix per wk-item: major parameter
		int n_out_pix_vert = 2; // n of output px horix per wk-item: major parameter
		int n_in_pix_horiz = n_out_pix_horiz; // n of input pix per wk_item
		int n_in_pix_vert = n_out_pix_vert; // n of input pix per wk_item
		int n_v_proc0 = (width_out + n_out_pix_horiz - 1) / n_out_pix_horiz;
		int n_v_proc1 = (height_out + n_out_pix_vert - 1) / n_out_pix_vert;
		int ocl_group_sz0 = 16;
		int ocl_group_sz1 = 16;
		int ocl_group_sz2 = 1;

		in_main_loop_ = inputs;

		for (int proc0 = ocl_group_sz0 / 2; n_v_proc0 <= proc0 && proc0 > 1; proc0 /= 2)
		{
				n_ins0 *= 2;
		}
		for (int proc1 = ocl_group_sz1 / 2; n_v_proc1 <= proc1 && proc1 > 1; proc1 /= 2)
		{
				n_ins1 *= 2;
		}

		n_ins = n_ins0 * n_ins1;
		if (n_ins > batch_sz)
		{
			ocl_group_sz1 /= 2;
			n_ins1 = 1;
			for (int proc1 = ocl_group_sz1 / 2; n_v_proc1 <= proc1 && proc1 > 1; proc1 /= 2)
			{
				n_ins1 *= 2;
			}
			n_ins = n_ins0 * n_ins1;
		}

		if (n_ins > batch_sz)
		{
			ocl_group_sz0 /= 2;
			n_ins0 = 1;
			for (int proc0 = ocl_group_sz0 / 2; n_v_proc0 <= proc0 && proc0 > 1; proc0 /= 2)
			{
				n_ins0 *= 2;
			}
			n_ins = n_ins0 * n_ins1;
		}


		int batch_aligned = 0;
		if ((batch_sz / n_ins) * n_ins == batch_sz)
		{
			batch_aligned = 1;
		}

		int out_aligned = 0;
		if ((outputs / n_outs) * n_outs == outputs)
		{
			out_aligned = 1;
		}

		int big = 0;
		if (ocl_group_sz0 * n_in_pix_horiz < width || ocl_group_sz1 * n_in_pix_vert < height)
		{
			big = 1;
		}
		int n_procs0 = ocl_group_sz0 / n_ins0;
		int n_procs1 = ocl_group_sz1 / n_ins1;

		int n_out_blocks = ((outputs + n_outs - 1) / n_outs);
		int n_stack_blocks = ((batch_sz + n_ins - 1) / n_ins);

// global work size
		int gbl0 = n_ins0 * ((n_v_proc0 + n_procs0 - 1) / (n_procs0)) *n_procs0;
		int gbl1 = n_ins1 * ((n_v_proc1 + n_procs1 - 1) / (n_procs1)) *n_procs1;
		int gbl2 = n_out_blocks * n_stack_blocks;


		int aligned_out = 1;

		if (gbl0 != n_ins0 * (width_out / n_out_pix_horiz) || gbl1 != n_ins1 * (height_out / n_out_pix_vert))
		{
			aligned_out = 0;
		}

		std::string comp_options =
			std::string("-D ADNN_GRP_SZ=") + std::to_string((long long)ocl_group_sz0 * ocl_group_sz1 * ocl_group_sz2)
			+ std::string(" -D ADNN_GRP_SZ0=") + std::to_string((long long)ocl_group_sz0)
			+ std::string(" -D ADNN_GRP_SZ1=") + std::to_string((long long)ocl_group_sz1)
			+ std::string(" -D ADNN_GRP_SZ2=") + std::to_string((long long)ocl_group_sz2)
			+ std::string(" -D ADNN_N_IN_CHNLS=") + std::to_string((long long)inputs)
			+ std::string(" -D ADNN_IN_WIDTH=") + std::to_string((long long)width)
			+ std::string(" -D ADNN_IN_HEIGHT=") + std::to_string((long long)height)
			+ std::string(" -D ADNN_IN_STRIDE=") + std::to_string((long long)bot_stride)
			+ std::string(" -D ADNN_IN_CHNL_STRIDE=") + std::to_string((long long)bot_channel_stride)
			+ std::string(" -D ADNN_IN_BATCH_STRIDE=") + std::to_string((long long)bot_batch_stride)
			+ std::string(" -D ADNN_LCL_N_IN_CHNLS=") + std::to_string((long long)(n_ins))
			+ std::string(" -D ADNN_LCL_N_OUT_CHNLS=") + std::to_string((long long)n_outs)
			+ std::string(" -D ADNN_BATCH_SZ=") + std::to_string((long long)batch_sz)
			+ std::string(" -D ADNN_FLTR_SZ0=") + std::to_string((long long)kernel_size0)
			+ std::string(" -D ADNN_FLTR_PAD_SZ0=") + std::to_string((long long)pad0)
			+ std::string(" -D ADNN_FLTR_STRIDE0=") + std::to_string((long long)stride0)
			+ std::string(" -D ADNN_FLTR_SZ1=") + std::to_string((long long)kernel_size1)
			+ std::string(" -D ADNN_FLTR_PAD_SZ1=") + std::to_string((long long)pad1)
			+ std::string(" -D ADNN_FLTR_STRIDE1=") + std::to_string((long long)stride1)
			+ std::string(" -D ADNN_WEI_SZ=") + std::to_string((long long)weights_hight*weights_stride)
			+ std::string(" -D ADNN_OUT_WIDTH=") + std::to_string((long long)width_out)
			+ std::string(" -D ADNN_OUT_HEIGHT=") + std::to_string((long long)height_out)
			+ std::string(" -D ADNN_OUT_STRIDE=") + std::to_string((long long)top_stride)
			+ std::string(" -D ADNN_OUT_CHNL_STRIDE=") + std::to_string((long long)top_channel_stride)
			+ std::string(" -D ADNN_OUT_BATCH_STRIDE=") + std::to_string((long long)top_batch_stride)
			+ std::string(" -D ADNN_N_OUT_PIX_SZ0=") + std::to_string((long long)n_out_pix_horiz)
			+ std::string(" -D ADNN_N_OUT_PIX_SZ1=") + std::to_string((long long)n_out_pix_vert)
			+ std::string(" -D ADNN_N_IN_PIX_SZ0=") + std::to_string((long long)n_in_pix_horiz)         // size of output processing group in 0 dim
			+ std::string(" -D ADNN_N_IN_PIX_SZ1=") + std::to_string((long long)n_in_pix_vert)         // size of output processing group in 1 dim
			+ std::string(" -D ADNN_N_OUT_CHNLS=") + std::to_string((long long)outputs)			//total number of output channels
			+ std::string(" -D ADNN_LCL_N_OUT_CHNLS=") + std::to_string((long long)n_outs)		//n of localy kept output channels
			+ std::string(" -D ADNN_WEIGHTS_STRIDE=") + std::to_string((long long)weights_stride)		//	weights stride
			+ std::string(" -D ADNN_N_STACKS=") + std::to_string((long long)n_stack_blocks)          // n of separate data stacks
			+ std::string(" -D ADNN_N_PROCS0=") + std::to_string((long long)n_procs0)         // n of processors per stack
			+ std::string(" -D ADNN_N_PROCS1=") + std::to_string((long long)n_procs1)         // n of processors per stack
			+ std::string(" -D ADNN_ALIGNED=") + std::to_string((long long)aligned_out)		//	dimesions aligned
			+ std::string(" -D ADNN_BATCH_ALIGNED=") + std::to_string((long long)batch_aligned)      // batch is multiple of n_ins
			+ std::string(" -D ADNN_OUT_ALINED=") + std::to_string((long long)out_aligned)        // outputs is multiple of n_outs

			+ std::string(" -D ADNN_BIG=") + std::to_string((long long)big)		//	resolution > 32 x 32

			+ getGenericCompOptions()
			;

		std::string kernel_file = "aDNNConv_NCHW_N3.cl";
		std::string kernel_name = "aDNNConv_NCHW_N3";

		// execution setup


		std::vector<size_t> l_wk;
		l_wk.push_back(ocl_group_sz0);
		l_wk.push_back(ocl_group_sz1);
		l_wk.push_back(ocl_group_sz2);


		std::vector<size_t> g_wk;




		g_wk.push_back(gbl0);
		g_wk.push_back(gbl1);
		g_wk.push_back(gbl2);


		CDNN_OCL_kern_exe kern_exe(this, kernel_name, kernel_file, comp_options, 0, &g_wk, &l_wk, 0);

		kern_exe.Construct();

		ocl_fwd_execs_.push_back(kern_exe);


		return(ret);
	}

	int aDNNodeConv::ConstructLE32_NCHW(void)
	{

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
		//ADNN_N_READ_PROC_SZ0         size of read processors (dim 0)= size of input tile
		//ADNN_N_READ_PROC_SZ1         size of read processors (dim 1) = size of input tile
		//ADNN_N_READ_PROC_LOG2SZ0     log2 size of read processors (dim 0)= size of input tile
		//ADNN_N_READ_PROC_LOG2SZ1     log2 size of read processors (dim 1) = size of input tile
		//ADNN_N_OUT_TILES0         n output tiles (dim 0)= n input tiles
		//ADNN_N_OUT_TILES1         n output tiles (dim 1) = n input tiles
		//ADNN_LOG2N_OUT_TILES0         log2 n output tiles (dim 0)= log2 n input tile
		//ADNN_LOG2N_OUT_TILES1         log2 n output tiles (dim 1) = log2 n input tile
		//ADNN_LCL_WEIGHTS          weights are in local memory


		int ret = 0;

// edge 0, dim 0
		int pad0 = getPad();
		int stride0 = getKernelStride();
		int kernel_size0 = getKernelSz();
// edge 0, dim 1
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


		int ocl_group_sz0 = 256;
		int ocl_group_sz1 = 1;
		int ocl_group_lg2sz1 = (int)ceil(log((double)ocl_group_sz1) / log(2.));
		int ocl_group_lg2sz0 = (int)ceil(log((double)ocl_group_sz0) / log(2.));


// input processing for images smaller or eq 32x32
// read one or more (4) entire image by a set of dedicated work-items ( = processor stride).
		int in_size = width * height;
		int n_ins = (in_size > (ocl_group_sz0 / 2)) ? 1 : (in_size > (ocl_group_sz0 / 4)) ? 2 : 4;
		n_ins = std::min(n_ins, inputs); // TT: not declared, should be in stdlib.h --> in <algorithms> so prefixed with std::
		int log2_n_ins = (int)ceil(log((double)n_ins) / log(2.));
		int in_main_loop = (inputs + n_ins - 1) / n_ins;

		in_main_loop_ = in_main_loop;

// tiling and output processing
		int big = (in_size > (32 * 32)) ? 1 : 0;

		int input_tile_sz0 = 32;
		int input_tile_sz1 = 32;

		int n_out_pix_horiz = (width_out < 2) ? 1 : (width_out < 4) ? 2 : 4;
		int n_out_pix_vert =  (height_out < 2) ? 1 : (height_out < 4) ? 2 : 4;

// here is input processing parameters for images larger than 32x32
// read a tile of 32x32 by a group.
// TO DO: it's [ossible to read more than 1 input
		int n_read_proc0 = 16;
		int n_read_proc1 = 16;

		assert(n_read_proc0*n_read_proc1 == ocl_group_sz0);

		int log2n_read_proc0 = (int)ceil(log((double)n_read_proc0) / log(2.));
		int log2n_read_proc1 = (int)ceil(log((double)n_read_proc1) / log(2.));

		int input_tile_log2sz0 = (int)ceil(log((double)input_tile_sz0) / log(2.));
		int input_tile_log2sz1 = (int)ceil(log((double)input_tile_sz1) / log(2.));

		int input_tile = input_tile_sz0 * input_tile_sz1;
		int n_input_tiles0 = (width + input_tile_sz0 - 1) / input_tile_sz0;
		int n_input_tiles1 = (height + input_tile_sz1 - 1) / input_tile_sz1;
		int log2n_input_tiles0 = (int)ceil(log((double)n_input_tiles0) / log(2.));
		int log2n_input_tiles1 = (int)ceil(log((double)n_input_tiles1) / log(2.));
		int n_input_tiles = n_input_tiles0 * n_input_tiles1;



		int aligned = 0;
		if ((top_stride / n_out_pix_horiz) * n_out_pix_horiz == top_stride && (top_height_stride / n_out_pix_vert) * n_out_pix_vert == top_height_stride)
		{
			aligned = 1;
		}


		int log2n_out_pix_horiz = (int)ceil(log((double)n_out_pix_horiz) / log(2.));
		int log2n_out_pix_vert = (int)ceil(log((double)n_out_pix_vert) / log(2.));

		int log2_width_out = (int)ceil(log((double)((big) ? input_tile_sz0 : width_out)) / log(2.));
		int log2_height_out = (int)ceil(log((double)((big) ? input_tile_sz1 : height_out)) / log(2.));
		int n_procs_x = 16;
		int n_procs_y = 16;
		int log2n_procs_x = (int)ceil(log((double)n_procs_x) / log(2.));
		int log2n_procs_y = (int)ceil(log((double)n_procs_y) / log(2.));
		int log2n_tiles_x = (log2n_procs_x - (log2_width_out - log2n_out_pix_horiz));
		int log2n_tiles_y = (log2n_procs_y - (log2_height_out - log2n_out_pix_vert));
		int n_tiles_x = (1 << log2n_tiles_x);
		int n_tiles_y = (1 << log2n_tiles_y);
		// outs total per group
		int n_outs = n_tiles_x * n_tiles_y;

		int n_outputs = (int)(outputs + n_outs - 1) / n_outs;

		int lcl_stride = ((big) ? input_tile_sz0 : width_out) + pad0 * 2;
		int lcl_height = ((big) ? input_tile_sz1 : height_out) + pad1 * 2;
		int lcl_sz = lcl_stride * lcl_height * n_ins;
		int weights_sz = n_outs * n_ins* kernel_size0 * kernel_size1;
// TO DO: WHY?
		int lcl_weights = (lcl_sz + weights_sz < (1 << 12) && !(kernel_size0 == 3 && (width <= 16 || height <= 16))) ? 1 : 0;

		std::string comp_options =
			std::string("-D ADNN_GRP_SZ=") + std::to_string((long long)ocl_group_sz0)
			+ std::string(" -D ADNN_GRP_SZ0=") + std::to_string((long long)ocl_group_sz0)
			+ std::string(" -D ADNN_GRP_SZ1=") + std::to_string((long long)ocl_group_sz1)
			+ std::string(" -D ADNN_GRP_LG2SZ0=") + std::to_string((long long)ocl_group_lg2sz0)
			+ std::string(" -D ADNN_GRP_LG2SZ1=") + std::to_string((long long)ocl_group_lg2sz1)
			+ std::string(" -D ADNN_N_IN_CHNLS=") + std::to_string((long long)inputs)
			+ std::string(" -D ADNN_IN_CHNL_SZ=") + std::to_string((long long)(height* bot_stride))
			+ std::string(" -D ADNN_IN_WIDTH=") + std::to_string((long long)width)
			+ std::string(" -D ADNN_IN_HEIGHT=") + std::to_string((long long)height)
			+ std::string(" -D ADNN_IN_STRIDE=") + std::to_string((long long)bot_stride)
			+ std::string(" -D ADNN_IN_CHNL_STRIDE=") + std::to_string((long long)bot_channel_stride)
			+ std::string(" -D ADNN_IN_BATCH_STRIDE=") + std::to_string((long long)bot_batch_stride)
			+ std::string(" -D ADNN_LCL_N_IN_CHNLS=") + std::to_string((long long)(n_ins))
			+ std::string(" -D ADNN_LCL_LOG2N_IN_CHNLS=") + std::to_string((long long)(log2_n_ins))
			+ std::string(" -D ADNN_LCL_N_OUT_CHNLS=") + std::to_string((long long)n_outs)
			+ std::string(" -D ADNN_BATCH_SZ=") + std::to_string((long long)batch_sz)
			+ std::string(" -D ADNN_FLTR_SZ0=") + std::to_string((long long)kernel_size0)
			+ std::string(" -D ADNN_FLTR_PAD_SZ0=") + std::to_string((long long)pad0)
			+ std::string(" -D ADNN_FLTR_STRIDE0=") + std::to_string((long long)stride0)
			+ std::string(" -D ADNN_FLTR_SZ1=") + std::to_string((long long)kernel_size1)
			+ std::string(" -D ADNN_FLTR_PAD_SZ1=") + std::to_string((long long)pad1)
			+ std::string(" -D ADNN_FLTR_STRIDE1=") + std::to_string((long long)stride1)
			+ std::string(" -D ADNN_IN_CHNL_LOOP=") + std::to_string((long long)in_main_loop)
			+ std::string(" -D ADNN_WEI_SZ=") + std::to_string((long long)weights_hight*weights_stride)
			+ std::string(" -D ADNN_OUT_WIDTH=") + std::to_string((long long)width_out)
			+ std::string(" -D ADNN_OUT_HEIGHT=") + std::to_string((long long)height_out)
			+ std::string(" -D ADNN_OUT_STRIDE=") + std::to_string((long long)top_stride)
			+ std::string(" -D ADNN_OUT_CHNL_STRIDE=") + std::to_string((long long)top_channel_stride)
			+ std::string(" -D ADNN_OUT_BATCH_STRIDE=") + std::to_string((long long)top_batch_stride)
			+ std::string(" -D ADNN_N_OUT_PIX_SZ0=") + std::to_string((long long)n_out_pix_horiz)
			+ std::string(" -D ADNN_N_OUT_PIX_SZ1=") + std::to_string((long long)n_out_pix_vert)
			+ std::string(" -D ADNN_LOG2N_OUT_PIX_SZ0=") + std::to_string((long long)log2n_out_pix_horiz)    // log2 of n output pixel per wk item in 0 dim
			+ std::string(" -D ADNN_LOG2N_OUT_PIX_SZ1=") + std::to_string((long long)log2n_out_pix_vert)	// log2 of n output pexels per wk item in 1 dim
			+ std::string(" -D ADNN_OUT_PROC_SZ0=") + std::to_string((long long)n_procs_x)         // size of output processing group in 0 dim
			+ std::string(" -D ADNN_OUT_PROC_SZ1=") + std::to_string((long long)n_procs_y)         // size of output processing group in 1 dim
			+ std::string(" -D ADNN_OUT_PROC_LOG2SZ0=") + std::to_string((long long)log2n_procs_x)     // log2 of size of output processing group in 0 dim
			+ std::string(" -D ADNN_OUT_PROC_LOG2SZ1=") + std::to_string((long long)log2n_procs_y)     // log2 of size of output processing group in 1 dim
			+ std::string(" -D ADNN_OUT_N_TILEPROC_SZ0=") + std::to_string((long long)n_tiles_x)      //   size of output tile processing group in 0 dim, 1 tile per 1 outpur channel.
			+ std::string(" -D ADNN_OUT_N_TILEPROC_SZ1=") + std::to_string((long long)n_tiles_y)       //  size of output tile processing group in 1 dim
			+ std::string(" -D ADNN_OUT_N_TILEPROC_LOG2SZ0=") + std::to_string((long long)log2n_tiles_x)    // log2 of size of tile output processing group in 0 dim,  ADNN_OUT_TILEPROC_LOG2SZ0 + ADNN_LOG2N_OUT_PIX_SZ0 = ADNN_OUT_PROC_LOG2SZ0
			+ std::string(" -D ADNN_OUT_N_TILEPROC_LOG2SZ1=") + std::to_string((long long)log2n_tiles_y)    // log2 of size of til output processing group in 1 dim
			+ std::string(" -D ADNN_N_OUT_CHNLS=") + std::to_string((long long)outputs)			//total number of output channels
			+ std::string(" -D ADNN_LCL_N_OUT_CHNLS=") + std::to_string((long long)n_outs)		//n of localy kept output channels
			+ std::string(" -D ADNN_WEIGHTS_STRIDE=") + std::to_string((long long)weights_stride)		//	weights stride
			+ std::string(" -D ADNN_ALIGNED=") + std::to_string((long long)aligned)		//	weights stride
			+ std::string(" -D ADNN_BIG=") + std::to_string((long long)big)		//	reasolution > 32 x 32
			+ std::string(" -D ADNN_OUT_TILE_SZ0=") + std::to_string((long long)input_tile_sz0)         //size of output tile(dim 0) = size of input tile
			+ std::string(" -D ADNN_OUT_TILE_SZ1=") + std::to_string((long long)input_tile_sz1)         //size of output tile(dim 1) = size of input tile
			+ std::string(" -D ADNN_OUT_TILE_LOG2SZ0=") + std::to_string((long long)input_tile_log2sz0)       //  log2 size of output tile(dim 0) = size of input tile
			+ std::string(" -D ADNN_OUT_TILE_LOG2SZ1=") + std::to_string((long long)input_tile_log2sz1)        // log2 size of output tile(dim 1) = size of input tile
			+ std::string(" -D ADNN_N_READ_PROC_SZ0=") + std::to_string((long long)n_read_proc0)         //size of read processors(dim 0) = size of input tile
			+ std::string(" -D ADNN_N_READ_PROC_SZ1=") + std::to_string((long long)n_read_proc1)         //size of read processors(dim 1) = size of input tile
			+ std::string(" -D ADNN_N_READ_PROC_LOG2SZ0=") + std::to_string((long long)log2n_read_proc0)     //log2 size of read processors(dim 0) = size of input tile
			+ std::string(" -D ADNN_N_READ_PROC_LOG2SZ1=") + std::to_string((long long)log2n_read_proc1)     //log2 size of read processors(dim 1) = size of input tile
			+ std::string(" -D ADNN_N_OUT_TILES0=") + std::to_string((long long)n_input_tiles0)        // n output tiles(dim 0) = n input tiles
			+ std::string(" -D ADNN_N_OUT_TILES1=") + std::to_string((long long)n_input_tiles1)        // n output tiles(dim 1) = n input tiles
			+ std::string(" -D ADNN_LOG2N_OUT_TILES0=") + std::to_string((long long)log2n_input_tiles0)     //    log2 n output tiles(dim 0) = log2 n input tile
			+ std::string(" -D ADNN_LOG2N_OUT_TILES1=") + std::to_string((long long)log2n_input_tiles1)      //   log2 n output tiles(dim 1) = log2 n input tile
			+ std::string(" -D ADNN_LCL_WEIGHTS=") + std::to_string((long long)lcl_weights)
			+ getGenericCompOptions();



//		generic_comp_otions_ += std::string(" -cl-std=CL2.0");

		std::string kernel_file = "aDNNConv_LE32_NCHW.cl";
		std::string kernel_name = "aDNNConv_LE32_NCHW";

		// execution setup


		std::vector<size_t> l_wk;
		l_wk.push_back(ocl_group_sz0);
		l_wk.push_back(ocl_group_sz1);
		l_wk.push_back(1);


		std::vector<size_t> g_wk;
		g_wk.push_back(n_input_tiles * l_wk[0]);
		g_wk.push_back(n_outputs);
		g_wk.push_back(batch_sz);



		CDNN_OCL_kern_exe kern_exe(this, kernel_name, kernel_file, comp_options, 0, &g_wk, &l_wk, 0);

		kern_exe.Construct();

		ocl_fwd_execs_.push_back(kern_exe);

		return (ret);
	}


	int aDNNodeConv::BuildGen(void)
	{
		int ret = 0;
		// forward
		// tensor for the host verification



		// memory has to bealocated utside of the pipeline by user
		const aDNNTensor & bot = getBotFwd();
		aDNNTensor & cols = getSlot(getBotNm() + ADNN_WIN_COLS_FWD_NM);
		cols.allocTensor();
		if (getDebugLevel() & 1)
		{
			aDNNTensor & cols_vf = getSlot(getBotNm() + ADNN_WIN_COLS_FWD_NM + ADNN_VERIFY_NM);
			cols_vf.allocTensor(_CBUF_MEM_SYS_ONLY);
		}




		cl_mem bot_mem = bot.getOCLBuffer();
		cl_mem cols_mem = cols.getOCLBuffer();

		float padding_value = 0;

		// pass all arguments once
		CDNN_OCL_kern_exe & kern_exe = ocl_fwd_execs_[0];
		int n_arg = 0;
		ocl_args kern_args;
		if (bot_mem)
		{
			kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &bot_mem);
		}
		n_arg++;
		if (cols_mem)
		{
			kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &cols_mem);
		}
		n_arg++;
		kern_args[n_arg] = std::make_pair(sizeof(aDType), &padding_value);
		n_arg++;


		kern_exe.Build(kern_args);

		return(ret);

	}


	int aDNNodeConv::Build(void)
	{
		int ret = 0;
		// forward
		// tensor for the host verification
		aDNNode::Build();

		if (win_alg_)
		{
			ret = BuildFwdWin_NCHW();
			return(ret);
		}
		else if (old_)
		{
#if ADNN_GENERIC_CONV
			ret = BuildGen();


			return(ret);
#endif
		}



		// memory has to bealocated utside of the pipeline by user
		const aDNNTensor & bot = getBotFwd();
		const aDNNTensor & top = getTopFwd();
		const aDNNTensor & wei = getBotWeightsFwd();
		const aDNNTensor & Bias = getBotBiasFwd();




		cl_mem bot_mem = bot.getOCLBuffer();
		cl_mem weights_mem = wei.getOCLBuffer();
		cl_mem top_mem = top.getOCLBuffer();
		cl_mem bias_mem = Bias.getOCLBuffer();
		int inputs = (int)bot.getDim(aDNN_TENSOR_DEPTH);

		float padding_value = 0;

		// pass all arguments once
		CDNN_OCL_kern_exe & kern_exe = ocl_fwd_execs_[0];
		int n_arg = 0;
		ocl_args kern_args;
		if (bot_mem)
		{
			kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &bot_mem);
		}
		n_arg++;
		if (weights_mem)
		{
			kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &weights_mem);
		}
		n_arg++;
		if (bias_mem)
		{
			kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &bias_mem);
		}
		n_arg++;
		if (top_mem)
		{
			kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &top_mem);
		}
		n_arg++;
		kern_args[n_arg] = std::make_pair(sizeof(aDType), &padding_value);
		n_arg++;

		if (!old_)
		{
			kern_args[n_arg] = std::make_pair(sizeof(int), &in_main_loop_);
			n_arg++;

		}

		kern_exe.Build(kern_args);

		return(ret);

	}


	int aDNNodeConv::RunFwd(const adnn_node_parameters * running_params)
	{
		int ret = 0;
		if (win_alg_)
		{

			ret = RunFwdWin_NCHW(running_params);
			return(ret);
		}


		// execute through specialized object
		ocl_args additional_args;
#if ADNN_GENERIC_CONV == 0

		if (running_params)
		{
			update(*running_params);

			int n_arg = 0;
			if (getInputEdge().isDataUpdated())
			{
				cl_mem bot_mem = ((aDNNTensor &)getInputEdge().getData()).getOCLBuffer();
				getInputEdge().setDataUpdated(false);
				additional_args[0] = std::make_pair(sizeof(cl_mem), &bot_mem);
			}

			if (getInputEdge().isWeightsUpdated())
			{
				cl_mem weights_mem = ((aDNNTensor &)getInputEdge().getWeightsData()).getOCLBuffer();
				getInputEdge().setWeightsUpdated(false);
				additional_args[1] = std::make_pair(sizeof(cl_mem), &weights_mem);
			}


			if (getOutputEdge().isDataUpdated())
			{
				cl_mem top_mem = ((aDNNTensor &)getOutputEdge().getData()).getOCLBuffer();
				getOutputEdge().setDataUpdated(false);
				additional_args[2] = std::make_pair(sizeof(cl_mem), &top_mem);
			}


		}
#endif
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

#if ADNN_GENERIC_CONV

#endif
		}

		if (isPerLayerTiming())
		{
			clFinish(ocl_fwd_execs_[0].getOclQueue());
			e = mach_absolute_time();
		}
		// verify

		if (getDebugLevel() == 1)
		{
			ret = VerifyFwd();
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




	int aDNNodeConv::RunHostFwd(void)
	{
		int ret = 0;
		// get from the list of tensors referred by this node
		aDNNTensor & top_vf = getSlot(getTopNm() + ADNN_VERIFY_NM);

		aDNNTensor & bot = (aDNNTensor &)getBotFwd();
		aDNNTensor & weights = (aDNNTensor &)getBotWeightsFwd();
		aDNNTensor & bias = (aDNNTensor &)getBotBiasFwd();


		aDType padding_value = 0;        // padding value

		int bot_width = (int)bot.getDim(aDNN_TENSOR_WIDTH);
		int bot_height = (int)bot.getDim(aDNN_TENSOR_HEIGHT);
		// TO DO: check top, bot dim are equal
		int kernel_size = getKernelSz();   // kernel 1 dim size
		int pad = getPad();                // padding size
		int stride = getKernelStride();    // scale factor

		int height_out = (int)top_vf.getDim(aDNN_TENSOR_HEIGHT);
		int width_out = (int)top_vf.getDim(aDNN_TENSOR_WIDTH);
		int vis_height = bot_height;
		int vis_width = bot_width;

		int n_batchs = (int)top_vf.getDim(aDNN_TENSOR_BATCH);
		int n_outputs = (int)top_vf.getDim(aDNN_TENSOR_DEPTH);
		int n_inputs = (int)bot.getDim(aDNN_TENSOR_DEPTH);

		int top_batch_stride = (int)top_vf.getStride(aDNN_TENSOR_DEPTH);
		int top_channel_stride = (int)top_vf.getStride(aDNN_TENSOR_HEIGHT);
		int top_stride = (int)top_vf.getStride(aDNN_TENSOR_WIDTH);
		int bot_batch_stride = (int)bot.getStride(aDNN_TENSOR_DEPTH);
		int bot_channel_stride = (int)bot.getStride(aDNN_TENSOR_HEIGHT);
		int bot_stride = (int)bot.getStride(aDNN_TENSOR_WIDTH);
		int weights_stride = (int)weights.getStride(aDNN_TENSOR_WIDTH);
		aDType * bot_ptr = (aDType *)bot.accessTensor(ADNN_MEM_ACCESS_READ);			// input "tensor" - batch x channels (input images, feature maps, slices) x width x height
		aDType * top_ptr = (aDType *)top_vf.accessTensor(ADNN_MEM_ACCESS_WRITE);		// output "te4nsor"  - batch x channels (output images, feature maps, slices) x width (scaled) x height (scaled)
		aDType * weights_ptr = (aDType *)weights.accessTensor(ADNN_MEM_ACCESS_READ);    // weights n output channels x n input channels x filter size_y x filter size_x
		aDType * bias_ptr = (aDType*)bias.accessTensor(ADNN_MEM_ACCESS_READ);          // bias

		aDType * run_bot_ptr = bot_ptr;
		aDType * run_top_ptr = top_ptr;
		aDType * run_weights_ptr = weights_ptr;

		// over all batches
		for (int b = 0; b < n_batchs; b++, run_bot_ptr += bot_batch_stride, run_top_ptr += top_batch_stride)
		{
			run_weights_ptr = weights_ptr;
			// over all output channels
			for (int o = 0; o < n_outputs; o++)
			{
				// sum up convolutions
				// over output image (scaled input)
				for (int j = 0; j < height_out; j++)
				{
					for (int i = 0; i < width_out; i++)
					{
						// over all input channels
						aDType accum = 0;
						for (int c = 0; c < n_inputs; c++)
						{
							// do convolution with kernel kernel_size x kerenl_size
							// with padding - left, right, top, bottom = pad, and value = 0
							for (int k_j = 0; k_j < kernel_size; k_j++)
							{

								int in_y = (j*stride + k_j - pad);
								for (int k_i = 0; k_i < kernel_size; k_i++)
								{
									int in_x = (i*stride + k_i - pad);
									aDType data_val = padding_value;
									if (!(in_y < 0 || in_x < 0 || in_y >= vis_height || in_x >= vis_width))
									{
										int in_data_off = c*bot_channel_stride + in_y * bot_stride + in_x;
										data_val = run_bot_ptr[in_data_off];
									}

									aDType wei_val = run_weights_ptr[o*weights_stride + c*kernel_size *kernel_size + k_j*kernel_size + k_i];

									accum += data_val * wei_val;
#if 0
									if (b == 0 && c == 0 && j == 0 && i == 7)
									{
										printf("c: %f %f %f\n",
											accum,
											data_val,
											wei_val
											);
									}
#endif

								}
							}

						}

						run_top_ptr[o*top_channel_stride + j*top_stride + i] = accum + bias_ptr[o]; // + bias

					}

				}

			}
		}

		bot.commitTensor();
		top_vf.commitTensor();
		weights.commitTensor();
		bias.commitTensor();

		return(ret);
	}

	int aDNNodeConv::RunHostGenFwd(void)
	{
		int ret = 0;
		aDNNTensor & bot = (aDNNTensor &)getBotFwd();
		aDNNTensor & cols_vf = getSlot(getBotNm() + ADNN_WIN_COLS_FWD_NM + ADNN_VERIFY_NM);
		int bot_batch_stride = (int)bot.getStride(aDNN_TENSOR_DEPTH);
		int bot_channel_stride = (int)bot.getStride(aDNN_TENSOR_HEIGHT);
		int bot_stride = (int)bot.getStride(aDNN_TENSOR_WIDTH);
		int batchs = (int)bot.getDim(aDNN_TENSOR_BATCH);
		int inputs = (int)bot.getDim(aDNN_TENSOR_DEPTH);
		int bot_width = (int)bot.getDim(aDNN_TENSOR_WIDTH);
		int bot_height = (int)bot.getDim(aDNN_TENSOR_HEIGHT);
		int cols_vf_stride = (int)cols_vf.getStride(aDNN_TENSOR_WIDTH);

		int kernel_size = getKernelSz();   // kernel 1 dim size
		int pad = getPad();                // padding size
		int stride = getKernelStride();    // scale factor

		aDType * bot_ptr = (aDType *)bot.accessTensor(ADNN_MEM_ACCESS_READ);			// 
		aDType * col_vf_ptr = (aDType *)cols_vf.accessTensor(ADNN_MEM_ACCESS_WRITE);			// 

#if 1
		for (int b = 0; b < batchs; ++b)
		{
			ADNN_im2col_cpu<aDType>((const aDType*)&bot_ptr[bot_batch_stride * b], inputs,
				bot_height, bot_width, kernel_size, pad,
				stride, &col_vf_ptr[b * bot_height* bot_width], cols_vf_stride);
		}
#endif
		bot.commitTensor();
		cols_vf.commitTensor();
		{
			aDNNTensor & cols = getSlot(getBotNm() + ADNN_WIN_COLS_FWD_NM);
			aDNNTensor & cols_vf = getSlot(getBotNm() + ADNN_WIN_COLS_FWD_NM + ADNN_VERIFY_NM);
			aDType * cols_vf_ptr = (aDType *)cols_vf.accessTensor(ADNN_MEM_ACCESS_READ);			// 
			aDType * cols_ptr = (aDType *)cols.accessTensor(ADNN_MEM_ACCESS_READ);			// 
			int cols_width = (int)cols.getDim(aDNN_TENSOR_WIDTH);
			int cols_height = (int)cols.getDim(aDNN_TENSOR_HEIGHT);
			int cols_stride = (int)cols.getStride(aDNN_TENSOR_WIDTH);
			int cols_vf_stride = (int)cols_vf.getStride(aDNN_TENSOR_WIDTH);

			bool match = true;
			for (int j = 0; j < cols_height && match; ++j)
			{
				for (int i = 0; i < cols_width && match; ++i)
				{
					aDType c_val = cols_vf_ptr[j*cols_vf_stride + i];
					aDType g_val = cols_ptr[j*cols_stride + i];

					if (c_val != g_val)
					{
						printf("img2col error at %d %d %d c_val=%f g_val=%f\n",
							j, i,
							j*cols_stride + i,
							c_val, g_val
							);
						match = false;
					}

				}
			}


		}
		return(ret);
	}

	int aDNNodeConv::VerifyFwd(void)
	{
		int ret = 0;
#if ADNN_GENERIC_CONV
		ret = RunHostGenFwd();
		return(ret);
#else
		ret = RunHostFwd();
#endif
		std::string top_nm;
		// get from the list of tensors referred by this node
		aDNNTensor & top_vf = getSlot(getTopNm() + ADNN_VERIFY_NM);

		aDNNTensor & top = (aDNNTensor &)getTopFwd();

		aDType * top_ptr = (aDType *)top.accessTensor(ADNN_MEM_ACCESS_READ);

		aDType * top_vf_ptr = (aDType *)top_vf.accessTensor(ADNN_MEM_ACCESS_READ);

		int width = (int)top.getDim(aDNN_TENSOR_WIDTH);
		int height = (int)top.getDim(aDNN_TENSOR_HEIGHT);
		int outputs = (int)top.getDim(aDNN_TENSOR_DEPTH);
		int batchs = (int)top.getDim(aDNN_TENSOR_BATCH);
		int top_v_batch_stride = (int)top_vf.getStride(aDNN_TENSOR_DEPTH);
		int top_v_channel_stride = (int)top_vf.getStride(aDNN_TENSOR_HEIGHT);
		int top_v_stride = (int)top_vf.getStride(aDNN_TENSOR_WIDTH);
		int top_batch_stride = (int)top.getStride(aDNN_TENSOR_DEPTH);
		int top_channel_stride = (int)top.getStride(aDNN_TENSOR_HEIGHT);
		int top_stride = (int)top.getStride(aDNN_TENSOR_WIDTH);


		double sqr_accum = 0;
		double max_err = -std::numeric_limits<float>::min();
		int max_b = 0, max_c = 0, max_i = 0, max_j = 0;

		const double allowedEps = 4;
		for (int b = 0; b < batchs; ++b)
		{
			for (int o = 0; o < outputs; ++o)
			{
				for (int j = 0; j < height; ++j)
				{
					for (int i = 0; i < width; ++i)
					{
						aDType c_val = top_vf_ptr[b*top_v_batch_stride + o*top_v_channel_stride + j*top_v_stride + i];
						aDType g_val = top_ptr[b*top_batch_stride + o*top_channel_stride + j*top_stride + i];

						sqr_accum += (c_val - g_val) * (c_val - g_val);
						if (std::abs(c_val - g_val) > max_err)
						{
							max_err = std::abs(c_val - g_val);
							max_b = b;
							max_c = o;
							max_i = i;
							max_j = j;
						}

					}
				}
			}
		}

		sqr_accum = sqrt(sqr_accum / ((double)batchs *outputs *height *width));

		int match = 1;

		if (std::isnan(sqr_accum) || !std::isfinite(sqr_accum) || sqr_accum > 0)
		{
			std::cout << "Error in conv forward propagation: " << getName() + " " << std::fixed << std::setw(15) << std::setprecision(13) << sqr_accum <<
				" Max err: " << std::fixed << std::setw(15) << std::setprecision(13) << max_err << std::endl;


			if (sqr_accum > (1. / 1000000000))
			{
				for (int b = 0; b < batchs && match; ++b)
				{
					for (int o = 0; o < outputs && match; ++o)
					{
						for (int j = 0; j < height && match; ++j)
						{
							for (int i = 0; i < width && match; ++i)
							{
								aDType c_val = top_vf_ptr[b*top_v_batch_stride + o*top_v_channel_stride + j*top_v_stride + i];
								aDType g_val = top_ptr[b*top_batch_stride + o*top_channel_stride + j*top_stride + i];


								double err = CalculateErr(c_val, g_val);
								if (err > allowedEps || std::isnan(c_val) || std::isnan(g_val) || !std::isfinite(c_val) || !std::isfinite(g_val))
								{
									std::cout << "Difference in conv forward propagation: " << getName() + " " << err << " too large at " << b << "," << o << ", " << j << "," << i <<
										" c_v = " << std::fixed << std::setw(11) << std::setprecision(9) << c_val <<
										" vs g_val = " << std::fixed << std::setw(11) << std::setprecision(9) << g_val << std::endl;
									match = 0;
								}
							}
						}
					}
				}
			}
		}

		top_vf.commitTensor();
		top.commitTensor();
		if (match)
		{
			std::cout << "Passed varifier: layer: conv: " << getName() << std::endl;
		}


		return(ret);
	}


	/************************************************************************************************************************
	**
	**			BACKWARD PROPAGATION
	**
	************************************************************************************************************************/

	int aDNNodeConv::ConstructBwd(void)
	{
		int ret = 0;

		ret = aDNNode::ConstructBwd();

		ret = ConstructWeightsBwd();

		const aDNNTensor & bot = getBotFwd();
		int batch_sz = (int)bot.getDim(aDNN_TENSOR_BATCH);

		aDNNTensor & weights_df = getSlot(getWeightsDiffNm());
// create weights sum
		adnn_data_parameters weight_df_sum_descr;
		memset(&weight_df_sum_descr, 0, sizeof(adnn_data_parameters));
		weight_df_sum_descr.data_format = ADNN_DF_FP32;
		weight_df_sum_descr.batch_format = ADNN_BF_NHW;
		weight_df_sum_descr.dims[0] = batch_sz;
//		weight_df_sum_descr.dims[1] = 1;
		weight_df_sum_descr.dims[1] = weights_df.getDim(aDNN_TENSOR_HEIGHT);
		weight_df_sum_descr.dims[2] = weights_df.getDim(aDNN_TENSOR_WIDTH);

		aDNNTensor & weights_df_psum = createSlot(getWeightsDiffNm() + ADNN_SUM_NM, weight_df_sum_descr);

		std::string comp_options = getGenericCompOptions();

		if (getDebugLevel() == 0)
		{
			comp_options += std::string("  -Wb,-hsail-reg-slots=8 -Wb,-hsail-reg32-pressure-limit=64 -Wb,-hsail-reg64-pressure-limit=64 ");
		}




		const aDNNTensor & top_df = getTopDiff();
//		const aDNNTensor & weights_df = getWeightsDiff();

		int pad = getPad();
		int stride = getKernelStride();
		int kernel_size = getKernelSz();
		int outputs = (int)top_df.getDim(aDNN_TENSOR_DEPTH);
		int inputs = (int)bot.getDim(aDNN_TENSOR_DEPTH);


		int top_df_batch_stride = (int)top_df.getStride(aDNN_TENSOR_DEPTH);
		int top_df_channel_stride = (int)top_df.getStride(aDNN_TENSOR_HEIGHT);
		int top_df_stride = (int)top_df.getStride(aDNN_TENSOR_WIDTH);
		int top_width = (int)top_df.getDim(aDNN_TENSOR_WIDTH);
		int top_height = (int)top_df.getDim(aDNN_TENSOR_HEIGHT);
		int bot_batch_stride = (int)bot.getStride(aDNN_TENSOR_DEPTH);
		int bot_channel_stride = (int)bot.getStride(aDNN_TENSOR_HEIGHT);
		int bot_stride = (int)bot.getStride(aDNN_TENSOR_WIDTH);
		int bot_width = (int)bot.getDim(aDNN_TENSOR_WIDTH);
		int bot_height = (int)bot.getDim(aDNN_TENSOR_HEIGHT);

		int weights_stride = (int)weights_df.getStride(aDNN_TENSOR_WIDTH);
		int weights_channel_stride = (int)weights_df.getStride(aDNN_TENSOR_HEIGHT);



// HEURISTICS
		int prv_h = 4;
		int prv_w = 4;

		// FIX IT!!!!
		int ocl_group_sz0 = 8;
		int ocl_group_sz1 = 8;

		if (stride == 1)
		{
			if (top_width <= 8)
			{
				prv_w = 1;
			}
			else if (top_width <= 16)
			{
				prv_w = 2;
			}
			else if (top_width <= 32)
			{
				prv_w = 4;
			}
			else
			{
				ocl_group_sz0 = 16;
				prv_w = 4;
			}
		}
		else
		{
			if (top_width <= 8)
			{
				prv_w = 1;
			}
			else
			{
				prv_w = 2;
			}
		}

		if (stride == 1)
		{
			if (top_height <= 8)
			{
				prv_h = 1;
			}
			else if (top_height <= 16)
			{
				prv_h = 2;
			}
			else if (top_height <= 32)
			{
				prv_h = 4;
			}
			else
			{
				ocl_group_sz1 = 16;
				prv_h = 2;
			}
		}
		else
		{
			if (top_height <= 8)
			{
				prv_h = 1;
			}
			else
			{
				prv_h = 2;
			}
		}

		int ocl_group_lg2sz1 = (int)ceil(log((double)ocl_group_sz1) / log(2.));
		int ocl_group_lg2sz0 = (int)ceil(log((double)ocl_group_sz0) / log(2.));

		int convbwd_tile_w = prv_w * ocl_group_sz0;
		int convbwd_tile_h = prv_h * ocl_group_sz1;

		int convbwd_n_tile_h = (top_width + convbwd_tile_w - 1) / convbwd_tile_w;
		int convbwd_n_tile_v = (top_height + convbwd_tile_h - 1) / convbwd_tile_h;


		comp_options +=
			std::string("-D ADNN_CONV_KERNEL_SZ=") + std::to_string((long long)kernel_size)
			+ std::string(" -D ADNN_CONV_N_OUTPUTS=") + std::to_string((long long)outputs)
			+ std::string(" -D ADNN_CONV_N_INPUTS=") + std::to_string((long long)inputs)
			+ std::string(" -D ADNN_CONV_PAD=") + std::to_string((long long)pad)
			+ std::string(" -D ADNN_CONV_STRIDE=") + std::to_string((long long)stride)
			+ std::string(" -D ADNN_CONVBWD_PRV_TOPDF_W=") + std::to_string((long long)prv_w)
			+ std::string(" -D ADNN_CONVBWD_PRV_TOPDF_H=") + std::to_string((long long)prv_h)
			+ std::string(" -D ADNN_CONVBWD_GROUP_SZ0=") + std::to_string((long long)ocl_group_sz0)
			+ std::string(" -D ADNN_CONVBWD_GROUP_SZ1=") + std::to_string((long long)ocl_group_sz1)
			+ std::string(" -D ADNN_CONVBWD_GROUP_LG2SZ0=") + std::to_string((long long)ocl_group_lg2sz0)
			+ std::string(" -D ADNN_CONVBWD_GROUP_LG2SZ1=") + std::to_string((long long)ocl_group_lg2sz1)
			+ std::string(" -D ADNN_CONV_BOT_BATCH_STRIDE=") + std::to_string((long long)bot_batch_stride)
			+ std::string(" -D ADNN_CONV_BOT_CHANNEL_STRIDE=") + std::to_string((long long)bot_channel_stride)
			+ std::string(" -D ADNN_CONV_BOT_STRIDE=") + std::to_string((long long)bot_stride)
			+ std::string(" -D ADNN_CONVBWD_TOPDF_BATCH_STRIDE=") + std::to_string((long long)top_df_batch_stride)
			+ std::string(" -D ADNN_CONVBWD_TOPDF_CHANNEL_STRIDE=") + std::to_string((long long)top_df_channel_stride)
			+ std::string(" -D ADNN_CONVBWD_TOPDF_STRIDE=") + std::to_string((long long)top_df_stride)
			+ std::string(" -D ADNN_CONV_BOT_WIDTH=") + std::to_string((long long)bot_width)
			+ std::string(" -D ADNN_CONV_BOT_HEIGHT=") + std::to_string((long long)bot_height)
			+ std::string(" -D ADNN_CONV_WEIGHTS_CHANNEL_STRIDE=") + std::to_string((long long)weights_channel_stride)
			+ std::string(" -D ADNN_CONV_WEIGHTS_STRIDE=") + std::to_string((long long)weights_stride)
			+ std::string(" -D ADNN_CONV_TOP_WIDTH=") + std::to_string((long long)top_width)
			+ std::string(" -D ADNN_CONV_TOP_HEIGHT=") + std::to_string((long long)top_height)
			+ std::string(" -D ADNN_CONV_BATCH_SZ=") + std::to_string((long long)batch_sz)
			+ std::string(" -D ADNN_CONVBWD_N_TILES_V=") + std::to_string((long long)convbwd_n_tile_v)
			+ std::string(" -D ADNN_CONVBWD_N_TILES_H=") + std::to_string((long long)convbwd_n_tile_h)
			;

		int do_bias = 0;
		comp_options += std::string(" -D ADNN_CONV_BIAS=") + std::to_string((long long)do_bias);


		// sum over batch
		int sum_grp_sz0 = 256;
		comp_options += std::string(" -D ADNN_CONVBSUM_GRP_SZ0=") + std::to_string((long long)sum_grp_sz0);

		// wrt B
		const aDNNTensor &  bot_df = getBotDiff();

		int bot_df_batch_stride = (int)bot_df.getStride(aDNN_TENSOR_DEPTH);
		int bot_df_channel_stride = (int)bot_df.getStride(aDNN_TENSOR_HEIGHT);
		int bot_df_stride = (int)bot_df.getStride(aDNN_TENSOR_WIDTH);


		// FIX THIS !!!
		int n_outs = 1;

		int n_out_pix_horiz_bwd = 4;
		int n_out_pix_vert_bwd = 2;

		int ocl_group_bwd_sz0 = 8;
		int ocl_group_bwd_sz1 = 8;


		if (bot_width <= 8)
		{
			n_out_pix_horiz_bwd = 1;
		}
		else if (bot_width <= 16 || stride > 1)
		{
			n_out_pix_horiz_bwd = 2;
		}
		else
		{
			n_out_pix_horiz_bwd = 4;
		}


		if (bot_height <= 8)
		{
			n_out_pix_vert_bwd = 1;
		}
		else if (bot_height <= 16)
		{
			n_out_pix_vert_bwd = 2;
		}

		n_outs = (inputs & 1) ? 1 : 2;


		int ocl_group_bwd_lg2sz0 = (int)ceil(log((double)ocl_group_bwd_sz1) / log(2.));;
		int ocl_group_bwd_lg2sz1 = (int)ceil(log((double)ocl_group_bwd_sz0) / log(2.));;

		comp_options += std::string(" -D ADNN_CONV_GROUP_SZ0=") + std::to_string((long long)ocl_group_bwd_sz0)
			+ std::string(" -D ADNN_CONV_GROUP_SZ1=") + std::to_string((long long)ocl_group_bwd_sz1)
			+ std::string(" -D ADNN_CONV_GROUP_LG2SZ0=") + std::to_string((long long)ocl_group_bwd_lg2sz0)
			+ std::string(" -D ADNN_CONV_GROUP_LG2SZ1=") + std::to_string((long long)ocl_group_bwd_lg2sz1)
			+ std::string(" -D ADNN_CONV_N_HORIZ_OUT_PIX=") + std::to_string((long long)n_out_pix_horiz_bwd)
			+ std::string(" -D ADNN_CONV_N_VERT_OUT_PIX=") + std::to_string((long long)n_out_pix_vert_bwd)
			+ std::string(" -D ADNN_CONVBWD_BOTDF_BATCH_STRIDE=") + std::to_string((long long)bot_df_batch_stride)
			+ std::string(" -D ADNN_CONVBWD_BOTDF_CHANNEL_STRIDE=") + std::to_string((long long)bot_df_channel_stride)
			+ std::string(" -D ADNN_CONVBWD_BOTDF_STRIDE=") + std::to_string((long long)bot_df_stride)
			+ std::string(" -D ADNN_CONV_N_OUTS=") + std::to_string((long long)n_outs);

		//		comp_options += parent_->getGenericCompOptions();
		// wrt to W
		{
			std::string kernel_file = "aDNNConvBwd1.cl";
			std::string kernel_name = "aDNNConvBwd_wrt_W";

			std::vector<size_t> l_wk;
			l_wk.push_back(ocl_group_sz0);
			l_wk.push_back(ocl_group_sz1);
			l_wk.push_back(1);

			std::vector<size_t> g_wk;
			g_wk.push_back(ocl_group_sz0);
			g_wk.push_back(ocl_group_sz1);
			g_wk.push_back(inputs*outputs *batch_sz);

			// could use separate queue
			CDNN_OCL_kern_exe kern_exe(this, kernel_name, kernel_file, comp_options, 0, &g_wk, &l_wk, 0);
			kern_exe.Construct();

			ocl_bwd_execs_.push_back(kern_exe);

		}
		// sum over batch
		{

			std::string kernel_file = "aDNNConvBwd1.cl";
			std::string kernel_name = "aDNNConvBwd_wrt_W_Bsum";


			std::vector<size_t> l_wk;
			l_wk.push_back(sum_grp_sz0);
			l_wk.push_back(1);
			l_wk.push_back(1);

			const aDNNTensor & weights_df_psum = getSlot(getWeightsDiffNm() + ADNN_SUM_NM);


			std::vector<size_t> g_wk;
			int weights_df_channel_stride = (int)weights_df.getStride(aDNN_TENSOR_HEIGHT);

			g_wk.push_back(weights_df_channel_stride);
			g_wk.push_back(1);
			g_wk.push_back(1);

			// could use separate queue
			CDNN_OCL_kern_exe kern_exe(this, kernel_name, kernel_file, comp_options, 0, &g_wk, &l_wk, 0);
			kern_exe.Construct();
			ocl_bwd_execs_.push_back(kern_exe);

		}

		//	comp_options += std::string("  -Wb,-hsail-reg-slots=8 -Wb,-hsail-reg32-pressure-limit=64 -Wb,-hsail-reg64-pressure-limit=64 ");
		// wrt B
		{
			std::string kernel_file = "aDNNConvBwd1.cl";
			std::string kernel_name = (stride == 1) ? "aDNNConvBwd_wrt_B" : "aDNNConvBwd_wrt_B2";


			std::vector<size_t> l_wk;
			l_wk.push_back(ocl_group_bwd_sz0);
			l_wk.push_back(ocl_group_bwd_sz1);
			l_wk.push_back(1);

			int n_horiz = (bot_width + n_out_pix_horiz_bwd * ocl_group_bwd_sz0 - 1) / (n_out_pix_horiz_bwd * ocl_group_bwd_sz0);
			int n_vert = (bot_height + n_out_pix_vert_bwd * ocl_group_bwd_sz1 - 1) / (n_out_pix_vert_bwd * ocl_group_bwd_sz1);

			std::vector<size_t> g_wk;
			g_wk.push_back(n_horiz * ocl_group_bwd_sz0);
			g_wk.push_back(n_vert * ocl_group_bwd_sz1);
			g_wk.push_back(inputs*batch_sz / n_outs);


			// the same queue
			CDNN_OCL_kern_exe kern_exe(this, kernel_name, kernel_file, comp_options, 0, &g_wk, &l_wk, 0);
			kern_exe.Construct();
			ocl_bwd_execs_.push_back(kern_exe);

		}

		return(ret);
	}


	int aDNNodeConv::BuildBwd(void)
	{
		int ret = 0;

		ret = aDNNode::BuildBwd();

		ret = BuildWeightsBwd();

		// wrt to W
		{
			aDNNTensor & weights_df_psum = getSlot(getWeightsDiffNm() + ADNN_SUM_NM);
			// allocate real memory
			weights_df_psum.allocTensor();

//			const aDNNTensor & weights_df = getWeightsDiff();
			const aDNNTensor & top_df = getTopDiff();
			const aDNNTensor & bot = getBotFwd();
			cl_mem top_df_mem = top_df.getOCLBuffer();
			cl_mem bot_mem = bot.getOCLBuffer();
			cl_mem weights_df_mem = weights_df_psum.getOCLBuffer();
			aDType padding_value = 0;

			int n_arg = 0;
			ocl_args kern_args;
			kern_args[n_arg++] = std::make_pair(sizeof(cl_mem), &top_df_mem);
			kern_args[n_arg++] = std::make_pair(sizeof(cl_mem), &bot_mem);
			kern_args[n_arg++] = std::make_pair(sizeof(cl_mem), &weights_df_mem);
			kern_args[n_arg++] = std::make_pair(sizeof(aDType), &padding_value);


			CDNN_OCL_kern_exe & kern_exe = ocl_bwd_execs_[0];

			ret = kern_exe.Build(kern_args);


		}
		// sum over batch
		{

			aDNNTensor & weights_df_psum = getSlot(getWeightsDiffNm() + ADNN_SUM_NM);

			const aDNNTensor & weights_df = getWeightsDiff();
			cl_mem weights_df_psum_mem = weights_df_psum.getOCLBuffer();
			cl_mem weights_df_mem = weights_df.getOCLBuffer();

			int n_arg = 0;
			ocl_args kern_args;
			kern_args[n_arg++] = std::make_pair(sizeof(cl_mem), &weights_df_psum_mem);
			kern_args[n_arg++] = std::make_pair(sizeof(cl_mem), &weights_df_mem);

			CDNN_OCL_kern_exe & kern_exe = ocl_bwd_execs_[1];

			ret = kern_exe.Build(kern_args);
		}

		//	comp_options += std::string("  -Wb,-hsail-reg-slots=8 -Wb,-hsail-reg32-pressure-limit=64 -Wb,-hsail-reg64-pressure-limit=64 ");
		// wrt B
		{
			const aDNNTensor & top_df = getTopDiff();
			const aDNNTensor & bot_df = getBotDiff();
			const aDNNTensor & weights = getBotWeightsFwd();
			cl_mem top_df_mem = top_df.getOCLBuffer();
			cl_mem weights_mem = weights.getOCLBuffer();
			cl_mem bot_df_mem = getBotDiff().getOCLBuffer();
			aDType padding_value = 0;


			int n_arg = 0;
			ocl_args kern_args;
			kern_args[n_arg++] = std::make_pair(sizeof(cl_mem), &top_df_mem);
			kern_args[n_arg++] = std::make_pair(sizeof(cl_mem), &weights_mem);
			kern_args[n_arg++] = std::make_pair(sizeof(cl_mem), &bot_df_mem);
			kern_args[n_arg++] = std::make_pair(sizeof(aDType), &padding_value);

			CDNN_OCL_kern_exe & kern_exe = ocl_bwd_execs_[2];

			ret = kern_exe.Build(kern_args);

		}



		return(ret);

	}

	int aDNNodeConv::RunBwd(const adnn_node_parameters * running_params)
	{
		int ret = 0;

		int iter = getNTimingIter();
		double s = 0, e = 0;

		if (isPerLayerTiming())
		{
			s = mach_absolute_time();
		}

		for (int i = 0; i < iter; i++)
		{

			{

				// might be a separate but currently the same queue
				cl_command_queue convQ = ((CaLibsOCL*)getBaseOcl())->getClQueue(0);
				ocl_bwd_execs_[0].ExecuteNoWait(NULL, convQ);
				ocl_bwd_execs_[1].ExecuteNoWait(NULL, convQ);
			}


			{
				// wrt Botttom data

				ocl_bwd_execs_[2].ExecuteNoWait(NULL);
			}

		}

		if (isPerLayerTiming())
		{
			clFinish(ocl_bwd_execs_[0].getOclQueue());
			e = mach_absolute_time();
		}
		// verify

		if (getDebugLevel() == 1)
		{
			ret = VerifyBwd();
		}

		if (isPerLayerMessaging())
		{
			// APPROX for now
//			const aDNNTensor & top_df = getTopDiff();
//			const aDNNTensor & bot = getBotFwd();

			int width = (int)getTopDiff().getDim(aDNN_TENSOR_WIDTH);
			int height = (int)getTopDiff().getDim(aDNN_TENSOR_HEIGHT);
			int outputs = (int)getTopDiff().getDim(aDNN_TENSOR_DEPTH);
			// TO DO: check top, bot dim are equal
			int kernel_size = getKernelSz();
			int pad = getPad();
			int stride = getKernelStride();
			int batch_sz = (int)getBotDiff().getDim(aDNN_TENSOR_BATCH);
			int inputs = (int)getBotDiff().getDim(aDNN_TENSOR_DEPTH);
			iter = (iter <= 0) ? 1 : iter;
			processing_time_ = subtractTimes(e, s);
			int ident = 4;
			double comp_compexity = (double)2 * inputs*width*height*kernel_size*kernel_size*outputs*batch_sz * 2;  // multuiply by 2 due to 2 issues
			printf("Passed layer: convolution back-propagation: \"%s\"\n", getName().c_str());
			printf("%*s" "Arguments: CxWxHxKxKxOxB: %dx%dx%dx%dx%dx%dx%d\n", ident, " ", inputs, width, height, kernel_size, kernel_size, outputs, batch_sz);
			if (isPerLayerTiming())
			{
				printf("%*s" "Performance: %6.2f ms, %6.3f TFLOPs\n", ident, " ", processing_time_ / iter, (comp_compexity * iter) / (processing_time_ * 1000000000));
			}

		}

		return(ret);

	}

	int aDNNodeConv::RunHostBwd(void)
	{
		int ret = 0;


		int pad = getPad();
		int stride = getKernelStride();
		int kernel_size = getKernelSz();

		aDNNTensor & bot = (aDNNTensor & )getBotFwd();
		int inputs = (int)bot.getDim(aDNN_TENSOR_DEPTH);
		int batch_sz = (int)bot.getDim(aDNN_TENSOR_BATCH);
		int bot_width = (int)bot.getDim(aDNN_TENSOR_WIDTH);
		int bot_height = (int)bot.getDim(aDNN_TENSOR_HEIGHT);


		aDNNTensor & top_df = (aDNNTensor & )getTopDiff();
		int outputs = (int)top_df.getDim(aDNN_TENSOR_DEPTH);
		int top_df_batch_stride = (int)top_df.getStride(aDNN_TENSOR_DEPTH);
		int top_df_channel_stride = (int)top_df.getStride(aDNN_TENSOR_HEIGHT);
		int top_df_stride = (int)top_df.getStride(aDNN_TENSOR_WIDTH);
		int top_width = (int)top_df.getDim(aDNN_TENSOR_WIDTH);
		int top_height = (int)top_df.getDim(aDNN_TENSOR_HEIGHT);

		aDType * top_df_ptr = (aDType *)top_df.accessTensor(ADNN_MEM_ACCESS_READ);


		// wrt W
		{

			int bot_batch_stride = (int)bot.getStride(aDNN_TENSOR_DEPTH);
			int bot_channel_stride = (int)bot.getStride(aDNN_TENSOR_HEIGHT);
			int bot_stride = (int)bot.getStride(aDNN_TENSOR_WIDTH);

			aDNNTensor & weights_df_v = getSlot(getWeightsDiffNm() + ADNN_VERIFY_NM);

			int weights_width = (int)weights_df_v.getDim(aDNN_TENSOR_WIDTH);
			int weights_height = (int)weights_df_v.getDim(aDNN_TENSOR_HEIGHT);
			int weights_df_v_stride = (int)weights_df_v.getStride(aDNN_TENSOR_WIDTH);

			aDType * bot_ptr = (aDType * )bot.accessTensor(ADNN_MEM_ACCESS_READ);
			aDType * weights_df_v_ptr = (aDType * )weights_df_v.accessTensor(ADNN_MEM_ACCESS_WRITE);

			int im2col_batch_stride = weights_width * top_width * top_height; // - bias
			aDType * im2col_ptr = new aDType[im2col_batch_stride * batch_sz];


			memset(weights_df_v_ptr, 0, weights_df_v.getSizeInBytes());
			for (int b = 0; b < batch_sz; ++b)
			{
				ADNN_im2col_cpu<aDType>((const aDType*)&bot_ptr[bot_batch_stride * b], inputs,
					bot_height, bot_width, kernel_size, pad,
					stride, &im2col_ptr[im2col_batch_stride * b]);
				// sum up over mini-batch without bias
				ADNN_mm_cpu<aDType>((const aDType*)&top_df_ptr[top_df_batch_stride * b], top_width * top_height, outputs, top_df_channel_stride, 0,
					(const aDType *)&im2col_ptr[im2col_batch_stride * b], top_width * top_height, weights_width, top_width * top_height, ADNN_MM_TRANSPOSE,
					weights_df_v_ptr, weights_width, weights_height, weights_df_v_stride, 0,
					1, 1);
#if 0
				// sum up bias
				for (int o = 0; o < outputs; ++o)
				{
					for (int j = 0; j < top_height; ++j)
					{
						for (int i = 0; i < top_width; i++)
						{
							weights_df_v_ptr[weights_df_v_stride * o + (weights_width - 1)] += top_df_ptr[top_df_batch_stride * b + top_df_channel_stride * o + top_df_stride *j + i];
						}
					}
				}
#endif

			}

			bot.commitTensor();
			weights_df_v.commitTensor();
			delete[] im2col_ptr;
	}

		// wrt Bottom
	{
		// if propogate???

		aDNNTensor & bot_df_v = getSlot(getBotDiffNm() + ADNN_VERIFY_NM);
		int bot_df_v_batch_stride = (int)bot_df_v.getStride(aDNN_TENSOR_DEPTH);
		int bot_df_vchannel_stride = (int)bot_df_v.getStride(aDNN_TENSOR_HEIGHT);
		int bot_df_vstride = (int)bot_df_v.getStride(aDNN_TENSOR_WIDTH);

		aDType *bot_df_v_ptr = (aDType *)bot_df_v.accessTensor(ADNN_MEM_ACCESS_WRITE);

		aDNNTensor & weights = (aDNNTensor &)getBotWeightsFwd();

		int weights_width = (int)weights.getDim(aDNN_TENSOR_WIDTH);
		int weights_height = (int)weights.getDim(aDNN_TENSOR_HEIGHT);
		int weights_stride = (int)weights.getStride(aDNN_TENSOR_WIDTH);
		int weights_batch_stride = (int)weights.getStride(aDNN_TENSOR_DEPTH);
		const aDType * weights_ptr = (const aDType * )weights.accessTensor(ADNN_MEM_ACCESS_READ);


		int col_we_df_width = top_width*top_height;
		int col_we_df_height = weights_width; // - bias
		int col_we_batch_stride = col_we_df_width * col_we_df_height;
		int col_we_stride = col_we_df_width;
		aDType * col_we_df_ptr = new aDType[col_we_batch_stride * batch_sz];


		for (int b = 0; b < batch_sz; ++b)
		{
			ADNN_mm_cpu<aDType>(weights_ptr, weights_width, weights_height, weights_stride, ADNN_MM_TRANSPOSE,
				(const aDType *)&top_df_ptr[top_df_batch_stride * b], top_width * top_height, outputs, top_df_channel_stride, 0,
				&col_we_df_ptr[col_we_batch_stride * b], col_we_df_width, col_we_df_height, col_we_stride, 0,
				1, 0); //- bias

			ADNN_col2im_cpu<aDType>(&col_we_df_ptr[col_we_batch_stride * b], inputs, bot_height, bot_width, kernel_size, pad,
				stride, &bot_df_v_ptr[bot_df_v_batch_stride*b]);

		}


		delete[] col_we_df_ptr;
		bot_df_v.commitTensor();
		weights.commitTensor();

	}

	top_df.commitTensor();

		return(ret);

	}

	int aDNNodeConv::VerifyBwd(void)
	{
		int ret = 0;
		ret = RunHostBwd();
		int total_match = 1;

		// wrt W
		{

			aDNNTensor & weights_df_v = getSlot(getWeightsDiffNm() + ADNN_VERIFY_NM);
			aDNNTensor & weights_df = (aDNNTensor &)getWeightsDiff();


			int weights_width = (int)weights_df.getDim(aDNN_TENSOR_WIDTH);
			int weights_height = (int)weights_df.getDim(aDNN_TENSOR_HEIGHT);
			int weights_df_stride = (int)weights_df.getStride(aDNN_TENSOR_WIDTH);
			int weights_df_v_stride = (int)weights_df_v.getStride(aDNN_TENSOR_WIDTH);

			aDType * weights_df_ptr = (aDType *)weights_df.accessTensor(ADNN_MEM_ACCESS_READ);
			aDType * weights_df_v_ptr = (aDType *)weights_df_v.accessTensor(ADNN_MEM_ACCESS_READ);



			double sqr_accum = 0;
			double max_err = -std::numeric_limits<float>::min();
			int max_b = 0, max_o = 0, max_i = 0, max_j = 0;
			for (int j = 0; j < weights_height; j++)
			{
				// without bias
				for (int i = 0; i < weights_width; i++)
				{
					aDType c_val = weights_df_v_ptr[j*weights_df_v_stride + i];
					aDType g_val = weights_df_ptr[j*weights_df_stride + i];
					sqr_accum += (c_val - g_val) * (c_val - g_val);
					if (std::abs(c_val - g_val) > max_err)
					{
						max_err = std::abs(c_val - g_val);
						max_i = i;
						max_j = j;
					}

				}
			}
			sqr_accum = sqrt(sqr_accum / ((double)weights_height * weights_width));

			int match = 1;

			if (sqr_accum > 0 || std::isnan(sqr_accum) || !std::isfinite(sqr_accum))
			{
				std::cout << "Error in conv back-propagation: " << getName() + " wrt W : " << std::fixed << std::setw(15) << std::setprecision(13) << sqr_accum <<
					" Max err: " << std::fixed << std::setw(15) << std::setprecision(13) << max_err << std::endl;

				if (sqr_accum > (1. / 1000000000))
				{
					const double allowedEps = (1 <<2);
					for (int j = 0; j < weights_height && match; j++)
					{
						// without bias
						for (int i = 0; i < weights_width && match; i++)
						{
							aDType c_val = weights_df_v_ptr[j*weights_df_v_stride + i];
							aDType g_val = weights_df_ptr[j*weights_df_stride + i];

							double err = CalculateErr(c_val, g_val);
							if (err > allowedEps || std::isnan(c_val) || std::isnan(g_val) || !std::isfinite(c_val) || !std::isfinite(g_val))
							{
								std::cout << "Difference conv back-propagation: " << getName() << " wrt W : " << err << " too large at " << i << "," << j <<
									" c_v = " << std::fixed << std::setw(13) << std::setprecision(11) << c_val <<
									" vs g_val = " << std::fixed << std::setw(13) << std::setprecision(11) << g_val << std::endl;
								match = 0;
								total_match &= match;
							}

						}
					}
				}
			}

			weights_df.commitTensor();
			weights_df_v.commitTensor();


		}

		// wrt to B

		{

			aDNNTensor & bot_df_v = getSlot(getBotDiffNm() + ADNN_VERIFY_NM);
			int bot_df_v_batch_stride = (int)bot_df_v.getStride(aDNN_TENSOR_DEPTH);
			int bot_df_v_channel_stride = (int)bot_df_v.getStride(aDNN_TENSOR_HEIGHT);
			int bot_df_v_stride = (int)bot_df_v.getStride(aDNN_TENSOR_WIDTH);

			aDType *bot_df_v_ptr = (aDType *)bot_df_v.accessTensor(ADNN_MEM_ACCESS_READ);

			aDNNTensor & bot_df = (aDNNTensor &)getBotDiff();
			int bot_width = (int)bot_df.getDim(aDNN_TENSOR_WIDTH);
			int bot_height = (int)bot_df.getDim(aDNN_TENSOR_HEIGHT);
			int bot_df_batch_stride = (int)bot_df.getStride(aDNN_TENSOR_DEPTH);
			int bot_df_channel_stride = (int)bot_df.getStride(aDNN_TENSOR_HEIGHT);
			int bot_df_stride = (int)bot_df.getStride(aDNN_TENSOR_WIDTH);


			int inputs = (int)bot_df.getDim(aDNN_TENSOR_DEPTH);
			int batchs = (int)bot_df.getDim(aDNN_TENSOR_BATCH);

			aDType *bot_df_ptr = (aDType *)bot_df.accessTensor(ADNN_MEM_ACCESS_READ);


			const aDNNTensor & top_df = getBotDiff();
			int outputs = (int)top_df.getDim(aDNN_TENSOR_DEPTH);

			double sqr_accum = 0;
			double max_err = -std::numeric_limits<double>::min();;
			int max_b = 0, max_c = 0, max_i = 0, max_j = 0;

			for (int b = 0; b < batchs; ++b)
			{
				for (int c = 0; c < inputs; ++c)
				{
					for (int j = 0; j < bot_height; ++j)
					{
						for (int i = 0; i < bot_width; ++i)
						{
							aDType c_val = bot_df_v_ptr[b*bot_df_v_batch_stride + c*bot_df_v_channel_stride + j*bot_df_v_stride + i];
							aDType g_val = bot_df_ptr[b*bot_df_batch_stride + c*bot_df_channel_stride + j*bot_df_stride + i];

							sqr_accum += (c_val - g_val) * (c_val - g_val);
							if (std::abs(c_val - g_val) > max_err)
							{
								max_err = std::abs(c_val - g_val);
								max_b = b;
								max_c = c;
								max_i = i;
								max_j = j;
							}

						}
					}
				}
			}

			sqr_accum = sqrt(sqr_accum / ((double)batchs *inputs *bot_height *bot_width));

			int match = 1;

			if (sqr_accum > 0 || std::isnan(sqr_accum) || !std::isfinite(sqr_accum))
			{
				std::cout << "Error in conv back-propagation: " << getName() + " wrt B : " << std::fixed << std::setw(15) << std::setprecision(13) << sqr_accum <<
					" Max err: " << std::fixed << std::setw(15) << std::setprecision(13) << max_err << std::endl;


				if (sqr_accum > (1. / 1000000000))
				{
					const double allowedEps = (1<<2);
					for (int b = 0; b < batchs && match; ++b)
					{
						for (int c = 0; c < inputs && match; ++c)
						{
							for (int j = 0; j < bot_height && match; ++j)
							{
								for (int i = 0; i < bot_width && match; ++i)
								{
									aDType c_val = bot_df_v_ptr[b*bot_df_v_batch_stride + c*bot_df_v_channel_stride + j*bot_df_v_stride + i];
									aDType g_val = bot_df_ptr[b*bot_df_batch_stride + c*bot_df_channel_stride + j*bot_df_stride + i];


									double err = CalculateErr(c_val, g_val);
									if (err > allowedEps || std::isnan(c_val) || std::isnan(g_val) || !std::isfinite(c_val) || !std::isfinite(g_val))
									{
										std::cout << "Difference in conv back-propagation: " << getName() << " wrt B : " << err << " too large at " << b << "," << c << ", " << i << "," << j <<
											" c_v = " << std::fixed << std::setw(11) << std::setprecision(9) << c_val <<
											" vs g_val = " << std::fixed << std::setw(11) << std::setprecision(9) << g_val << std::endl;
										match = 0;
										total_match &= match;
									}
								}
							}
						}
					}
				}
			}


			bot_df_v.commitTensor();
			bot_df.commitTensor();
		}


		if (total_match)
		{
			std::cout << "Passed varifier: layer: conv back-propagation: " << getName() << std::endl;
		}

		return (ret);
	}

	/************************************************************************************************************************
	**
	**				UPDATE WEIGHTS
	**
	************************************************************************************************************************/

	int aDNNodeConv::UpdateWeights(void)
	{
		int ret = 0;
		int iter = getNTimingIter();
// before actual weight are updated
		if (getDebugLevel() == 1)
		{
			ret = UpdateWeightsHost();
		}

		double s = 0, e = 0;
		if (isPerLayerTiming())
		{
			s = mach_absolute_time();
		}

		for (int i = 0; i < iter; i++)
		{
			ret = UpdateWeightsInternal();
		}


		if (isPerLayerTiming())
		{
			clFinish(ocl_update_execs_[0].getOclQueue());
			e = mach_absolute_time();
		}
// after actual weights update
		if (getDebugLevel() == 1)
		{
			ret = VerifyUpdateWeights();
		}

		if (isPerLayerMessaging())
		{
			// APPROX for now
			const aDNNTensor & weights = getBotWeightsFwd();

			int weights_width = (int)weights.getDim(aDNN_TENSOR_WIDTH);
			int weights_height = (int)weights.getDim(aDNN_TENSOR_HEIGHT);
			iter = (iter <= 0) ? 1 : iter;
			processing_time_ = subtractTimes(e, s);
			int ident = 4;
			double comp_compexity = (double)weights_width * weights_height * 6;  // multuiply by 2 due to 2 issues
			printf("Passed layer: update convolution weights: \"%s\"\n", getName().c_str());
			printf("%*s" "Arguments: WxH: %dx%d\n", ident, " ", weights_width, weights_height);
			if (isPerLayerTiming())
			{
				printf("%*s" "Performance: %6.2f ms, %6.3f TFLOPs\n", ident, " ", processing_time_ / iter, (comp_compexity * iter) / (processing_time_ * 1000000000));
			}
		}


		// update counter
		setInternalCounter(getInternalCounter() + 1);

		return(ret);
	}


} // adnn



