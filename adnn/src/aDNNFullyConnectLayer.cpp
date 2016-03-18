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

namespace adnn
{

	/************************************************************************************************************************
	**
	**			aDNNodeFullyConnect Class
	**
	************************************************************************************************************************/

	/**
	* Constructors
	*/
	aDNNodeFullyConnect::aDNNodeFullyConnect(const ADNNBase & lib, const adnn_node_parameters & node_params)
		:aDNNode(lib, node_params)
	{
	}


	aDNNodeFullyConnect::aDNNodeFullyConnect(void)
		: aDNNode()
	{
	}


	aDNNodeFullyConnect::aDNNodeFullyConnect(const aDNNodeFullyConnect & rh)
	{
		*this = rh;
	}

	const aDNNode & aDNNodeFullyConnect:: operator = (const aDNNodeFullyConnect & rh)
	{
		*(aDNNode*)this = *(aDNNode*)&rh;
		return *this;
	}

	/**
	* Destructor
	*/

	aDNNodeFullyConnect::~aDNNodeFullyConnect(void)
	{
	}


	int aDNNodeFullyConnect::Connect(void)
	{
		int ret = 0;
		return(ret);
	}




	int aDNNodeFullyConnect::Run(void)
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
	/*------------------------------------------------------------------------------------------------------------------

	M - batch size
	K - number of inputs
	N - number of outputs

	forward pass:
	bot: MxK (M - number of rows)
	weights: NxK
	bias: 1XN

	bot = bot * transpose(weights): MxN + M * bias

	--------------------------------------------------------------------------------------------------------------------*/


	int aDNNodeFullyConnect::Construct(void)
	{
		int ret = 0;

		// to create internal system memory tensor for verification
		ConstructOutput();

		ConstructOptions();



		return(ret);
	}

	int aDNNodeFullyConnect::ConstructOptions(void)
	{
		int ret = 0;
		const aDNNTensor & mA = getBotFwd();
		const aDNNTensor & mC = getTopFwd();
		const aDNNTensor & mB = getBotWeightsFwd();
		//		const aDNNTensor & bias = getBotBiasFwd();
		int mA_width = (int)(mA.getDim(aDNN_TENSOR_WIDTH) * mA.getDim(aDNN_TENSOR_HEIGHT) * mA.getDim(aDNN_TENSOR_DEPTH));
		if (mA.getDim(aDNN_TENSOR_WIDTH) != mA.getStride(aDNN_TENSOR_WIDTH))
		{
			printf("Error: FullyConnect: cannot handle a non-flatten input\n");
		}

		assert(mA.getDim(aDNN_TENSOR_WIDTH) == mA.getStride(aDNN_TENSOR_WIDTH));

		int mA_height = (int)mA.getDim(aDNN_TENSOR_BATCH);
		int mA_stride = (int)mA.getStride(aDNN_TENSOR_DEPTH);
		int mB_width = (int)mB.getDim(aDNN_TENSOR_WIDTH);
		int mB_height = (int)mB.getDim(aDNN_TENSOR_HEIGHT);
		int mB_stride = (int)mB.getStride(aDNN_TENSOR_WIDTH);
		int mC_width = (int)mC.getDim(aDNN_TENSOR_WIDTH);
		int mC_height = (int)mC.getDim(aDNN_TENSOR_BATCH);
		int mC_stride = (int)mC.getStride(aDNN_TENSOR_WIDTH);

		int horiz_read_len = 16;
		int horiz_read_loop = 1;
		int horiz_read_lg2len = (int)ceil(log((double)horiz_read_len) / log(2.));
		int outer_loop = (mA_width + horiz_read_len - 1) / horiz_read_len;
		int out_n_cols;
		int out_n_rows;
		int priv_buf_len = 4;
		int accum_loop = (horiz_read_len * horiz_read_loop) / priv_buf_len;

		int ocl_group_sz0 = 8;
		int ocl_group_sz1 = 8;
		int ocl_group_lg2sz1 = (int)ceil(log((double)ocl_group_sz1) / log(2.));
		int ocl_group_lg2sz0 = (int)ceil(log((double)ocl_group_sz0) / log(2.));

		out_n_cols = mC_width / ocl_group_sz0;
		int n_out_pix_horiz = (out_n_cols > 4) ? 4 : (out_n_cols == 0) ? 1 : out_n_cols;
		//		n_out_pix_horiz_ = (out_n_cols > 2) ? 2 : (out_n_cols == 0) ? 1 : out_n_cols;
		out_n_rows = mC_height / ocl_group_sz1;
		int n_out_pix_vert = (out_n_rows > 4) ? 4 : (out_n_rows == 0) ? 1 : out_n_rows;
		//		n_out_pix_vert_ = (out_n_rows > 2) ? 2 : (out_n_rows == 0) ? 1 : out_n_rows;

		int vert_read_step = (ocl_group_sz0 * ocl_group_sz1) / horiz_read_len;
		int vert_mB_read_loop = (ocl_group_sz0 * n_out_pix_horiz) / vert_read_step;
		int vert_mA_read_loop = (ocl_group_sz1 * n_out_pix_vert) / vert_read_step;

		std::string comp_options =
			std::string(" -D ADNN_MM_N_HORIZ_OUT_PIX=") + std::to_string((long long)n_out_pix_horiz)
			+ std::string(" -D ADNN_MM_N_VERT_OUT_PIX=") + std::to_string((long long)n_out_pix_vert)
			+ std::string(" -D ADNN_MM_GROUP_SZ0=") + std::to_string((long long)ocl_group_sz0)
			+ std::string(" -D ADNN_MM_GROUP_SZ1=") + std::to_string((long long)ocl_group_sz1)
			+ std::string(" -D ADNN_MM_GROUP_LG2SZ0=") + std::to_string((long long)ocl_group_lg2sz0)
			+ std::string(" -D ADNN_MM_GROUP_LG2SZ1=") + std::to_string((long long)ocl_group_lg2sz1)
			+ std::string(" -D ADNN_MM_READ_LG2=") + std::to_string((long long)horiz_read_lg2len)
			+ std::string(" -D ADNN_MM_HORIZ_READ_LOOP=") + std::to_string((long long)horiz_read_loop)
			+ std::string(" -D ADNN_MM_OUTER_LOOP=") + std::to_string((long long)outer_loop)
			+ std::string(" -D ADNN_MM_MA_VERT_READ_LOOP=") + std::to_string((long long)vert_mA_read_loop)
			+ std::string(" -D ADNN_MM_MA_VERT_READ_STEP=") + std::to_string((long long)vert_read_step)
			+ std::string(" -D ADNN_MM_MB_VERT_READ_LOOP=") + std::to_string((long long)vert_mB_read_loop)
			+ std::string(" -D ADNN_MM_MB_VERT_READ_STEP=") + std::to_string((long long)vert_read_step)
			+ std::string(" -D ADNN_MM_MA_WIDTH=") + std::to_string((long long)mA_width)
			+ std::string(" -D ADNN_MM_MA_HEIGHT=") + std::to_string((long long)mA_height)
			+ std::string(" -D ADNN_MM_MA_STRIDE=") + std::to_string((long long)mA_stride)
			+ std::string(" -D ADNN_MM_MB_WIDTH=") + std::to_string((long long)mB_width)
			+ std::string(" -D ADNN_MM_MB_HEIGHT=") + std::to_string((long long)mB_height)
			+ std::string(" -D ADNN_MM_MB_STRIDE=") + std::to_string((long long)mB_stride)
			+ std::string(" -D ADNN_MM_MC_WIDTH=") + std::to_string((long long)mC_width)
			+ std::string(" -D ADNN_MM_MC_HEIGHT=") + std::to_string((long long)mC_height)
			+ std::string(" -D ADNN_MM_MC_STRIDE=") + std::to_string((long long)mC_stride)
			+ std::string(" -D ADNN_MM_ACCUM_LOOP=") + std::to_string((long long)accum_loop)
			+ std::string(" -D ADNN_MM_PRV_BUF=") + std::to_string((long long)priv_buf_len)
			+ getGenericCompOptions()
			;

		std::string kernel_file = "aDNNMatMat.cl";
		std::string kernel_name = "aDNN_FC"; // "aDNN_MM_TP";

		int i_n_group_horiz = (mC_width + ocl_group_sz0 * n_out_pix_horiz - 1) / (ocl_group_sz0 * n_out_pix_horiz);
		int i_n_group_vert = (mC_height + ocl_group_sz1 * n_out_pix_vert - 1) / (ocl_group_sz1 * n_out_pix_vert);

		std::vector<size_t> l_wk;
		l_wk.push_back(ocl_group_sz0);
		l_wk.push_back(ocl_group_sz1);
		l_wk.push_back(1);


		std::vector<size_t> g_wk;
		g_wk.push_back(i_n_group_horiz * l_wk[0]);
		g_wk.push_back(i_n_group_vert * l_wk[1]);
		g_wk.push_back(1);



		CDNN_OCL_kern_exe kern_exe(this, kernel_name, kernel_file, comp_options, 0, &g_wk, &l_wk, 0);

		kern_exe.Construct();

		ocl_fwd_execs_.push_back(kern_exe);

		return (ret);
	}


	int aDNNodeFullyConnect::Build(void)
	{
		int ret = 0;	// tensor for the host verification
		aDNNode::Build();

		// memory has to bealocated utside of the pipeline by user
		const aDNNTensor & mA = getBotFwd();
		const aDNNTensor & mC = getTopFwd();
		const aDNNTensor & mB = getBotWeightsFwd();
		const aDNNTensor & Bias = getBotBiasFwd();


		cl_mem mA_mem = mA.getOCLBuffer();
		cl_mem mB_mem = mB.getOCLBuffer();
		cl_mem mC_mem = mC.getOCLBuffer();
		cl_mem bias_mem = Bias.getOCLBuffer();

		// pass all arguments once
		int n_arg = 0;
		ocl_args kern_args;
		if (mA_mem)
		{
			kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &mA_mem);
		}
		n_arg++;
		if (mB_mem)
		{
			kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &mB_mem);
		}
		n_arg++;
		if (bias_mem)
		{
			kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &bias_mem);
		}
		n_arg++;
		if (mC_mem)
		{
			kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &mC_mem);
		}
		n_arg++;

		CDNN_OCL_kern_exe & kern_exe = ocl_fwd_execs_[0];

		kern_exe.Build(kern_args);


		return(ret);

	}

	int aDNNodeFullyConnect::RunFwd(const adnn_node_parameters * running_params)
	{
		int ret = 0;
		// execute through specialized object
		ocl_args additional_args;

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

			if (getInputEdge().isBiasUpdated() )
			{
				cl_mem bias_mem = ((aDNNTensor &)getInputEdge().getBiasData()).getOCLBuffer();
				getInputEdge().setBiasUpdated(false);
				additional_args[2] = std::make_pair(sizeof(cl_mem), &bias_mem);
			}


			if (getOutputEdge().isDataUpdated())
			{
				cl_mem top_mem = ((aDNNTensor &)getOutputEdge().getData()).getOCLBuffer();
				getOutputEdge().setDataUpdated(false);
				additional_args[3] = std::make_pair(sizeof(cl_mem), &top_mem);
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

			int inputs = (int)wei.getDim(aDNN_TENSOR_WIDTH);
			int outputs = (int)wei.getDim(aDNN_TENSOR_HEIGHT);

			// TO DO: check top, bot dim are equal
			int batch_sz = (int)bot.getDim(aDNN_TENSOR_BATCH);
			iter = (iter <= 0) ? 1 : iter;
			processing_time_ = subtractTimes(e, s);
			int ident = 4;
			printf("Passed layer:fully connected: \"%s\"\n", getName().c_str());
			printf("%*s" "Arguments: CxOxB: %dx%dx%d\n", ident, " ", inputs, outputs, batch_sz);
			if (isPerLayerTiming())
			{
				printf("%*s" "Performance: %6.2f ms, %6.3f TFLOPs\n", ident, " ", processing_time_ / iter, ((double)2 * inputs*outputs*batch_sz * iter) / (processing_time_ * 1000000000));
			}
		}


		return(ret);

	}


	int aDNNodeFullyConnect::RunHostFwd(void)
	{
		int ret = 0;
		std::string top_nm;
		top_nm = getOutputEdgeName();
		// get from the list of tensors referred by this node
		aDNNTensor & top_vf = getSlot(getTopNm() + ADNN_VERIFY_NM);

		aDNNTensor & bot = (aDNNTensor &)getBotFwd();
		aDNNTensor & weights = (aDNNTensor &)getBotWeightsFwd();
		aDNNTensor & bias = (aDNNTensor &)getBotBiasFwd();


		int w = (int)bot.getDim(aDNN_TENSOR_WIDTH);
		int h = (int)bot.getDim(aDNN_TENSOR_HEIGHT);
		int c = (int)bot.getDim(aDNN_TENSOR_DEPTH);

		int in_cols = w*h*c;

		int in_rows = (int)bot.getDim(aDNN_TENSOR_BATCH);
		int weight_cols = (int)weights.getDim(aDNN_TENSOR_WIDTH);
		int weights_rows = (int)weights.getDim(aDNN_TENSOR_HEIGHT);

		int bot_stride = (int)bot.getStride(aDNN_TENSOR_DEPTH);
		int weights_stride = (int)weights.getStride(aDNN_TENSOR_WIDTH);
		int top_stride = (int)top_vf.getStride(aDNN_TENSOR_WIDTH);
		//		int we_t_str = (int)mB->getStride(ANN_TENSOR_WIDTH);

		int top_rows = (int)top_vf.getDim(aDNN_TENSOR_BATCH);
		int top_cols = (int)top_vf.getDim(aDNN_TENSOR_WIDTH);

		aDType * bot_ptr = (aDType *)bot.accessTensor(ADNN_MEM_ACCESS_READ);
		aDType * top_ptr = (aDType *)top_vf.accessTensor(ADNN_MEM_ACCESS_WRITE);
		aDType * weights_ptr = (aDType *)weights.accessTensor(ADNN_MEM_ACCESS_READ);
		//	aDType * weights_t_ptr = mB->accessTensor(CL_MAP_READ);
		aDType * bias_ptr = (aDType *)bias.accessTensor(ADNN_MEM_ACCESS_READ);


		aDType * run_bot_ptr = bot_ptr;
		aDType * run_top_ptr = top_ptr;
		aDType * run_weights_ptr = weights_ptr;

		for (int k = 0; k < top_rows; ++k, run_top_ptr += top_stride)
		{
			for (int l = 0; l < top_cols; ++l)
			{
				// bias
				run_top_ptr[l] = bias_ptr[l];

#if 0
				printf("C:%d %d   %f\n", l, k, bias_ptr[l]);

#endif

				for (int j = 0; j < in_cols; ++j)
				{

					// transposed weight
					run_top_ptr[l] += bot_ptr[k*bot_stride + j] * weights_ptr[l*weights_stride + j];

				}

				run_top_ptr[l] += bias_ptr[l];
			}
		}

		top_vf.commitTensor();
		bot.commitTensor();
		weights.commitTensor();
		bias.commitTensor();
		//		mB->commitTensor();


		return(ret);
	}

	int aDNNodeFullyConnect::VerifyFwd(void)
	{
		int ret = 0;
		ret = RunHostFwd();
		std::string top_nm;
		// get from the list of tensors referred by this node
		aDNNTensor & top_vf = getSlot(getTopNm() + ADNN_VERIFY_NM);

		aDNNTensor & top = (aDNNTensor &)getTopFwd();

		aDType * top_ptr = (aDType *)top.accessTensor(CL_MAP_READ);

		aDType * top_vf_ptr = (aDType *)top_vf.accessTensor(CL_MAP_READ);

		int width = (int)top.getDim(aDNN_TENSOR_WIDTH);
		int batchs = (int)top.getDim(aDNN_TENSOR_BATCH);
		int top_v_batch_stride = (int)top_vf.getStride(aDNN_TENSOR_WIDTH);
		int top_batch_stride = (int)top.getStride(aDNN_TENSOR_WIDTH);


		double sqr_accum = 0;
		double max_err = -std::numeric_limits<double>::min();   // was FLT_MIN 
		int max_b = 0, max_c = 0, max_i = 0, max_j = 0;

		const double allowedEps = 4;
		for (int b = 0; b < batchs; ++b)
		{
			for (int i = 0; i < width; ++i)
			{
				aDType c_val = top_vf_ptr[b*top_v_batch_stride + i];
				aDType g_val = top_ptr[b*top_batch_stride + i];

				sqr_accum += (c_val - g_val) * (c_val - g_val);
				if (std::abs(c_val - g_val) > max_err)
				{
					max_err = std::abs(c_val - g_val);
					max_b = b;
					max_i = i;
				}

			}
		}

		sqr_accum = sqrt(sqr_accum / ((double)batchs *width));

		int match = 1;

		if (std::isnan(sqr_accum) || !std::isfinite(sqr_accum) || sqr_accum > 0)
		{
			std::cout << "Error in conv forward propagation: " << getName() + " " << std::fixed << std::setw(15) << std::setprecision(13) << sqr_accum <<
				" Max err: " << std::fixed << std::setw(15) << std::setprecision(13) << max_err << std::endl;


			if (sqr_accum > (1. / 1000000000))
			{
				for (int b = 0; b < batchs && match; ++b)
				{
					for (int i = 0; i < width && match; ++i)
					{
						aDType c_val = top_vf_ptr[b*top_v_batch_stride + i];
						aDType g_val = top_ptr[b*top_batch_stride + i];


						double err = CalculateErr(c_val, g_val);
						if (err > allowedEps || std::isnan(c_val) || std::isnan(g_val) || !std::isfinite(c_val) || !std::isfinite(g_val))
						{
							std::cout << "Difference in conv forward propagation: " << getName() + " " << err << " too large at " << b << ", " << i <<
								" c_v = " << std::fixed << std::setw(11) << std::setprecision(9) << c_val <<
								" vs g_val = " << std::fixed << std::setw(11) << std::setprecision(9) << g_val << std::endl;
							match = 0;
						}
					}
				}
			}
		}

		top_vf.commitTensor();
		top.commitTensor();
		if (match)
		{
			std::cout << "Passed varifier: layer: fully connected: " << getName() << std::endl;
		}

		return(ret);
	}


	/************************************************************************************************************************
	**
	**			BACKWARD PROPAGATION
	**
	************************************************************************************************************************/

	/*------------------------------------------------------------------------------------------------------------------

	M - batch size
	K - number of inputs
	N - number of outputs

	backward pass:
	top_diff: MxN (M - number of rows)
	weights_diff: NxK: transpose(top_diff) * bot: NxM * MxK = NxK
	bias_diff: 1XN

	bot_diff = top_diff * transpose(weights): MxN + M * bias

	--------------------------------------------------------------------------------------------------------------------*/
	int aDNNodeFullyConnect::ConstructBwd(void)
	{
		int ret = 0;

		ret = aDNNode::ConstructBwd();

		ret = ConstructWeightsBwd();

// create trasnpose top diff slot !!!!!
		const aDNNTensor & top_df = getSlot(getTopDiffNm());
		adnn_data_parameters top_df_descr;
		top_df.getParams(top_df_descr);
		for (int i = 0; i < ADNN_MAX_TENSOR_DIM; ++i)
		{
			top_df_descr.strides[i] = 0;
		}
		size_t temp = top_df_descr.dims[0];
		top_df_descr.dims[0] = top_df_descr.dims[1];
		top_df_descr.dims[1] = temp;

		const aDNNTensor & top_df_transp = createSlot(getTopDiffNm() + ADNN_TRANSPOSE_NM, top_df_descr);


		std::string comp_options;
// TO DO :: FIXED IT!!!
		// transpose top diff
		int ocl_mt_group_sz0 = 16;
		int ocl_mt_group_sz1 = 16;
		int n_mt_out_pix_horiz;
		int n_mt_out_pix_vert;
		int mA_row_loop;

		{
			int mA_width = (int)top_df.getDim(aDNN_TENSOR_WIDTH);
			int mA_height = (int)top_df.getDim(aDNN_TENSOR_HEIGHT);
			int mA_stride = (int)top_df.getStride(aDNN_TENSOR_WIDTH);
			int mB_stride = (int)top_df_transp.getStride(aDNN_TENSOR_WIDTH);



			int out_n_cols = mA_height / ocl_mt_group_sz0;
			n_mt_out_pix_horiz = (out_n_cols > 2) ? 2 : (out_n_cols == 0) ? 1 : out_n_cols;
			int out_n_rows = mA_width / ocl_mt_group_sz1;
			n_mt_out_pix_vert = (out_n_rows > 2) ? 2 : (out_n_rows == 0) ? 1 : out_n_rows;

			int min_n_mt = std::min(n_mt_out_pix_horiz, n_mt_out_pix_vert);
			n_mt_out_pix_horiz = n_mt_out_pix_vert = min_n_mt;
			comp_options =
				(
				std::string(" -D ADNN_MT_N_HORIZ_OUT_PIX=") + std::to_string((long long)n_mt_out_pix_horiz)
				+ std::string(" -D ADNN_MT_N_VERT_OUT_PIX=") + std::to_string((long long)n_mt_out_pix_vert)
				+ std::string(" -D ADNN_MT_GROUP_SZ0=") + std::to_string((long long)ocl_mt_group_sz0)
				+ std::string(" -D ADNN_MT_GROUP_SZ1=") + std::to_string((long long)ocl_mt_group_sz1)
				+ std::string(" -D ADNN_MT_MA_WIDTH=") + std::to_string((long long)mA_width)
				+ std::string(" -D ADNN_MT_MA_HEIGHT=") + std::to_string((long long)mA_height)
				+ std::string(" -D ADNN_MT_MA_STRIDE=") + std::to_string((long long)mA_stride)
				+ std::string(" -D ADNN_MT_MB_STRIDE=") + std::to_string((long long)mB_stride)
				)
				;

		}

		int grp_sr_sz0;
	{

		int mA_stride = (int)top_df_transp.getStride(aDNN_TENSOR_WIDTH);
		int mA_width = (int)top_df_transp.getDim(aDNN_TENSOR_WIDTH);
		grp_sr_sz0 = (mA_width <= 64) ? 64 : (mA_width <= 128) ? 128 : 256;

		mA_row_loop = (mA_width + grp_sr_sz0 - 1) / grp_sr_sz0; // ADNN_SR_MA_ROW_LOOP
		comp_options +=
			(
			std::string(" -D ADNN_SR_GROUP_SZ0=") + std::to_string((long long)grp_sr_sz0)
			+ std::string(" -D ADNN_SR_MA_WIDTH=") + std::to_string((long long)mA_width)
			+ std::string(" -D ADNN_SR_MA_STRIDE=") + std::to_string((long long)mA_stride)
			+ std::string(" -D ADNN_SR_MA_ROW_LOOP=") + std::to_string((long long)mA_row_loop)
			)
			;

	}

		comp_options += getGenericCompOptions();


		// transpose
		std::string kernel_file = "aDNNMatOps.cl";
		std::string kernel_name = "aDNN_MatTrans";


		int mA_width = (int)top_df.getDim(aDNN_TENSOR_WIDTH);
		int mA_height = (int)top_df.getDim(aDNN_TENSOR_HEIGHT);
		int i_n_group_horiz = (mA_width + ocl_mt_group_sz0 * n_mt_out_pix_horiz - 1) / (ocl_mt_group_sz0 * n_mt_out_pix_horiz);
		int i_n_group_vert = (mA_height + ocl_mt_group_sz1 * n_mt_out_pix_vert - 1) / (ocl_mt_group_sz1 * n_mt_out_pix_vert);

		std::vector<size_t> l_wk;
		l_wk.push_back(ocl_mt_group_sz0);
		l_wk.push_back(ocl_mt_group_sz1);
		l_wk.push_back(1);


		std::vector<size_t> g_wk;
		g_wk.push_back(i_n_group_horiz * l_wk[0]);
		g_wk.push_back(i_n_group_vert * l_wk[1]);
		g_wk.push_back(1);



		CDNN_OCL_kern_exe kern_exe_tr(this, kernel_name, kernel_file, comp_options, 0, &g_wk, &l_wk, 0);

		kern_exe_tr.Construct();

		ocl_bwd_execs_.push_back(kern_exe_tr);


		// row sum

		kernel_name = "aDNN_SumRow";

		mA_width = (int)top_df_transp.getDim(aDNN_TENSOR_WIDTH);
		mA_height = (int)top_df_transp.getDim(aDNN_TENSOR_HEIGHT);
		i_n_group_horiz = (mA_width + ocl_mt_group_sz0 * n_mt_out_pix_horiz - 1) / (ocl_mt_group_sz0 * n_mt_out_pix_horiz);

		l_wk[0] = grp_sr_sz0;
		l_wk[1] = 1;

		g_wk[0] = mA_row_loop * l_wk[0];
		g_wk[1] = mA_height;

		CDNN_OCL_kern_exe kern_exe_sr(this, kernel_name, kernel_file, comp_options, 0, &g_wk, &l_wk, 0);

		kern_exe_sr.Construct();

		ocl_bwd_execs_.push_back(kern_exe_sr);




		return(ret);
	}


	int aDNNodeFullyConnect::BuildBwd(void)
	{
		int ret = 0;

		ret = aDNNode::BuildBwd();

		ret = BuildWeightsBwd();

		const aDNNTensor & top_df = getSlot(getTopDiffNm());
		aDNNTensor & top_df_transp = (aDNNTensor & )getSlot(getTopDiffNm() + ADNN_TRANSPOSE_NM);
		ret = top_df_transp.allocTensor();

		cl_mem mA_mem = top_df.getOCLBuffer();
		cl_mem mB_mem = top_df_transp.getOCLBuffer();

		// pass all arguments once
		int n_arg = 0;
		ocl_args kern_args;
		kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &mA_mem);
		n_arg++;
		kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &mB_mem);

		CDNN_OCL_kern_exe & kern0_exe = ocl_bwd_execs_[0];

		kern0_exe.Build(kern_args);

		const aDNNTensor & bias_df = getSlot(getBiasDiffNm());
		cl_mem bias_df_mem = bias_df.getOCLBuffer();

		n_arg = 0;
		kern_args.clear();
		kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &mB_mem);
		n_arg++;
		kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &bias_df_mem);

		CDNN_OCL_kern_exe & kern1_exe = ocl_bwd_execs_[1];

		kern1_exe.Build(kern_args);


		return(ret);

	}

	int aDNNodeFullyConnect::RunBwd(const adnn_node_parameters * running_params)
	{
		int ret = 0;

		int iter = getNTimingIter();
		double s = 0, e = 0;

		const aDNNTensor & top_df = getSlot(getTopDiffNm());
		aDNNTensor & weights_df = (aDNNTensor &)getSlot(getWeightsDiffNm());
		const aDNNTensor & weights = getBotWeightsFwd();
		const aDNNTensor & bot = getBotFwd();
		aDNNTensor & bot_df = (aDNNTensor & )getSlot(getBotDiffNm());

		if (isPerLayerTiming())
		{
			s = mach_absolute_time();
		}

		for (int i = 0; i < iter; i++)
		{


			// weights grad
#if 1
			{
				int transposeA = 1;
				int transposeB = 0;
				double alpha = 1;
				double beta = 0;
				// mA
				size_t a_cols = top_df.getDim(aDNN_TENSOR_WIDTH) /** top_df.getDim(aDNN_TENSOR_HEIGHT) * top_df.getDim(aDNN_TENSOR_DEPTH)*/;
				size_t a_rows = top_df.getDim(aDNN_TENSOR_BATCH);
				// mB
				size_t b_cols = bot.getDim(aDNN_TENSOR_WIDTH) * bot.getDim(aDNN_TENSOR_HEIGHT) * bot.getDim(aDNN_TENSOR_DEPTH);
				size_t b_rows = bot.getDim(aDNN_TENSOR_BATCH);
				// mC
				size_t c_cols = weights_df.getDim(aDNN_TENSOR_WIDTH);
				size_t c_rows = weights_df.getDim(aDNN_TENSOR_HEIGHT);

				weights_df.mul2(c_cols, c_rows, (aDNNTensor &)top_df, a_cols, a_rows, (aDNNTensor & )bot, b_cols, b_rows, transposeA, transposeB, alpha, beta);
			}


			//bottom diff
			{
				int transposeA = 0;
				int transposeB = 0;
				double alpha = 1;
				double beta = 0;
				// mA
				size_t a_cols = top_df.getDim(aDNN_TENSOR_WIDTH)/* * top_df.getDim(aDNN_TENSOR_HEIGHT) * top_df.getDim(aDNN_TENSOR_DEPTH)*/;
				size_t a_rows = top_df.getDim(aDNN_TENSOR_BATCH);
				// mB
				size_t b_cols = weights.getDim(aDNN_TENSOR_WIDTH);
				size_t b_rows = weights.getDim(aDNN_TENSOR_HEIGHT);
				// mC
				size_t c_cols = bot_df.getDim(aDNN_TENSOR_WIDTH) * bot_df.getDim(aDNN_TENSOR_HEIGHT) * bot_df.getDim(aDNN_TENSOR_DEPTH);
				size_t c_rows = bot_df.getDim(aDNN_TENSOR_BATCH);

				bot_df.mul2(c_cols, c_rows, (aDNNTensor &)top_df, a_cols, a_rows, (aDNNTensor & )weights, b_cols, b_rows, transposeA, transposeB, alpha, beta);
			}

#endif
#if 1
			// bias grad
			{
				ocl_bwd_execs_[0].ExecuteNoWait(NULL);
			}


			{
				ocl_bwd_execs_[1].ExecuteNoWait(NULL);

			}

#endif
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
			const aDNNTensor & top_df = getSlot(getTopDiffNm());
			const aDNNTensor & weights_df = getSlot(getWeightsDiffNm());
			const aDNNTensor & weights = getBotWeightsFwd();
			const aDNNTensor & bot = getBotFwd();
			const aDNNTensor & bot_df = getSlot(getBotDiffNm());

			int inputs = (int)weights.getDim(aDNN_TENSOR_WIDTH);
			int outputs = (int)weights.getDim(aDNN_TENSOR_HEIGHT);

			// TO DO: check top, bot dim are equal
			int batch_sz = (int)bot.getDim(aDNN_TENSOR_BATCH);
			iter = (iter <= 0) ? 1 : iter;
			processing_time_ = subtractTimes(e, s);
			int ident = 4;
			printf("Passed layer:fully connected back propagation: \"%s\"\n", getName().c_str());
			printf("%*s" "Arguments: CxOxB: %dx%dx%d\n", ident, " ", inputs, outputs, batch_sz);
			if (isPerLayerTiming())
			{
				printf("%*s" "Performance: %6.2f ms, %6.3f TFLOPs\n", ident, " ", processing_time_ / iter, ((double)2 * inputs*outputs*batch_sz * 2 * iter) / (processing_time_ * 1000000000));
			}
		}

		return(ret);

	}

	int aDNNodeFullyConnect::RunHostBwd(void)
	{
		int ret = 0;
		aDNNTensor & top_df = (aDNNTensor &)getSlot(getTopDiffNm());
		aDNNTensor & weights_df_vr = (aDNNTensor &)getSlot(getWeightsDiffNm() + ADNN_VERIFY_NM);
		aDNNTensor & bot = (aDNNTensor &)getBotFwd();
		aDNNTensor & weights = (aDNNTensor &)getBotWeightsFwd();
		aDNNTensor & bot_df_vr = (aDNNTensor &)getSlot(getBotDiffNm() + ADNN_VERIFY_NM);
		// weights diff
		{
			int a_flags = ADNN_MM_TRANSPOSE;
			int b_flags = 0;
			int c_flags = 0;
			double alpha = 1;
			double beta = 0;
			// mA
			size_t a_cols = top_df.getDim(aDNN_TENSOR_WIDTH)/* * top_df.getDim(aDNN_TENSOR_HEIGHT) * top_df.getDim(aDNN_TENSOR_DEPTH)*/;
			size_t a_rows = top_df.getDim(aDNN_TENSOR_BATCH);
			size_t a_stride = top_df.getStride(aDNN_TENSOR_WIDTH);
			// mB
			size_t b_cols = bot.getDim(aDNN_TENSOR_WIDTH) * bot.getDim(aDNN_TENSOR_HEIGHT) * bot.getDim(aDNN_TENSOR_DEPTH);
			size_t b_rows = bot.getDim(aDNN_TENSOR_BATCH);
			size_t b_stride = bot.getStride(aDNN_TENSOR_DEPTH);
			// mC
			size_t c_cols = weights_df_vr.getDim(aDNN_TENSOR_WIDTH);
			size_t c_rows = weights_df_vr.getDim(aDNN_TENSOR_HEIGHT);
			size_t c_stride = weights_df_vr.getStride(aDNN_TENSOR_WIDTH);
			const aDType * a_ptr = (const aDType *)top_df.accessTensor(ADNN_MEM_ACCESS_READ);
			const aDType * b_ptr = (const aDType *)bot.accessTensor(ADNN_MEM_ACCESS_READ);
			aDType * c_ptr = (aDType *)weights_df_vr.accessTensor(ADNN_MEM_ACCESS_WRITE);

			ADNN_mm_cpu<aDType>(a_ptr, a_cols, a_rows, a_stride, a_flags,
				b_ptr, b_cols, b_rows, b_stride, b_flags,
				c_ptr, c_cols, c_rows, c_stride, c_flags,
				alpha, beta);

			top_df.commitTensor();
			bot.commitTensor();
			weights_df_vr.commitTensor();
		}
		// bot diff
		{
			int a_flags = 0;
			int b_flags = 0;
			int c_flags = 0;
			double alpha = 1;
			double beta = 0;
			// mA
			size_t a_cols = top_df.getDim(aDNN_TENSOR_WIDTH)/* * top_df.getDim(aDNN_TENSOR_HEIGHT) * top_df.getDim(aDNN_TENSOR_DEPTH)*/;
			size_t a_rows = top_df.getDim(aDNN_TENSOR_BATCH);
			size_t a_stride = top_df.getStride(aDNN_TENSOR_WIDTH);
			// mB
			size_t b_cols = weights.getDim(aDNN_TENSOR_WIDTH);
			size_t b_rows = weights.getDim(aDNN_TENSOR_HEIGHT);
			size_t b_stride = weights.getStride(aDNN_TENSOR_WIDTH);
			// mC
			size_t c_cols = bot_df_vr.getDim(aDNN_TENSOR_WIDTH) * bot_df_vr.getDim(aDNN_TENSOR_HEIGHT) * bot_df_vr.getDim(aDNN_TENSOR_DEPTH);
			size_t c_rows = bot_df_vr.getDim(aDNN_TENSOR_BATCH);
			size_t c_stride = bot_df_vr.getStride(aDNN_TENSOR_DEPTH);
			const aDType * a_ptr = (const aDType *)top_df.accessTensor(ADNN_MEM_ACCESS_READ);
			const aDType * b_ptr = (const aDType *)weights.accessTensor(ADNN_MEM_ACCESS_READ);
			aDType * c_ptr = (aDType *)bot_df_vr.accessTensor(ADNN_MEM_ACCESS_WRITE);

			ADNN_mm_cpu<aDType>(a_ptr, a_cols, a_rows, a_stride, a_flags,
				b_ptr, b_cols, b_rows, b_stride, b_flags,
				c_ptr, c_cols, c_rows, c_stride, c_flags,
				alpha, beta);

			top_df.commitTensor();
			weights.commitTensor();
			bot_df_vr.commitTensor();
		}
		return(ret);

	}

	int aDNNodeFullyConnect::VerifyBwd(void)
	{
		int ret = 0;
		ret = RunHostBwd();
		int match = 1;
		const double allowedEps = 3;

		{
			aDNNTensor & weights_df_vr = (aDNNTensor &)getSlot(getWeightsDiffNm() + ADNN_VERIFY_NM);
			aDNNTensor & weights_df = (aDNNTensor &)getSlot(getWeightsDiffNm());
			const aDType * weights_df_ptr = (const aDType * )weights_df.accessTensor(ADNN_MEM_ACCESS_READ);
			const aDType * weights_df_vr_ptr = (const aDType *)weights_df_vr.accessTensor(ADNN_MEM_ACCESS_READ);
			int width = (int)weights_df.getDim(aDNN_TENSOR_WIDTH);
			int height = (int)weights_df.getDim(aDNN_TENSOR_HEIGHT);
			int weights_df_vr_stride = (int)weights_df_vr.getStride(aDNN_TENSOR_WIDTH);
			int weights_df_stride = (int)weights_df.getStride(aDNN_TENSOR_WIDTH);
			for (int j = 0; j < height && match; j++)
			{
				for (int i = 0; i < width && match; i++)
				{
					aDType c_val = weights_df_vr_ptr[j*weights_df_vr_stride + i];
					aDType g_val = weights_df_ptr[j*weights_df_stride + i];
					double err = CalculateErr(c_val, g_val);
					if (err > allowedEps)
					{
						std::cout << "Difference in weights diff " << err << " too large at " << i << "," << j << " c_v = " << c_val << " vs g_val = " << g_val << std::endl;
						match = 0;
					}
				}
			}
			weights_df_vr.commitTensor();
			weights_df.commitTensor();

			
		}
	{

		aDNNTensor & bot_df = (aDNNTensor &)getSlot(getBotDiffNm());
		aDNNTensor & bot_df_vr = (aDNNTensor &)getSlot(getBotDiffNm() + ADNN_VERIFY_NM);
		size_t cols = bot_df.getDim(aDNN_TENSOR_WIDTH) * bot_df.getDim(aDNN_TENSOR_HEIGHT) * bot_df.getDim(aDNN_TENSOR_DEPTH);
		size_t rows = bot_df.getDim(aDNN_TENSOR_BATCH);
		int bot_df_stride = (int)bot_df.getStride(aDNN_TENSOR_DEPTH);
		const aDType * bot_df_ptr = (const aDType * )bot_df.accessTensor(ADNN_MEM_ACCESS_READ);
		int bot_df_vr_stride = (int)bot_df_vr.getStride(aDNN_TENSOR_DEPTH);
		const aDType * bot_df_vr_ptr = (const aDType *)bot_df_vr.accessTensor(ADNN_MEM_ACCESS_READ);
		for (int j = 0; j < rows && match; j++)
		{
			for (int i = 0; i < cols && match; i++)
			{
				aDType c_val = bot_df_vr_ptr[j*bot_df_vr_stride + i];
				aDType g_val = bot_df_ptr[j*bot_df_stride + i];
				double err = CalculateErr(c_val, g_val);
				if (err > allowedEps)
				{
					std::cout << "Difference in bottom diff " << err << " too large at " << i << "," << j << " c_v = " << c_val << " vs g_val = " << g_val << std::endl;
					match = 0;
				}
			}
		}
		bot_df_vr.commitTensor();
		bot_df.commitTensor();


	}
	if (match)
	{
		std::cout << "Passed varifier: layer: fully connected back-propagation: " << getName() << std::endl;
	}


		return (ret);
	}


/************************************************************************************************************************
**
**				UPDATE WEIGHTS
**
************************************************************************************************************************/

	int aDNNodeFullyConnect::UpdateWeights(void)
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

			int weights_width = (int)weights.getDim(aDNN_TENSOR_WIDTH) + 1;
			int weights_height = (int)weights.getDim(aDNN_TENSOR_HEIGHT);
			iter = (iter <= 0) ? 1 : iter;
			processing_time_ = subtractTimes(e, s);
			int ident = 4;
			double comp_compexity = (double)weights_width * weights_height * 6;  // multuiply by 2 due to 2 issues
			printf("Passed layer: update fully connected weights: \"%s\"\n", getName().c_str());
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






