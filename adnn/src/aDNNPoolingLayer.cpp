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

#include "aDNNInternal.hpp"


namespace adnn
{


	/************************************************************************************************************************
	**
	**			aDNNodePooling Class
	**
	************************************************************************************************************************/

	/**
	* Constructors
	*/
	aDNNodePooling::aDNNodePooling(const ADNNBase & lib, const adnn_node_parameters & node_params)
		:aDNNode(lib, node_params)
	{
	}


	aDNNodePooling::aDNNodePooling(void)
		: aDNNode()
	{
	}


	aDNNodePooling::aDNNodePooling(const aDNNodePooling & rh)
	{
		*this = rh;
	}

	const aDNNode & aDNNodePooling:: operator = (const aDNNodePooling & rh)
	{
		*(aDNNode*)this = *(aDNNode*)&rh;
		return *this;
	}

	/**
	* Destructor
	*/

	aDNNodePooling::~aDNNodePooling(void)
	{
	}


	int aDNNodePooling::Connect(void)
	{
		int ret = 0;
		return(ret);
	}




	int aDNNodePooling::Run(void)
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
	int aDNNodePooling::Construct(void)
	{
		int ret = 0;

		// to create internal system memory tensor for verification
		ConstructOutput();

		ConstructOptions();



		return(ret);
	}

	int aDNNodePooling::ConstructOptions(void)
	{
		int ret = 0;

		int pad = getPad();
		int stride = getKernelStride();
		int kernel_size = getKernelSz();

		const aDNNTensor & bot = getBotFwd();
		const aDNNTensor & top = getTopFwd();

		int bot_width = (int)bot.getDim(aDNN_TENSOR_WIDTH);
		int bot_height = (int)bot.getDim(aDNN_TENSOR_HEIGHT);
		int inputs = (int)bot.getDim(aDNN_TENSOR_DEPTH);
		int bot_stride = (int)bot.getStride(aDNN_TENSOR_WIDTH);
		int bot_channel_stride = (int)bot.getStride(aDNN_TENSOR_HEIGHT);
		int bot_batch_stride = (int)bot.getStride(aDNN_TENSOR_DEPTH);

		int top_width = (int)top.getDim(aDNN_TENSOR_WIDTH);
		int top_height = (int)top.getDim(aDNN_TENSOR_HEIGHT);
		int top_stride = (int)top.getStride(aDNN_TENSOR_WIDTH);
		int top_channel_stride = (int)top.getStride(aDNN_TENSOR_HEIGHT);
		int	top_batch_stride = (int)top.getStride(aDNN_TENSOR_DEPTH);

		int outputs = (int)top.getDim(aDNN_TENSOR_DEPTH);
		int height_out = (int)top.getDim(aDNN_TENSOR_HEIGHT);
		int width_out = (int)top.getDim(aDNN_TENSOR_WIDTH);


		int ocl_group_sz0 = 8;
		int ocl_group_sz1 = 8;
		int ocl_group_lg2sz0 = (int)ceil(log((double)ocl_group_sz0) / log(2.));
		int ocl_group_lg2sz1 = (int)ceil(log((double)ocl_group_sz1) / log(2.));;
		int n_out_pix_horiz = 1;
		int n_out_pix_vert = 1;


		if (top_width < ocl_group_sz0 * 4 || top_height < ocl_group_sz1 * 4)
		{
			n_out_pix_horiz = 1;
			n_out_pix_vert = 1;
		}
		else
		{
			n_out_pix_horiz = 2;
			n_out_pix_vert = 2;
		}

		int pooling_method = getPoolingMethod();
		int op_id = (pooling_method == ADNN_POOLING_MAX) ? ADNN_POOLING_OP_MAX : ADNN_POOLING_OP_AVE;

		int m_indxs_stride = 1;
		int m_indxs_channel_stride = 1;
		int m_indxs_batch_stride = 1;


		if (pooling_method == ADNN_POOLING_MAX)
		{

			// get top dimensions
			adnn_data_parameters indx_descr;
			top.getParams(indx_descr);
			for (int i = 0; i < ADNN_MAX_TENSOR_DIM; ++i)
			{
				indx_descr.strides[i] = 0;
			}

			indx_descr.data_format = ADNN_DF_UI8;

			std::string edge_nm = getOutputEdgeName();
			aDNNTensor & m_indxs = createSlot(edge_nm + ADNN_MAXINDX_NM, indx_descr);

			m_indxs_stride = (int)m_indxs.getStride(aDNN_TENSOR_WIDTH);
			m_indxs_channel_stride = (int)m_indxs.getStride(aDNN_TENSOR_HEIGHT);
			m_indxs_batch_stride = (int)m_indxs.getStride(aDNN_TENSOR_DEPTH);
		}
		std::string comp_options =
			std::string(" -D ADNN_POOLING_KERNEL_SZ=") + std::to_string((long long)kernel_size)
			+ std::string(" -D ADNN_POOLING_OP_ID=") + std::to_string((long long)op_id)
			+ std::string(" -D ADNN_POOLING_N_OUTPUTS=") + std::to_string((long long)outputs)
			+ std::string(" -D ADNN_POOLING_N_CHANNELS=") + std::to_string((long long)inputs)
			+ std::string(" -D ADNN_POOLING_PAD=") + std::to_string((long long)pad)
			+ std::string(" -D ADNN_POOLING_STRIDE=") + std::to_string((long long)stride)
			+ std::string(" -D ADNN_POOLING_N_HORIZ_OUT_PIX=") + std::to_string((long long)n_out_pix_horiz)
			+ std::string(" -D ADNN_POOLING_N_VERT_OUT_PIX=") + std::to_string((long long)n_out_pix_vert)
			+ std::string(" -D ADNN_POOLING_GROUP_SZ0=") + std::to_string((long long)ocl_group_sz0)
			+ std::string(" -D ADNN_POOLING_GROUP_SZ1=") + std::to_string((long long)ocl_group_sz1)
			+ std::string(" -D ADNN_POOLING_GROUP_LG2SZ0=") + std::to_string((long long)ocl_group_lg2sz0)
			+ std::string(" -D ADNN_POOLING_GROUP_LG2SZ1=") + std::to_string((long long)ocl_group_lg2sz1)
			+ std::string(" -D ADNN_POOLING_BOT_BATCH_STRIDE=") + std::to_string((long long)bot_batch_stride)
			+ std::string(" -D ADNN_POOLING_BOT_CHANNEL_STRIDE=") + std::to_string((long long)bot_channel_stride)
			+ std::string(" -D ADNN_POOLING_BOT_STRIDE=") + std::to_string((long long)bot_stride)
			+ std::string(" -D ADNN_POOLING_TOP_BATCH_STRIDE=") + std::to_string((long long)top_batch_stride)
			+ std::string(" -D ADNN_POOLING_TOP_CHANNEL_STRIDE=") + std::to_string((long long)top_channel_stride)
			+ std::string(" -D ADNN_POOLING_TOP_STRIDE=") + std::to_string((long long)top_stride)
			+ std::string(" -D ADNN_POOLING_BOT_WIDTH=") + std::to_string((long long)bot_width)
			+ std::string(" -D ADNN_POOLING_BOT_HEIGHT=") + std::to_string((long long)bot_height)
			+ std::string(" -D ADNN_POOLING_TOP_WIDTH=") + std::to_string((long long)top_width)
			+ std::string(" -D ADNN_POOLING_TOP_HEIGHT=") + std::to_string((long long)top_height)
			+ std::string(" -D ADNN_POOLING_MINDX_BATCH_STRIDE=") + std::to_string((long long)m_indxs_batch_stride)
			+ std::string(" -D ADNN_POOLING_MINDX_CHANNEL_STRIDE=") + std::to_string((long long)m_indxs_channel_stride)
			+ std::string(" -D ADNN_POOLING_MINDX_STRIDE=") + std::to_string((long long)m_indxs_stride)



			+getGenericCompOptions()
			;

		std::string kernel_file;
		std::string kernel_name;

		int g_wk_width = (int)((top.getDim(aDNN_TENSOR_WIDTH) + ocl_group_sz0 * n_out_pix_horiz - 1) / (ocl_group_sz0 * n_out_pix_horiz));
		int g_wk_height = (int)((top.getDim(aDNN_TENSOR_HEIGHT) + ocl_group_sz1 * n_out_pix_vert - 1) / (ocl_group_sz1 * n_out_pix_vert));

		std::vector<size_t> l_wk;
		l_wk.push_back(ocl_group_sz0);
		l_wk.push_back(ocl_group_sz1);
		l_wk.push_back(1);

		std::vector<size_t> g_wk;
		g_wk.push_back(g_wk_width * ocl_group_sz0);
		g_wk.push_back(g_wk_height * ocl_group_sz1);
		g_wk.push_back(top.getDim(aDNN_TENSOR_DEPTH) * top.getDim(aDNN_TENSOR_BATCH));


		kernel_file = "aDNNPooling.cl";

		if (pooling_method == ADNN_POOLING_MAX)
		{

			kernel_name = "aDNNPoolingMax";

		}
		else if (pooling_method == ADNN_POOLING_AVE)
		{

			kernel_name = "aDNNPoolingAve";

		}
		else
		{
			printf("Layer: %s. Error: unknowm method\n", getName().c_str());
			ret = -1;
		}

		if (ret == 0)
		{
			CDNN_OCL_kern_exe kern_exe(this, kernel_name, kernel_file, comp_options, 0, &g_wk, &l_wk, 0);

			kern_exe.Construct();

			ocl_fwd_execs_.push_back(kern_exe);
		}
		return (ret);
	}


	int aDNNodePooling::Build(void)
	{
		int ret = 0;
		const aDNNTensor & bot = getBotFwd();
		const aDNNTensor & top = getTopFwd();

		cl_mem bot_mem = bot.getOCLBuffer();

		cl_mem top_mem = top.getOCLBuffer();
		cl_mem m_indxs_mem = 0;
		// pass all arguments once

		// memory has to be allocated outside of the pipeline by the user

		CDNN_OCL_kern_exe & kern_exe = ocl_fwd_execs_[0];
		int n_arg = 0;
		ocl_args kern_args;
		if (bot_mem)
		{
			kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &bot_mem);
		}
		n_arg++;

		if (top_mem)
		{
			kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &top_mem);
		}
		n_arg++;

		int pooling_method = getPoolingMethod();
		if (pooling_method == ADNN_POOLING_MAX)
		{

			std::string edge_nm = getOutputEdgeName();
			aDNNTensor & m_indxs = (aDNNTensor &)getSlot(edge_nm + ADNN_MAXINDX_NM);
			ret = m_indxs.allocTensor();
			m_indxs_mem = m_indxs.getOCLBuffer();
			kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &m_indxs_mem);
		}

		kern_exe.Build(kern_args);

		return(ret);

	}


	int aDNNodePooling::RunFwd(const adnn_node_parameters * running_params)
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


			if (getOutputEdge().isDataUpdated())
			{
				cl_mem top_mem = ((aDNNTensor &)getOutputEdge().getData()).getOCLBuffer();
				getOutputEdge().setDataUpdated(false);
				additional_args[2] = std::make_pair(sizeof(cl_mem), &top_mem);
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


			int top_width = (int)top.getDim(aDNN_TENSOR_WIDTH);
			int top_height = (int)top.getDim(aDNN_TENSOR_HEIGHT);
			int bot_width = (int)bot.getDim(aDNN_TENSOR_WIDTH);
			int bot_height = (int)bot.getDim(aDNN_TENSOR_HEIGHT);
			int outputs = (int)top.getDim(aDNN_TENSOR_DEPTH);


			int batch_sz = (int)bot.getDim(aDNN_TENSOR_BATCH);
			int inputs = (int)bot.getDim(aDNN_TENSOR_DEPTH);

			size_t size = bot.getSize();
			iter = (iter <= 0) ? 1 : iter;
			processing_time_ = subtractTimes(e, s);
			int ident = 4;
			printf("Passed layer:pooling: \"%s\"\n", getName().c_str());
			printf("%*s" "Arguments: C x inW x inH x O x outW x outH x B: %d x %d x %d x %d x %d x %d x %d\n", ident, " ", inputs, bot_width, bot_height, outputs, top_width, top_height, batch_sz);
			if (isPerLayerTiming())
			{
				printf("%*s" "Performance: %6.2f ms\n", ident, " ", processing_time_ / iter);
			}
		}


		return(ret);

	}


	int aDNNodePooling::RunHostFwd(void)
	{
		int ret = 0;

		return(ret);
	}

	int aDNNodePooling::VerifyFwd(void)
	{
		int ret = 0;
		ret = RunHostFwd();

		aDNNTensor & bot = (aDNNTensor & )getBotFwd();
		aDNNTensor & top = (aDNNTensor & )getTopFwd();
		aDType * bot_ptr = (aDType *)bot.accessTensor(ADNN_MEM_ACCESS_READ);
		aDType * top_ptr = (aDType *)top.accessTensor(ADNN_MEM_ACCESS_READ);
		int pad = getPad();
		int stride = getKernelStride();
		int kernel_size = getKernelSz();

		int n_outputs = (int)top.getDim(aDNN_TENSOR_DEPTH);
		int n_batchs = (int)top.getDim(aDNN_TENSOR_BATCH);
		int bot_width = (int)bot.getDim(aDNN_TENSOR_WIDTH);
		int bot_height = (int)bot.getDim(aDNN_TENSOR_HEIGHT);
		int top_width = (int)top.getDim(aDNN_TENSOR_WIDTH);
		int top_height = (int)top.getDim(aDNN_TENSOR_HEIGHT);
		int bot_stride = (int)bot.getStride(aDNN_TENSOR_WIDTH);
		int bot_channel_stride = (int)bot.getStride(aDNN_TENSOR_HEIGHT);
		int bot_batch_stride = (int)bot.getStride(aDNN_TENSOR_DEPTH);

		int top_stride = (int)top.getStride(aDNN_TENSOR_WIDTH);
		int top_channel_stride = (int)top.getStride(aDNN_TENSOR_HEIGHT);
		int	top_batch_stride = (int)top.getStride(aDNN_TENSOR_DEPTH);


		int match = 1;
		int pooling_method = getPoolingMethod();

		for (int b = 0; b < n_batchs && match; b++)
		{
			for (int o = 0; o < n_outputs && match; o++)
			{
				for (int j = 0; j < top_height && match; j++)
				{
					for (int i = 0; i < top_width && match; i++)
					{
						// c-emulator
						aDType res = 0;
						if (pooling_method == ADNN_POOLING_MAX)
						{
							res = -FLT_MAX;
							int hstart = j * stride;
							int wstart = i * stride;
							int hend = std::min(hstart + kernel_size, bot_height);
							int wend = std::min(wstart + kernel_size, bot_width);
							for (int h = hstart; h < hend; ++h)
							{
								for (int w = wstart; w < wend; ++w)
								{
									res = std::max(res, bot_ptr[b*bot_batch_stride + o * bot_channel_stride + h * bot_stride + w]);
								}
							}
						}
						else if (pooling_method == ADNN_POOLING_AVE)
						{
							//						allowedEps = 4;
							res = 0;
							int hstart = j * stride - pad;
							int wstart = i * stride - pad;
							int hend = std::min(hstart + kernel_size, bot_height + pad);
							int wend = std::min(wstart + kernel_size, bot_width + pad);
							int pool_size = (hend - hstart) * (wend - wstart);
							hstart = std::max(hstart, 0);
							wstart = std::max(wstart, 0);
							hend = std::min(hend, bot_height);
							wend = std::min(wend, bot_width);
							for (int h = hstart; h < hend; ++h) {
								for (int w = wstart; w < wend; ++w) {
									res +=
										bot_ptr[b*bot_batch_stride + o * bot_channel_stride + h * bot_stride + w];
								}
							}
							res /= pool_size;
						}
						else
						{
							std::cout << "ERROR: unknown operator : layer: pooling: " << getName() << std::endl;
							match = 0;
							continue;
						}
						aDType c_val = res;
						aDType g_val = top_ptr[b*top_batch_stride + o * top_channel_stride + j * top_stride + i];
						double err = CalculateErr(c_val, g_val);
						double allowedEps = 3;
						if (err > allowedEps || std::isnan(c_val) || std::isnan(g_val) || !std::isfinite(c_val) || !std::isfinite(g_val))
						{
							std::cout << "Difference " << err << " too large at " << i << ", " << j << ", " << o << ", " << b << " c_v = " << c_val << " vs g_val = " << g_val << std::endl;
							match = 0;
						}
					}
				}
			}
		}


		if (match)
		{
			std::cout << "Passed varifier: layer: pooling: " << getName() << std::endl;
		}



		top.commitTensor();
		bot.commitTensor();

		return(ret);
	}


	/************************************************************************************************************************
	**
	**			BACKWARD PROPAGATION
	**
	************************************************************************************************************************/

	int aDNNodePooling::ConstructBwd(void)
	{
		int ret = 0;
		ret = aDNNode::ConstructBwd();


		const aDNNTensor & bot = getBotFwd();
		const aDNNTensor & top = getTopFwd();
		const aDNNTensor & bot_df = getBotDiff();
		const aDNNTensor & top_df = getTopDiff();

		int outputs = (int)bot.getDim(aDNN_TENSOR_DEPTH);
		int bot_batch_stride = (int)bot.getStride(aDNN_TENSOR_DEPTH);
		int bot_channel_stride = (int)bot.getStride(aDNN_TENSOR_HEIGHT);
		int bot_stride = (int)bot.getStride(aDNN_TENSOR_WIDTH);
		int bot_width = (int)bot.getDim(aDNN_TENSOR_WIDTH);
		int bot_height = (int)bot.getDim(aDNN_TENSOR_HEIGHT);

		int bot_df_batch_stride = (int)bot_df.getStride(aDNN_TENSOR_DEPTH);
		int bot_df_channel_stride = (int)bot_df.getStride(aDNN_TENSOR_HEIGHT);
		int bot_df_stride = (int)bot_df.getStride(aDNN_TENSOR_WIDTH);



		int top_batch_stride = (int)top.getStride(aDNN_TENSOR_DEPTH);
		int top_channel_stride = (int)top.getStride(aDNN_TENSOR_HEIGHT);
		int top_stride = (int)top.getStride(aDNN_TENSOR_WIDTH);
		int top_width = (int)top.getDim(aDNN_TENSOR_WIDTH);
		int top_height = (int)top.getDim(aDNN_TENSOR_HEIGHT);

		int top_df_batch_stride = (int)top_df.getStride(aDNN_TENSOR_DEPTH);
		int top_df_channel_stride = (int)top_df.getStride(aDNN_TENSOR_HEIGHT);
		int top_df_stride = (int)top_df.getStride(aDNN_TENSOR_WIDTH);

		int kernel_size = getKernelSz();
		int pad = getPad();
		int stride = getKernelStride();


		int n_out_pix_horiz = stride;
		int n_out_pix_vert = stride;
		int ocl_group_sz0 = 8;
		int ocl_group_sz1 = 8;
		int ocl_group_lg2sz0 = (int)ceil(log((double)ocl_group_sz0) / log(2.));
		int ocl_group_lg2sz1 = (int)ceil(log((double)ocl_group_sz1) / log(2.));

		int pooling_method = getPoolingMethod();
		if (pooling_method == ADNN_POOLING_MAX)
		{
			pad = 0;
		}

		std::string comp_options =
			std::string(" -D ADNN_POOLING_KERNEL_SZ=") + std::to_string((long long)kernel_size)
			+ std::string(" -D ADNN_POOLING_N_OUTPUTS=") + std::to_string((long long)outputs)
			+ std::string(" -D ADNN_POOLING_PAD=") + std::to_string((long long)pad)
			+ std::string(" -D ADNN_POOLING_STRIDE=") + std::to_string((long long)stride)
			+ std::string(" -D ADNN_POOLBWD_N_HORIZ_OUT_PIX=") + std::to_string((long long)n_out_pix_horiz)
			+ std::string(" -D ADNN_POOLBWD_N_VERT_OUT_PIX=") + std::to_string((long long)n_out_pix_vert)
			+ std::string(" -D ADNN_POOLBWD_GROUP_SZ0=") + std::to_string((long long)ocl_group_sz0)
			+ std::string(" -D ADNN_POOLBWD_GROUP_SZ1=") + std::to_string((long long)ocl_group_sz1)
			+ std::string(" -D ADNN_POOLBWD_GROUP_LG2SZ0=") + std::to_string((long long)ocl_group_lg2sz0)
			+ std::string(" -D ADNN_POOLBWD_GROUP_LG2SZ1=") + std::to_string((long long)ocl_group_lg2sz1)
			+ std::string(" -D ADNN_POOLBWD_BOT_BATCH_STRIDE=") + std::to_string((long long)bot_batch_stride)
			+ std::string(" -D ADNN_POOLBWD_BOT_CHANNEL_STRIDE=") + std::to_string((long long)bot_channel_stride)
			+ std::string(" -D ADNN_POOLBWD_BOT_STRIDE=") + std::to_string((long long)bot_stride)
			+ std::string(" -D ADNN_POOLBWD_TOP_BATCH_STRIDE=") + std::to_string((long long)top_batch_stride)
			+ std::string(" -D ADNN_POOLBWD_TOP_CHANNEL_STRIDE=") + std::to_string((long long)top_channel_stride)
			+ std::string(" -D ADNN_POOLBWD_TOP_STRIDE=") + std::to_string((long long)top_stride)
			+ std::string(" -D ADNN_POOLBWD_BOT_WIDTH=") + std::to_string((long long)bot_width)
			+ std::string(" -D ADNN_POOLBWD_BOT_HEIGHT=") + std::to_string((long long)bot_height)
			+ std::string(" -D ADNN_POOLBWD_TOP_WIDTH=") + std::to_string((long long)top_width)
			+ std::string(" -D ADNN_POOLBWD_TOP_HEIGHT=") + std::to_string((long long)top_height)
			+ std::string(" -D ADNN_POOLBWD_BOTDF_BATCH_STRIDE=") + std::to_string((long long)bot_df_batch_stride)
			+ std::string(" -D ADNN_POOLBWD_BOTDF_CHANNEL_STRIDE=") + std::to_string((long long)bot_df_channel_stride)
			+ std::string(" -D ADNN_POOLBWD_BOTDF_STRIDE=") + std::to_string((long long)bot_df_stride)
			+ std::string(" -D ADNN_POOLBWD_TOPDF_BATCH_STRIDE=") + std::to_string((long long)top_df_batch_stride)
			+ std::string(" -D ADNN_POOLBWD_TOPDF_CHANNEL_STRIDE=") + std::to_string((long long)top_df_channel_stride)
			+ std::string(" -D ADNN_POOLBWD_TOPDF_STRIDE=") + std::to_string((long long)top_df_stride)

			+ getGenericCompOptions()
			;




		int g_wk_width = (int)((bot_df.getDim(aDNN_TENSOR_WIDTH) + ocl_group_sz0 * n_out_pix_horiz - 1) / (ocl_group_sz0 * n_out_pix_horiz));
		int g_wk_height = (int)((bot_df.getDim(aDNN_TENSOR_HEIGHT) + ocl_group_sz1 * n_out_pix_vert - 1) / (ocl_group_sz1 * n_out_pix_vert));

		std::string kernel_file = "aDNNPoolingBwd.cl";
		std::string kernel_name;
		std::vector<size_t> l_wk;
		std::vector<size_t> g_wk;
		l_wk.push_back(ocl_group_sz0);
		l_wk.push_back(ocl_group_sz1);
		l_wk.push_back(1);

		g_wk.push_back(g_wk_width * ocl_group_sz0);
		g_wk.push_back(g_wk_height * ocl_group_sz1);
		g_wk.push_back(bot_df.getDim(aDNN_TENSOR_DEPTH) * bot_df.getDim(aDNN_TENSOR_BATCH));


		if (pooling_method == ADNN_POOLING_MAX)
		{
			kernel_name = "aDNNPoolingMaxBwd";
		}
		else if (pooling_method == ADNN_POOLING_AVE)
		{
			kernel_name = "aDNNPoolingAveBwd";
		}
		else
		{
			printf("Layer: %s. Error: unknowm method\n", getName().c_str());
			ret = -1;
		}

		if (ret == 0)
		{
			CDNN_OCL_kern_exe kern_exe(this, kernel_name, kernel_file, comp_options, 0, &g_wk, &l_wk, 0);

			kern_exe.Construct();

			ocl_bwd_execs_.push_back(kern_exe);
		}


		return(ret);
	}


	int aDNNodePooling::BuildBwd(void)
	{
		int ret = 0;

		ret = aDNNode::BuildBwd();

		const aDNNTensor & bot_df = getBotDiff();
		const aDNNTensor & top_df = getTopDiff();

		CDNN_OCL_kern_exe & kern_exe = ocl_bwd_execs_[0];
		int n_arg = 0;
		ocl_args kern_args;

		int pooling_method = getPoolingMethod();


		if (pooling_method == ADNN_POOLING_AVE || pooling_method == ADNN_POOLING_MAX)
		{

			cl_mem top_df_mem = top_df.getOCLBuffer();
			cl_mem bot_df_mem = bot_df.getOCLBuffer();

			kern_args[n_arg++] = std::make_pair(sizeof(cl_mem), &top_df_mem);
			kern_args[n_arg++] = std::make_pair(sizeof(cl_mem), &bot_df_mem);

//			std::string edge_nm = getOutputEdgeName();
//			aDNNTensor & m_indxs = (aDNNTensor & )getSlot(edge_nm + ADNN_MAXINDX_NM);
		}

		if (pooling_method == ADNN_POOLING_MAX)
		{
			const aDNNTensor & bot = getBotFwd();
			const aDNNTensor & top = getTopFwd();

			cl_mem top_mem = top.getOCLBuffer();
			cl_mem bot_mem = bot.getOCLBuffer();

			kern_args[n_arg++] = std::make_pair(sizeof(cl_mem), &top_mem);
			kern_args[n_arg++] = std::make_pair(sizeof(cl_mem), &bot_mem);
		}

		kern_exe.Build(kern_args);

		return(ret);

	}

	int aDNNodePooling::RunBwd(const adnn_node_parameters * running_params)
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
			ocl_bwd_execs_[0].ExecuteNoWait(NULL);

		}

		if (isPerLayerTiming())
		{
			clFinish(ocl_bwd_execs_[0].getOclQueue());
			e = mach_absolute_time();
		}


		if (getDebugLevel() == 1)
		{
			VerifyBwd();
		}

		if (isPerLayerMessaging())
		{
			const aDNNTensor & bot_df = getBotDiff();
			const aDNNTensor & top_df = getTopDiff();

			int out_width = (int)bot_df.getDim(aDNN_TENSOR_WIDTH);
			int out_height = (int)bot_df.getDim(aDNN_TENSOR_HEIGHT);

			int in_width = (int)top_df.getDim(aDNN_TENSOR_WIDTH);
			int in_height = (int)top_df.getDim(aDNN_TENSOR_HEIGHT);

			int inputs = (int)bot_df.getDim(aDNN_TENSOR_DEPTH);
			int outputs = (int)bot_df.getDim(aDNN_TENSOR_DEPTH);
			int batch_sz = (int)bot_df.getDim(aDNN_TENSOR_BATCH);

			iter = (iter <= 0) ? 1 : iter;
			processing_time_ = subtractTimes(e, s);
			int ident = 4;

			printf("Passed layer: pooling back-propagation: \"%s\"\n", getName().c_str());
			printf("%*s" "Arguments: CxIN_WxIN_HxOUT_Wx_OUT_HxOxB: %dx%dx%dx%dx%dx%dx%d\n", ident, " ", inputs, in_width, in_height, out_width, out_height, outputs, batch_sz);
			if (isPerLayerTiming())
			{
				printf("%*s" "Performance: %6.2f ms\n", ident, " ", processing_time_ / iter);
			}

		}

		return(ret);

	}

	int aDNNodePooling::RunHostBwd(void)
	{
		int ret = 0;
		aDNNTensor & bot_df_v = getSlot(getBotDiffNm() + ADNN_VERIFY_NM);
		aDNNTensor & bot = (aDNNTensor & )getBotFwd();
		aDNNTensor & top = (aDNNTensor &)getTopFwd();
//		aDNNTensor & bot_df = getBotDiff();
		aDNNTensor & top_df = (aDNNTensor &)getTopDiff();

		aDType * bot_df_v_ptr = (aDType *)bot_df_v.accessTensor(ADNN_MEM_ACCESS_WRITE);
		aDType * top_df_ptr = (aDType *)top_df.accessTensor(ADNN_MEM_ACCESS_READ);
		aDType * bot_ptr = (aDType *)bot.accessTensor(ADNN_MEM_ACCESS_READ);
		aDType * top_ptr = (aDType *)top.accessTensor(ADNN_MEM_ACCESS_READ);


		int bot_df_v_batch_stride = (int)bot_df_v.getStride(aDNN_TENSOR_DEPTH);
		int bot_df_v_channel_stride = (int)bot_df_v.getStride(aDNN_TENSOR_HEIGHT);
		int bot_df_v_stride = (int)bot_df_v.getStride(aDNN_TENSOR_WIDTH);
		int bot_batch_stride = (int)bot.getStride(aDNN_TENSOR_DEPTH);
		int bot_channel_stride = (int)bot.getStride(aDNN_TENSOR_HEIGHT);
		int bot_stride = (int)bot.getStride(aDNN_TENSOR_WIDTH);
		int bot_width = (int)bot.getDim(aDNN_TENSOR_WIDTH);
		int bot_height = (int)bot.getDim(aDNN_TENSOR_HEIGHT);
		int n_outputs = (int)bot.getDim(aDNN_TENSOR_DEPTH);
		int n_batchs = (int)bot.getDim(aDNN_TENSOR_BATCH);

		int top_df_batch_stride = (int)top_df.getStride(aDNN_TENSOR_DEPTH);
		int top_df_channel_stride = (int)top_df.getStride(aDNN_TENSOR_HEIGHT);
		int top_df_stride = (int)top_df.getStride(aDNN_TENSOR_WIDTH);
		int top_width = (int)top.getDim(aDNN_TENSOR_WIDTH);
		int top_height = (int)top.getDim(aDNN_TENSOR_HEIGHT);
		int top_batch_stride = (int)top.getStride(aDNN_TENSOR_DEPTH);
		int top_channel_stride = (int)top.getStride(aDNN_TENSOR_HEIGHT);
		int top_stride = (int)top.getStride(aDNN_TENSOR_WIDTH);


		int kernel_size = getKernelSz();
		int pad = getPad();
		int stride = getKernelStride();


		int pooling_method = getPoolingMethod();
		memset(bot_df_v_ptr, 0, bot_df_v.getSizeInBytes());
		for (int b = 0; b < n_batchs; b++)
		{
			for (int o = 0; o < n_outputs; o++)
			{
				int  bot_off = b * bot_batch_stride + o * bot_channel_stride;
				int  bot_df_v_off = b * bot_df_v_batch_stride + o * bot_df_v_channel_stride;
				int  top_df_off = b * top_df_batch_stride + o * top_df_channel_stride;
				int  top_off = b * top_batch_stride + o * top_channel_stride;

				if (pooling_method == ADNN_POOLING_MAX)
				{

					for (int j = 0; j < top_height; j++)
					{
						for (int i = 0; i < top_width; i++)
						{

							int hstart = j * stride;
							int wstart = i * stride;
							int hend = std::min(hstart + kernel_size, bot_height);
							int wend = std::min(wstart + kernel_size, bot_width);
							for (int h = hstart; h < hend; ++h) {
								for (int w = wstart; w < wend; ++w) {
									bot_df_v_ptr[bot_df_v_off + h * bot_df_v_stride + w] +=
										top_df_ptr[top_df_off + j * top_df_stride + i] *
										(bot_ptr[bot_off + h * bot_stride + w] ==
										top_ptr[top_off + j * top_stride + i]);
#if 0
									if (b == 0 && o == 5 && w == 17 && h == 0)
									{
										printf("C:max: %d %d   %13.11f  %13.11f  %13.11f %13.11f\n",
											i, j,
											bot_df_v_ptr[bot_df_v_off + h * bot_df_v_stride + w],
											top_df_ptr[top_df_off + j * top_df_stride + i],
											bot_ptr[bot_off + h * bot_stride + w],
											top_ptr[top_off + j * top_stride + i]
											);
									}
#endif
								}
							}

						}
					}

				}
				else if (pooling_method == ADNN_POOLING_AVE)
				{

					for (int j = 0; j < bot_height; j++)
					{
						for (int i = 0; i < bot_width; i++)
						{
							// c-emulator
							aDType res = 0;

							res = 0;
							bot_df_v_ptr[bot_df_v_off + j * bot_df_v_stride + i] = 0;
							int w = i + pad;
							int h = j + pad;
							int phstart = (h < kernel_size) ? 0 : (h - kernel_size) / stride + 1;
							int phend = std::min(h / stride + 1, top_height);
							int pwstart = (w < kernel_size) ? 0 : (w - kernel_size) / stride + 1;
							int pwend = std::min(w / stride + 1, top_width);
							aDType gradient = 0;
							for (int ph = phstart; ph < phend; ++ph) {
								for (int pw = pwstart; pw < pwend; ++pw) {
									// figure out the pooling size
									int hstart = ph * stride - pad;
									int wstart = pw * stride - pad;
									int hend = std::min(hstart + kernel_size, bot_height + pad);
									int wend = std::min(wstart + kernel_size, bot_width + pad);
									int pool_size = (hend - hstart) * (wend - wstart);
									gradient += top_df_ptr[top_df_off + ph * top_df_stride + pw] / pool_size;

#if 0
									if (b == 0 && o == 3 && i == 6 && j == 0)
									{
										printf("C:com: %10.8f %10.8f %10.8f %d\n", gradient, top_ptr[top_off + ph * top_stride + pw] / pool_size, top_ptr[top_off + ph * top_stride + pw], pool_size);
									}

#endif
								}
							}
							bot_df_v_ptr[bot_df_v_off + j * bot_df_v_stride + i] = gradient;
						}
					}
				}
				else
				{
					std::cout << "ERROR: unknown operator : layer: pooling back-propagation: " << getName() << std::endl;
					continue;
				}
#if 0
				aDType c_val = res;
				aDType g_val = top_ptr[b*top_batch_stride + o * top_channel_stride + j * top_stride + i];
				double err = CalculateErr(c_val, g_val);
				if (err > allowedEps)
				{
					std::cout << "Difference " << err << " too large at " << i << ", " << j << ", " << o << ", " << b << " c_v = " << c_val << " vs g_val = " << g_val << std::endl;
					match = 0;
				}
#endif


			}
		}


		top.commitTensor();
		bot.commitTensor();
		bot_df_v.commitTensor();
		top_df.commitTensor();


		return(ret);

	}

	int aDNNodePooling::VerifyBwd(void)
	{
		int ret = 0;
		ret = RunHostBwd();
		aDNNTensor & bot_df_v = getSlot(getBotDiffNm() + ADNN_VERIFY_NM);
		aDNNTensor & bot_df = (aDNNTensor & )getBotDiff();
		int bot_batch_stride = (int)bot_df.getStride(aDNN_TENSOR_DEPTH);
		int bot_channel_stride = (int)bot_df.getStride(aDNN_TENSOR_HEIGHT);
		int bot_stride = (int)bot_df.getStride(aDNN_TENSOR_WIDTH);
		int bot_width = (int)bot_df.getDim(aDNN_TENSOR_WIDTH);
		int bot_height = (int)bot_df.getDim(aDNN_TENSOR_HEIGHT);
		int n_outputs = (int)bot_df.getDim(aDNN_TENSOR_DEPTH);
		int n_batchs = (int)bot_df.getDim(aDNN_TENSOR_BATCH);
		int bot_v_batch_stride = (int)bot_df_v.getStride(aDNN_TENSOR_DEPTH);
		int bot_v_channel_stride = (int)bot_df_v.getStride(aDNN_TENSOR_HEIGHT);
		int bot_v_stride = (int)bot_df_v.getStride(aDNN_TENSOR_WIDTH);
		aDType * bot_ptr = (aDType *)bot_df.accessTensor(ADNN_MEM_ACCESS_READ);
		aDType * bot_v_ptr = (aDType *)bot_df_v.accessTensor(ADNN_MEM_ACCESS_READ);


		double sqr_accum = 0;
		double max_err = -std::numeric_limits<double>::min();
		int max_b = 0, max_o = 0, max_i = 0, max_j = 0;

		for (int b = 0; b < n_batchs; b++)
		{
			for (int o = 0; o < n_outputs; o++)
			{
				for (int j = 0; j < bot_height; j++)
				{
					for (int i = 0; i < bot_width; i++)
					{
						aDType c_val = bot_v_ptr[b*bot_v_batch_stride + o * bot_v_channel_stride + j * bot_v_stride + i];
						aDType g_val = bot_ptr[b*bot_batch_stride + o * bot_channel_stride + j * bot_stride + i];
						sqr_accum += (c_val - g_val) * (c_val - g_val);
						if (std::abs(c_val - g_val) > max_err)
						{
							max_err = std::abs(c_val - g_val);
							max_b = b;
							max_o = o;
							max_i = i;
							max_j = j;
						}

					}
				}
			}
		}

		sqr_accum = sqrt(sqr_accum / ((double)n_batchs *n_outputs * bot_height *bot_width));
		int match = 1;

		if (sqr_accum > 0 || std::isnan(sqr_accum) || !std::isfinite(sqr_accum))
		{
			std::cout << "Error in pooling back-propagation " << getName() + " : " << std::fixed << std::setw(15) << std::setprecision(13) << sqr_accum <<
				" Max err: " << std::fixed << std::setw(15) << std::setprecision(13) << max_err << std::endl;

			double allowedEps = 4;
			if (sqr_accum > (1. / 1000000000))
			{
				for (int b = 0; b < n_batchs && match; b++)
				{
					for (int o = 0; o < n_outputs && match; o++)
					{
						for (int j = 0; j < bot_height && match; j++)
						{
							for (int i = 0; i < bot_width && match; i++)
							{
								aDType c_val = bot_v_ptr[b*bot_v_batch_stride + o * bot_v_channel_stride + j * bot_v_stride + i];
								aDType g_val = bot_ptr[b*bot_batch_stride + o * bot_channel_stride + j * bot_stride + i];


								double err = CalculateErr(c_val, g_val);
								if (err > allowedEps || std::isnan(c_val) || std::isnan(g_val) || !std::isfinite(c_val) || !std::isfinite(g_val))
								{
									std::cout << "Difference in pooling back-propagation " << getName() + " " << err << " is too large at " << b << ", " << o << ", " << i << ", " << j <<
										" c_v = " << std::fixed << std::setw(13) << std::setprecision(11) << c_val <<
										" vs g_val = " << std::fixed << std::setw(13) << std::setprecision(11) << g_val << std::endl;
									match = 0;
								}

							}
						}
					}
				}
			}

		}

		bot_df.commitTensor();
		bot_df_v.commitTensor();

		if (match)
		{
			std::cout << "Passed varifier: layer: pooling back-propagation: " << getName() << std::endl;
		}



		return (ret);
	}



} // adnn






