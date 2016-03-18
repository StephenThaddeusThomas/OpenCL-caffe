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

/*

http://stats.stackexchange.com/questions/79454/softmax-layer-in-a-neural-network

the Jacobian matrix of exp(x[i]) / sum(exp(x[])

?h[i]/?z[j] = hi(1?hj) :i = j

?h[i]/?z[j] = ?hihj : i?j
These two concepts definitions can be conveniently combined using a construct called the Kronecker Delta, so the definition of the gradient becomes

?h[i]/?z[j] = h[i](?[i][j]?h[j])
So the Jacobian is a square matrix[J]ij = hi(?ij?hj)
we need to get the input errors from the output errors that are already computed.Since the gradient of the output error ?hi depends on all of the inputs, then the gradient of the input xi is

[?x]k = ?i = 1?hi, k
Given the Jacobian matrix defined above, this is implemented trivially as the product of the matrix and the output error vector :

?(l) = J * ?(l + 1)
If the softmax layer is your output layer, then combining it with the cross - entropy cost model simplifies the computation to simply

?(l) = h ?t
where t  is the vector of labels, and h  is the output from the softmax function.Not only is the simplified form convenient, it is also extremely useful from a numerical stability standpoint.

*/

namespace adnn
{



	/************************************************************************************************************************
	**
	**			aDNNodeSoftMax Class
	**
	************************************************************************************************************************/

	/**
	* Constructors
	*/
	aDNNodeSoftMax::aDNNodeSoftMax(const ADNNBase & lib, const adnn_node_parameters & node_params)
		:aDNNode(lib, node_params)
	{
		with_crossentropy_loos_ = false;
	}


	aDNNodeSoftMax::aDNNodeSoftMax(void)
		: aDNNode()
	{
		with_crossentropy_loos_ = false;
	}


	aDNNodeSoftMax::aDNNodeSoftMax(const aDNNodeSoftMax & rh)
	{
		*this = rh;

	}

	const aDNNode & aDNNodeSoftMax:: operator = (const aDNNodeSoftMax & rh)
	{
		*(aDNNode*)this = *(aDNNode*)&rh;
		with_crossentropy_loos_ = rh.with_crossentropy_loos_;
		return *this;
	}

	/**
	* Destructor
	*/

	aDNNodeSoftMax::~aDNNodeSoftMax(void)
	{
	}


	int aDNNodeSoftMax::Connect(void)
	{
		int ret = 0;
		return(ret);
	}




	int aDNNodeSoftMax::Run(void)
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

	int aDNNodeSoftMax::Construct(void)
	{
		int ret = 0;

		if (isCrossEntrypyLoss() && getOutputEdgeType() == ADNN_ED_SINK)
		{
			return(ret);
		}


		// to create internal system memory tensor for verification
		ConstructOutput();

		ConstructOptions();



		return(ret);
	}

	int aDNNodeSoftMax::ConstructOptions(void)
	{
		int ret = 0;
		size_t batch_sz = 0;


		const aDNNTensor & bot = getBotFwd();
#if 1
		// it's a loss layer and there is no diff slot
		// create one

		if (isCrossEntrypyLoss() && isSlotEmpty(getBotDiffNm()))
		{
			cloneSlot(getBotDiffNm(), bot);
			if (getDebugLevel() == 1 && isSlotEmpty(getBotDiffNm() + ADNN_VERIFY_NM))
			{
				cloneSlot(getBotDiffNm() + ADNN_VERIFY_NM, getBotDiff());
			}
		}
#endif

		const aDNNTensor & top = (isCrossEntrypyLoss()) ? getBotDiff() : getTopFwd();

		// 2nd input edge
		//		const aDNNTensor & labels = getBotFwd(1);

		int width = (int)bot.getDim(aDNN_TENSOR_WIDTH);
		if (width >= (2 << 11))
		{
			ret = -1;
			printf("ERROR: SoftMax has not been implemented for more than 2K categories\n");
			return(ret);
		}

		batch_sz = bot.getDim(aDNN_TENSOR_BATCH);

		int ocl_group_sz1 = 1;
		int ocl_group_sz0 = (width <= 64) ? 64 : (width <= 128) ? 128 : (width <= 192) ? 192 : 256;
		int ocl_group_lg2sz0 = (int)ceil(log((double)ocl_group_sz0) / log(2.));
		int in_loop = ((int)width + ocl_group_sz0 - 1) / ocl_group_sz0;
		int in_align_len = in_loop * ocl_group_sz0;
		int lcl_data_lg2len = (int)ceil(log((double)in_align_len) / log(2.));
		int lcl_data_len = (1 << lcl_data_lg2len);

		int in_stride = (int)bot.getStride(aDNN_TENSOR_WIDTH);
		int	out_stride = (int)top.getStride(aDNN_TENSOR_WIDTH);


		std::string comp_options =
			std::string(" -D ADNN_SM_GROUP_SZ0=") + std::to_string((long long)ocl_group_sz0)
			+ std::string(" -D ADNN_SM_GROUP_SZ1=") + std::to_string((long long)ocl_group_sz1)
			+ std::string(" -D ADNN_SM_GROUP_LG2SZ0=") + std::to_string((long long)ocl_group_lg2sz0)
			+ std::string(" -D ADNN_SM_IN_LEN=") + std::to_string((long long)width)
			+ std::string(" -D ADNN_SM_IN_LOOP=") + std::to_string((long long)in_loop)
			+ std::string(" -D ADNN_SM_LCL_DATA_LEN=") + std::to_string((long long)lcl_data_len)
			+ std::string(" -D ADNN_SM_IN_STRIDE=") + std::to_string((long long)in_stride)
			+ std::string(" -D ADNN_SM_OUT_STRIDE=") + std::to_string((long long)out_stride)
			+ std::string(" -D ADNN_SM_LCL_DATA_LG2LEN=") + std::to_string((long long)lcl_data_lg2len)

			+ getGenericCompOptions()
			;

		std::string kernel_file = "aDNNSoftMax.cl";
		std::string kernel_name = (isCrossEntrypyLoss()) ? "aDNN_SM_withCrossEntropyLoss" : "aDNN_SM";

		std::vector<size_t> l_wk;

		l_wk.push_back(ocl_group_sz0);
		l_wk.push_back(ocl_group_sz1);
		l_wk.push_back(1);


		std::vector<size_t> g_wk;

		g_wk.push_back(in_align_len);
		g_wk.push_back(batch_sz);
		g_wk.push_back(1);


		CDNN_OCL_kern_exe kern_exe(this, kernel_name, kernel_file, comp_options, 0, &g_wk, &l_wk, 0);

		kern_exe.Construct();

		ocl_fwd_execs_.push_back(kern_exe);

		return (ret);
	}

	int aDNNodeSoftMax::BuildIntnl(void)
	{
		int ret = 0;



		const aDNNTensor & bot = getBotFwd();
		aDNNTensor & top = (aDNNTensor &)((isCrossEntrypyLoss()) ? getBotDiff() : getTopFwd());

		// 2nd input edge

		// memory might be allocated outside of the pipeline by the user

		cl_mem bot_mem = bot.getOCLBuffer();
		cl_mem top_mem = top.getOCLBuffer();

		CDNN_OCL_kern_exe & kern_exe = ocl_fwd_execs_[0];
		int n_arg = 0;
		ocl_args kern_args;
		if (bot_mem)
		{
			kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &bot_mem);
		}
		n_arg++;

		if (isCrossEntrypyLoss())
		{
			const aDNNTensor & labels = getBotFwd(1);
			cl_mem labels_mem = labels.getOCLBuffer();

			if (labels_mem)
			{
				kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &labels_mem);
			}
			n_arg++;
		}
		if (top_mem)
		{
			kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &top_mem);
		}
		n_arg++;

		kern_exe.Build(kern_args);

		return(ret);

	}


	int aDNNodeSoftMax::Build(void)
	{
		int ret = 0;


		if (isCrossEntrypyLoss() && getOutputEdgeType() == ADNN_ED_SINK)
		{
			return(ret);
		}

		ret = aDNNode::Build();
		ret = BuildIntnl();
		return(ret);

	}

	int aDNNodeSoftMax::RunFwd(const adnn_node_parameters * running_params)
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

			if (isCrossEntrypyLoss() && getInputEdge(1).isDataUpdated())
			{
				cl_mem labels_mem = ((aDNNTensor &)getInputEdge(1).getData()).getOCLBuffer();
				getInputEdge(1).setDataUpdated(false);
				additional_args[1] = std::make_pair(sizeof(cl_mem), &labels_mem);
			}

			if (!isCrossEntrypyLoss() && getOutputEdge().isDataUpdated())
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

			int batch_sz = (int)bot.getDim(aDNN_TENSOR_BATCH);
			int width = (int)bot.getDim(aDNN_TENSOR_WIDTH);

			iter = (iter <= 0) ? 1 : iter;
			processing_time_ = subtractTimes(e, s);
			int ident = 4;
			printf("Passed layer: SoftMax%s: \"%s\"\n",
				(isCrossEntrypyLoss()) ? " with cross entropy loss" : "",
				getName().c_str());
			printf("%*s" "Arguments: WxB: %dx%d\n", ident, " ", width, batch_sz);
			if (isPerLayerTiming())
			{
				printf("%*s" "Performance: %6.2f ms\n", ident, " ", processing_time_ / iter);
			}
		}

		return(ret);

	}


	int aDNNodeSoftMax::RunHostFwd(void)
	{
		int ret = 0;

		std::string top_nm;
		top_nm = getOutputEdgeName();
		aDNNTensor & top_v = (isCrossEntrypyLoss()) ? getSlot(getBotDiffNm() + ADNN_VERIFY_NM) : getSlot(top_nm + ADNN_VERIFY_NM);
		aDNNTensor & bot = (aDNNTensor & )getBotFwd();

			int bot_stride = (int)bot.getStride(aDNN_TENSOR_WIDTH);
			int top_stride = (int)top_v.getStride(aDNN_TENSOR_WIDTH);

			int top_rows = (int)top_v.getDim(aDNN_TENSOR_BATCH);
			int top_cols = (int)top_v.getDim(aDNN_TENSOR_WIDTH);

			aDType * bot_ptr = (aDType *)bot.accessTensor(CL_MAP_READ);
			aDType * top_ptr = (aDType *)top_v.accessTensor(CL_MAP_WRITE);

			aDType* accum = new aDType[top_rows];
			// find max
			for (int j = 0; j < top_rows; ++j)
			{

				accum[j] = bot_ptr[j*bot_stride];
				for (int i = 1; i < top_cols; ++i)
				{

					accum[j] = std::max(accum[j], bot_ptr[j*bot_stride + i]);
				}
			}

			// substruct and exp
			for (int j = 0; j < top_rows; ++j)
			{

				for (int i = 0; i < top_cols; ++i)
				{
					aDType sub_val = bot_ptr[j*bot_stride + i] - accum[j];
					top_ptr[j*top_stride + i] = exp(sub_val);

				}
			}

			// sum up
			for (int j = 0; j < top_rows; ++j)
			{
				accum[j] = 0;
				for (int i = 0; i < top_cols; ++i)
				{
					accum[j] += top_ptr[j*top_stride + i];
				}
			}


			// divide
			for (int j = 0; j < top_rows; ++j)
			{
				aDType scaler = 1.f / accum[j];

				for (int i = 0; i < top_cols; ++i)
				{

					top_ptr[j*top_stride + i] *= scaler;

				}



			}

			if (isCrossEntrypyLoss())
			{
				// 2nd input edge
				aDNNTensor & labels = (aDNNTensor &)getBotFwd(1);
				aDType * label_ptr = (aDType *)labels.accessTensor(CL_MAP_READ);

				for (int j = 0; label_ptr && j < top_rows; ++j)
				{

					int index = (int)label_ptr[j];

					top_ptr[j*top_stride + index] -= 1;
					for (int i = 0; i < top_cols; ++i)
					{
						top_ptr[j*top_stride + i] /= top_rows;

					}


				}

				labels.commitTensor();

			}

			if (accum)
			{
				delete[] accum;
			}

			bot.commitTensor();
			top_v.commitTensor();



		return(ret);
	}

	int aDNNodeSoftMax::VerifyFwd(void)
	{
		int ret = 0;
		ret = RunHostFwd();
		std::string top_nm;
		top_nm = getOutputEdgeName();
		// get from the list of tensors referred by this node
		aDNNTensor & top_vf = (isCrossEntrypyLoss()) ? getSlot(getBotDiffNm() + ADNN_VERIFY_NM) : getSlot(top_nm + ADNN_VERIFY_NM);;

		aDNNTensor & top = (aDNNTensor &)((isCrossEntrypyLoss()) ? getBotDiff() : getTopFwd());

		aDType * top_ptr = (aDType *)top.accessTensor(CL_MAP_READ);

		aDType * top_vf_ptr = (aDType *)top_vf.accessTensor(CL_MAP_READ);

		int width = (int)top.getDim(aDNN_TENSOR_WIDTH);
		int batchs = (int)top.getDim(aDNN_TENSOR_BATCH);
		int top_v_batch_stride = (int)top_vf.getStride(aDNN_TENSOR_WIDTH);
		int top_batch_stride = (int)top.getStride(aDNN_TENSOR_WIDTH);


		double sqr_accum = 0;
		double max_err = -std::numeric_limits<double>::min();  // FLT_MIN 
		int max_b = 0, max_c = 0, max_i = 0, max_j = 0;

		const double allowedEps = 6;
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
			std::cout << "Error in softmax forward propagation: " << getName() + " " << std::fixed << std::setw(15) << std::setprecision(13) << sqr_accum <<
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
							std::cout << "Difference in softmax forward propagation: " << getName() + " " << err << " too large at " << b << ", " << i <<
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
			std::cout << "Passed varifier: forward propagation layer: softmax: " << getName() << std::endl;
		}
		return(ret);
	}


	/************************************************************************************************************************
	**
	**			FORWARD PROPAGATION
	**
	************************************************************************************************************************/

// TO DO : IS IT GENERAL CASE????
	int aDNNodeSoftMax::ConstructBwd(void)
	{
		int ret = 0;

		ret = aDNNode::ConstructBwd();

		if (isCrossEntrypyLoss() && getOutputEdgeType() == ADNN_ED_SINK)
		{
			ret = ConstructOptions();
		}

		return(ret);
	}


	int aDNNodeSoftMax::BuildBwd(void)
	{
		int ret = 0;

		aDNNode::BuildBwd();
		if (isCrossEntrypyLoss() && getOutputEdgeType() == ADNN_ED_SINK)
		{
			ret = BuildIntnl();
		}
		return(ret);
	}

	int aDNNodeSoftMax::RunBwd(const adnn_node_parameters * running_params)
	{
		int ret = 0;

//		RunFwd(running_params);

		return(ret);
	}

} // adnn






