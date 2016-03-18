// aDNNNeuronLayer.cpp

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
   **			aDNNodeNeuron Class
   **
   ************************************************************************************************************************/

  aDNNodeNeuron::aDNNodeNeuron(const ADNNBase & lib, const adnn_node_parameters & node_params)
		:aDNNode(lib, node_params)
  { }

  aDNNodeNeuron::aDNNodeNeuron(void) : aDNNode()  { }

  aDNNodeNeuron::aDNNodeNeuron(const aDNNodeNeuron & rh)  {  *this = rh;  }

  const aDNNode & aDNNodeNeuron:: operator = (const aDNNodeNeuron & rh)
  {
    *(aDNNode*)this = *(aDNNode*)&rh;
    return *this;
  }

  aDNNodeNeuron::~aDNNodeNeuron(void)	{ }

  int aDNNodeNeuron::Connect(void)
  {
    int ret = 0;
    return(ret);
  }

  int aDNNodeNeuron::Run(void)
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

  int aDNNodeNeuron::Construct(void)
  {
    int ret = 0;

    // to create internal system memory tensor for verification
    ConstructOutput();
    ConstructOptions();

    return(ret);
  }

  int aDNNodeNeuron::ConstructOptions(void)
  {
    int ret = 0;
    int type = getNeuronType();
    const aDNNTensor & bot = getBotFwd();
    size_t size = bot.getSize();

    if (((size / 4) * 4) != size)
      {
	printf("Error: buffer size is not multipel of 4.\n");
	ret = ADNN_GENERAL_FAILURE;
	return(ret);
      }

    size_t glbl_wk = size / 4;

    int ocl_group_sz0 = 256;
    int ocl_group_sz1 = 1;

    std::string comp_options = std::string(" -D ADNN_NRN_GROUP_SZ0=")   + std::to_string((long long)ocl_group_sz0)
			     + std::string(" -D ADNN_NRN_GROUP_SZ1=")   + std::to_string((long long)ocl_group_sz1)
			     + std::string(" -D ADNN_NRN_OP_ID=")       + std::to_string((long long)type)
			     + std::string(" -D ADNN_ACCEL=") + std::to_string((long long)ADNN_ACCEL_GPU)
			     + getGenericCompOptions();
    
    std::string kernel_file = "aDNNNeuron.cl";
    std::string kernel_name = "aDNNNeuron4";

    std::vector<size_t> l_wk;
    l_wk.push_back(ocl_group_sz0);
    l_wk.push_back(ocl_group_sz1);
    l_wk.push_back(1);

    std::vector<size_t> g_wk;
    g_wk.push_back(glbl_wk);
    g_wk.push_back(1);
    g_wk.push_back(1);

    CDNN_OCL_kern_exe kern_exe(this, kernel_name, kernel_file, comp_options, 0, &g_wk, &l_wk, 0);

    kern_exe.Construct();

    ocl_fwd_execs_.push_back(kern_exe);
    return (ret);
  }

  int aDNNodeNeuron::Build(void)
  {
    int ret = 0;

    aDNNode::Build();

    aDType power = 0;
    aDType scale = 0;
    aDType shift = 0;
    getNeuronArgs(power, scale, shift);
    const aDNNTensor & bot = getBotFwd();
    const aDNNTensor & top = getTopFwd();

    cl_mem bot_mem = bot.getOCLBuffer();

    cl_mem top_mem = top.getOCLBuffer();

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

    kern_args[n_arg++] = std::make_pair(sizeof(aDType), &power);
    kern_args[n_arg++] = std::make_pair(sizeof(aDType), &scale);
    kern_args[n_arg++] = std::make_pair(sizeof(aDType), &shift);
    
    kern_exe.Build(kern_args);
    return(ret);
  }

  int aDNNodeNeuron::RunFwd(const adnn_node_parameters * running_params)
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
	    additional_args[1] = std::make_pair(sizeof(cl_mem), &top_mem);
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
	

			int width = (int)top.getDim(aDNN_TENSOR_WIDTH);
			int height = (int)top.getDim(aDNN_TENSOR_HEIGHT);
			int outputs = (int)top.getDim(aDNN_TENSOR_DEPTH);


			int batch_sz = (int)bot.getDim(aDNN_TENSOR_BATCH);
			int inputs = (int)bot.getDim(aDNN_TENSOR_DEPTH);

			size_t size = bot.getSize();
			iter = (iter <= 0) ? 1 : iter;
			processing_time_ = subtractTimes(e, s);
			int ident = 4;
			printf("Passed layer: neuron: \"%s\"\n", getName().c_str());
			printf("%*s" "Arguments: CxWxHxOxB: %dx%dx%dx%dx%d\n", ident, " ", inputs, width, height, outputs, batch_sz);
			if (isPerLayerTiming())
			{
				printf("%*s" "Performance: %6.2f ms\n", ident, " ", processing_time_ / iter);
			}
		}
	



		return(ret);

	}


	int aDNNodeNeuron::RunHostFwd(void)
	{
		int ret = 0;

		return(ret);
	}

	int aDNNodeNeuron::VerifyFwd(void)
	{
		int ret = 0;
		ret = RunHostFwd();

		aDNNTensor & bot = (aDNNTensor &)getBotFwd();
		aDNNTensor & top = (aDNNTensor &)getTopFwd();

		aDType * bot_ptr = (aDType *)bot.accessTensor(ADNN_MEM_ACCESS_READ);
		aDType * top_ptr = (aDType *)top.accessTensor(ADNN_MEM_ACCESS_READ);
		size_t size = bot.getSize();



		aDType power = 0;
		aDType scale = 0;
		aDType shift = 0;
		getNeuronArgs(power, scale, shift);

		int match = 1;
		const double allowedEps = 6;

		for (size_t i = 0; i < size / 4 && match; i++)
		{
			aDType c_res[4];
			const aDType * data = &bot_ptr[i * 4];
			switch (getNeuronType())
			{
			case ADNN_NEURON_PASTHRU:		//	x	
				ActivationFunction_PassThru(c_res, data);
				break;
			case ADNN_NEURON_LOGISTIC:	//	1 / (1 + e^-x)	//Sigmoid
				ActivationFunction_Sigmoid(c_res, data);
				break;
			case ADNN_NEURON_TANH:		//	a * tanh( b * x)
				ActivationFunction_TanH(c_res, data, shift, scale);
				break;
			case ADNN_NEURON_RELU:		//	max(0, x)
				ActivationFunction_ReLU(c_res, data, scale);
				break;
			case ADNN_NEURON_BRELU:		//	min(a, max(0, x))
				ActivationFunction_BReLU(c_res, data, shift);
				break;
			case ADNN_NEURON_SOFTRELU:	//	log(1 + e^x)   // bonomial normal log likelihood
				ActivationFunction_BNLL(c_res, data);
				break;
			case ADNN_NEURON_ABS:			//	abs(x)
				ActivationFunction_Abs(c_res, data);
				break;
			case ADNN_NEURON_SQUARE:		//	x^2
				ActivationFunction_Square(c_res, data);
				break;
			case ADNN_NEURON_SQR:			//	sqr(x)
				ActivationFunction_Sqrt(c_res, data);
				break;
			case ADNN_NEURON_LINEAR:		//	a + b *x
				ActivationFunction_Linear(c_res, data, shift, scale);
				break;
			case ADNN_NEURON_POWER:		// (a + b * x ) ^power
				ActivationFunction_Power(c_res, data, power, shift, scale);
				break;
			default:
				printf("ERROR: unknown neuron tyoe: %d\n", getNeuronType());
				break;
			}
			const aDType * g_res = &top_ptr[i * 4];
			for (int k = 0; k < 4; k++)
			{
				aDType c_val = c_res[k];
				aDType g_val = g_res[k];
				double err = CalculateErr(c_val, g_val);

				if (err > allowedEps || std::isnan(c_val) || std::isnan(g_val) || !std::isfinite(c_val) || !std::isfinite(g_val))
				{
					std::cout << "Difference in neuron layer: " << getName() + " " << err << " too large at " << i * 4 + k << " c_v = " << c_val << " vs g_val = " << g_val << std::endl;
					match = 0;
				}
			}

		}

		if (match)
		{
			std::cout << "Passed varifier: layer: neuron: " << getName() << std::endl;
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

	int aDNNodeNeuron::ConstructBwd(void)
	{
		int ret = 0;
		ret = aDNNode::ConstructBwd();


		const aDNNTensor & bot_df = getBotDiff();

		size_t size = bot_df.getSize();
		if (((size / 4) * 4) != size)
		{
			printf("Error: buffer size is not multipel of 4.\n");
			ret = ADNN_GENERAL_FAILURE;
			return(ret);
		}

		int type = getNeuronType();

		size_t glbl_wk = size / 4;

		int ocl_group_sz0 = 256;
		int ocl_group_sz1 = 1;



		std::string comp_options =
			std::string(" -D ADNN_NRN_GROUP_SZ0=") + std::to_string((long long)ocl_group_sz0)
			+ std::string(" -D ADNN_NRN_GROUP_SZ1=") + std::to_string((long long)ocl_group_sz1)
			+ std::string(" -D ADNN_NRN_OP_ID=") + std::to_string((long long)type)
			+ std::string(" -D ADNN_ACCEL=") + std::to_string((long long)ADNN_ACCEL_GPU)
			+ getGenericCompOptions()
			;
		std::string kernel_file = "aDNNNeuron.cl";
		std::string kernel_name = "aDNNNeuron4_Bwd";

		std::vector<size_t> l_wk;
		l_wk.push_back(ocl_group_sz0);
		l_wk.push_back(ocl_group_sz1);
		l_wk.push_back(1);


		std::vector<size_t> g_wk;
		g_wk.push_back(glbl_wk);
		g_wk.push_back(1);
		g_wk.push_back(1);

		CDNN_OCL_kern_exe kern_exe(this, kernel_name, kernel_file, comp_options, 0, &g_wk, &l_wk, 0);

		kern_exe.Construct();

		ocl_bwd_execs_.push_back(kern_exe);

		return(ret);
	}


	int aDNNodeNeuron::BuildBwd(void)
	{
		int ret = 0;
		ret = aDNNode::BuildBwd();
		const aDNNTensor & bot = getBotFwd();
		const aDNNTensor & top = getTopFwd();
		const aDNNTensor & bot_df = getBotDiff();
		const aDNNTensor & top_df = getTopDiff();
		aDType power = 0;
		aDType scale = 0;
		aDType shift = 0;
		getNeuronArgs(power, scale, shift);
		aDType diff_scale = scale * power;
		cl_mem bot_mem = bot.getOCLBuffer();
		cl_mem top_mem = top.getOCLBuffer();
		cl_mem bot_df_mem = bot_df.getOCLBuffer();
		cl_mem top_df_mem = top_df.getOCLBuffer();

		CDNN_OCL_kern_exe & kern_exe = ocl_bwd_execs_[0];
		int n_arg = 0;
		ocl_args kern_args;

		kern_args[n_arg++] = std::make_pair(sizeof(cl_mem), &bot_df_mem);
		kern_args[n_arg++] = std::make_pair(sizeof(cl_mem), &top_df_mem);
		kern_args[n_arg++] = std::make_pair(sizeof(cl_mem), &bot_mem);
		kern_args[n_arg++] = std::make_pair(sizeof(cl_mem), &top_mem);

		kern_args[n_arg++] = std::make_pair(sizeof(aDType), &diff_scale);
		kern_args[n_arg++] = std::make_pair(sizeof(aDType), &power);
		kern_args[n_arg++] = std::make_pair(sizeof(aDType), &scale);
		kern_args[n_arg++] = std::make_pair(sizeof(aDType), &shift);

		kern_exe.Build(kern_args);

		return(ret);

	}

	int aDNNodeNeuron::RunBwd(const adnn_node_parameters * running_params)
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
			ret = ocl_bwd_execs_[0].ExecuteNoWait(NULL);
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

			const aDNNTensor & bot_df = getBotDiff();
			const aDNNTensor & top_df = getTopDiff();

			int out_width = (int)bot_df.getDim(aDNN_TENSOR_WIDTH);
			int out_height = (int)bot_df.getDim(aDNN_TENSOR_HEIGHT);

			int in_width = (int)top_df.getDim(aDNN_TENSOR_WIDTH);
			int in_height = (int)top_df.getDim(aDNN_TENSOR_HEIGHT);

			int inputs = (int)top_df.getDim(aDNN_TENSOR_DEPTH);
			int outputs = (int)bot_df.getDim(aDNN_TENSOR_DEPTH);
			int batch_sz = (int)bot_df.getDim(aDNN_TENSOR_BATCH);

			iter = (iter <= 0) ? 1 : iter;
			processing_time_ = subtractTimes(e, s);
			int ident = 4;
			printf("Passed layer: neuron back-propagation: \"%s\"\n", getName().c_str());
			printf("%*s" "Arguments: CxWxHxOxB: %dx%dx%dx%dx%d\n", ident, " ", inputs, in_width, in_height, outputs, batch_sz);
			if (isPerLayerTiming())
			{
				printf("%*s" "Performance: %6.2f ms\n", ident, " ", processing_time_ / iter);
			}
		}

		return(ret);

	}

	int aDNNodeNeuron::RunHostBwd(void)
	{
		int ret = 0;


		return(ret);

	}

	int aDNNodeNeuron::VerifyBwd(void)
	{
		int ret = 0;
		ret = RunHostBwd();

		aDNNTensor & bot = (aDNNTensor & )getBotFwd();
		aDNNTensor & top = (aDNNTensor &)getTopFwd();
		aDNNTensor & bot_df = (aDNNTensor &)getBotDiff();
		aDNNTensor & top_df = (aDNNTensor &)getTopDiff();
		aDType power = 0;
		aDType scale = 0;
		aDType shift = 0;
		getNeuronArgs(power, scale, shift);
		aDType diff_scale = scale * power;


		aDType * bot_ptr = (aDType *)bot.accessTensor(ADNN_MEM_ACCESS_READ);
		aDType * top_ptr = (aDType *)top.accessTensor(ADNN_MEM_ACCESS_READ);
		aDType * bot_df_ptr = (aDType *)bot_df.accessTensor(ADNN_MEM_ACCESS_READ);
		aDType * top_df_ptr = (aDType *)top_df.accessTensor(ADNN_MEM_ACCESS_READ);
		size_t size = bot.getSize();

		int neuron_type = getNeuronType();

		int match = 1;
		if (neuron_type == ADNN_NEURON_RELU)
		{
			const double allowedEps = 3;

			for (size_t i = 0; i < size / 4 && match; i++)
			{
				aDType bot_df_v_p[4];
				ActivationFunction_ReLU_Diff(bot_df_v_p, &top_df_ptr[i * 4], &bot_ptr[i * 4], scale);
				const aDType * bot_df_p = &bot_df_ptr[i * 4];
				for (int k = 0; k < 4; k++)
				{
					aDType c_val = bot_df_v_p[k];
					aDType g_val = bot_df_p[k];
					double err = CalculateErr(c_val, g_val);

					if (err > allowedEps || std::isnan(c_val) || std::isnan(g_val))
					{
						std::cout << "Difference in neuron back-propagation: " << getName() + " " << err << " too large at " << i * 4 + k << " c_v = " << c_val << " vs g_val = " << g_val << std::endl;
						match = 0;
					}
				}



			}
		}
		else if (neuron_type == ADNN_NEURON_LOGISTIC)
		{
			// 1/(1 + exp(-x))  
			printf("Neuron: ERROR: bwd func %d has not been implemented yet\n", neuron_type);
		}
		else if (neuron_type == ADNN_NEURON_TANH)
		{
			// (exp(2x) -1) / (exp(2x) + 1)
			printf("Neuron: ERROR: bwd func %d has not been implemented yet\n", neuron_type);
		}
		else if (neuron_type == ADNN_NEURON_ABS)
		{
			printf("Neuron: ERROR: bwd func %d has not been implemented yet\n", neuron_type);
		}
		else if (neuron_type == ADNN_NEURON_POWER)
		{
			// (shift + scale * x ) ^power
			printf("Neuron: ERROR: bwd func %d has not been implemented yet\n", neuron_type);
		}
		else if (neuron_type == ADNN_NEURON_SOFTRELU)
		{
			//	log(1 + exp(x))
			printf("Neuron: ERROR: bwd func %d has not been implemented yet\n", neuron_type);
		}
		else
		{
			printf("Neuron: ERROR: unknown bwd func %d\n", neuron_type);
		}

		bot.commitTensor();
		top.commitTensor();
		bot_df.commitTensor();
		top_df.commitTensor();

		if (match)
		{
			std::cout << "Passed varifier: layer: neuron back-propagation: " << getName() << std::endl;
		}

		return (ret);
	}



} // adnn








