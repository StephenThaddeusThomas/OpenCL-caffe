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
   **			aDNNodeLRN Class
   **
   ************************************************************************************************************************/

  aDNNodeLRN::aDNNodeLRN(const ADNNBase & lib, const adnn_node_parameters & node_params)
    :aDNNode(lib, node_params) { }
  
  aDNNodeLRN::aDNNodeLRN(void)	: aDNNode() { }

  aDNNodeLRN::aDNNodeLRN(const aDNNodeLRN & rh)
  {
    *this = rh;
  }

  const aDNNode & aDNNodeLRN:: operator = (const aDNNodeLRN & rh)
  {
    *(aDNNode*)this = *(aDNNode*)&rh;
    return *this;
  }

  aDNNodeLRN::~aDNNodeLRN(void)   { }

  int aDNNodeLRN::Connect(void)
  {
    int ret = 0;
    return(ret);
  }

  int aDNNodeLRN::Run(void)
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

  int aDNNodeLRN::Construct(void)
  {
    int ret = 0;

    // to create internal system memory tensor for verification
    ConstructOutput();
    ConstructOptions();
    return(ret);
  }

  int aDNNodeLRN::ConstructOptions(void)
  {
    int ret = 0;

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

    int height_out = (int)top.getDim(aDNN_TENSOR_HEIGHT);
    int width_out = (int)top.getDim(aDNN_TENSOR_WIDTH);
    int outputs = (int)top.getDim(aDNN_TENSOR_DEPTH);
    int batch = (int)top.getDim(aDNN_TENSOR_BATCH);
    
    // addition scale buffer for backpropagation compute saving
    const aDNNTensor & scale = cloneSlot(getTopNm() + ADNN_SCALE_NM, top);

    // tensor for the host verification
    if (getDebugLevel() == 1)
      {
	cloneSlot(getTopNm() + ADNN_SCALE_NM + ADNN_VERIFY_NM, scale);
      }

    int scale_stride = (int)scale.getStride(aDNN_TENSOR_WIDTH);
    int scale_channel_stride = (int)scale.getStride(aDNN_TENSOR_HEIGHT);
    int	scale_batch_stride = (int)scale.getStride(aDNN_TENSOR_DEPTH);

    ADNN_LRN_REGION norm_reg = getNormRegion();
    int local_area = getLocalArea();
    int pre_pad = (local_area - 1) / 2;
    int pad = local_area - pre_pad - 1;

    int top_df_stride = 1;
    int top_df_channel_stride = 1;
    int	top_df_batch_stride = 1;

    int bot_df_stride = 1;
    int bot_df_channel_stride = 1;
    int	bot_df_batch_stride = 1;

    int n_out_pix_horiz = 1;
    int n_out_pix_vert = 1;
    int ocl_group_sz0 = 8;
    int ocl_group_sz1 = 8;

    if (norm_reg == ADNN_LRN_ACROSS_CHANNELS)
      {
	ocl_group_sz0 = (top_width <= 8) ? 8 : 16;
	ocl_group_sz1 = (top_height <= 8) ? 8 : 16;
      }
    else
      {
	n_out_pix_horiz = (top_width <= 8) ? 1 : (top_width <= 16) ? 2 : 4;
	n_out_pix_vert = (top_height <= 8) ? 1 : (top_height <= 16) ? 2 : 4;;
      }
    
    int ocl_group_lg2sz0 = (int)ceil(log((double)ocl_group_sz0) / log(2.));
    int ocl_group_lg2sz1 = (int)ceil(log((double)ocl_group_sz1) / log(2.));

    std::string comp_options = std::string(" -D ADNN_LRN_KERNEL_SZ=") + std::to_string((long long)local_area)
                             + std::string(" -D ADNN_LRN_N_OUTPUTS=") + std::to_string((long long)outputs)
                             + std::string(" -D ADNN_LRN_N_CHANNELS=") + std::to_string((long long)inputs)
			     + std::string(" -D ADNN_LRN_PAD=") + std::to_string((long long)pad)
			     + std::string(" -D ADNN_LRN_N_HORIZ_OUT_PIX=") + std::to_string((long long)n_out_pix_horiz)
			     + std::string(" -D ADNN_LRN_N_VERT_OUT_PIX=") + std::to_string((long long)n_out_pix_vert)
			     + std::string(" -D ADNN_LRN_GROUP_SZ0=") + std::to_string((long long)ocl_group_sz0)
			     + std::string(" -D ADNN_LRN_GROUP_SZ1=") + std::to_string((long long)ocl_group_sz1)
			     + std::string(" -D ADNN_LRN_GROUP_LG2SZ0=") + std::to_string((long long)ocl_group_lg2sz0)
			     + std::string(" -D ADNN_LRN_GROUP_LG2SZ1=") + std::to_string((long long)ocl_group_lg2sz1)
			     + std::string(" -D ADNN_LRN_BOT_BATCH_STRIDE=") + std::to_string((long long)bot_batch_stride)
			     + std::string(" -D ADNN_LRN_BOT_CHANNEL_STRIDE=") + std::to_string((long long)bot_channel_stride)
			     + std::string(" -D ADNN_LRN_BOT_STRIDE=") + std::to_string((long long)bot_stride)
			     + std::string(" -D ADNN_LRN_TOP_BATCH_STRIDE=") + std::to_string((long long)top_batch_stride)
			     + std::string(" -D ADNN_LRN_TOP_CHANNEL_STRIDE=") + std::to_string((long long)top_channel_stride)
			     + std::string(" -D ADNN_LRN_TOP_STRIDE=") + std::to_string((long long)top_stride)
			     + std::string(" -D ADNN_LRN_BOT_WIDTH=") + std::to_string((long long)bot_width)
			     + std::string(" -D ADNN_LRN_BOT_HEIGHT=") + std::to_string((long long)bot_height)
			     + std::string(" -D ADNN_LRN_TOP_WIDTH=") + std::to_string((long long)top_width)
			     + std::string(" -D ADNN_LRN_TOP_HEIGHT=") + std::to_string((long long)top_height)
			     + std::string(" -D ADNN_LRN_SCALE_BATCH_STRIDE=") + std::to_string((long long)scale_batch_stride)
			     + std::string(" -D ADNN_LRN_SCALE_CHANNEL_STRIDE=") + std::to_string((long long)scale_channel_stride)
			     + std::string(" -D ADNN_LRN_SCALE_STRIDE=") + std::to_string((long long)scale_stride)
			     + std::string(" -D ADNN_LRN_TOPDF_BATCH_STRIDE=") + std::to_string((long long)top_df_batch_stride)
			     + std::string(" -D ADNN_LRN_TOPDF_CHANNEL_STRIDE=") + std::to_string((long long)top_df_channel_stride)
			     + std::string(" -D ADNN_LRN_TOPDF_STRIDE=") + std::to_string((long long)top_df_stride)
			     + std::string(" -D ADNN_LRN_BOTDF_BATCH_STRIDE=") + std::to_string((long long)bot_df_batch_stride)
			     + std::string(" -D ADNN_LRN_BOTDF_CHANNEL_STRIDE=") + std::to_string((long long)bot_df_channel_stride)
			     + std::string(" -D ADNN_LRN_BOTDF_STRIDE=") + std::to_string((long long)bot_df_stride)
			     + std::string(" -D ADNN_LRN_BATCH_SZ=") + std::to_string((long long)batch)
			     + getGenericCompOptions();

    std::string kernel_file = "aDNNLRN.cl";
    std::string kernel_name = (norm_reg == ADNN_LRN_ACROSS_CHANNELS) ? "aDNNLRNAcrossChannels1" : "aDNNLRNWithinChannel";

    std::vector<size_t> l_wk;

    l_wk.push_back(ocl_group_sz0);
    l_wk.push_back(ocl_group_sz1);
    l_wk.push_back(1);

    std::vector<size_t> g_wk;

    if (getNormRegion() == ADNN_LRN_ACROSS_CHANNELS)
      {
	g_wk.push_back(top_width);
	g_wk.push_back(top_height);
	g_wk.push_back(batch);
      }
    else
      {
	int g_wk_width = (int)((top_width + ocl_group_sz0 * n_out_pix_horiz - 1) / (ocl_group_sz0 * n_out_pix_horiz));
	int g_wk_height = (int)((top_height + ocl_group_sz1 * n_out_pix_vert - 1) / (ocl_group_sz1 * n_out_pix_vert));
	
	g_wk.push_back(g_wk_width * ocl_group_sz0);
	g_wk.push_back(g_wk_height * ocl_group_sz1);
	g_wk.push_back(outputs * batch);
      }

    CDNN_OCL_kern_exe kern_exe(this, kernel_name, kernel_file, comp_options, 0, &g_wk, &l_wk, 0);

    kern_exe.Construct();

    ocl_fwd_execs_.push_back(kern_exe);

    return (ret);
  }

  int aDNNodeLRN::Build(void)
  {
    int ret = 0;

    aDNNode::Build();

    int local_area = getLocalArea();
    aDType alpha = (aDType)getAlpha();

    // whithin channel alphaoverarea is going to be culculate based on actual areal size (cut by borders).
    aDType alphaoverarea = (aDType)((getNormRegion() == ADNN_LRN_ACROSS_CHANNELS) ? alpha / local_area : alpha / (local_area * local_area));
    aDType beta = (aDType)getBeta();

    const aDNNTensor & bot = getBotFwd();
    const aDNNTensor & top = getTopFwd();

    // take from the list of tensors referred by this node
    aDNNTensor & scale = getSlot(getTopNm() + ADNN_SCALE_NM);

    // allocate scale buffers
    scale.allocTensor();

    if (getDebugLevel() == 1)
      {
	// add to the list of tensors referred by this node
	aDNNTensor & scale_vr = getSlot(getTopNm() + ADNN_SCALE_NM + ADNN_VERIFY_NM);
	scale_vr.allocTensor(_CBUF_MEM_SYS_ONLY);
      }

    cl_mem bot_mem = bot.getOCLBuffer();
    cl_mem top_mem = top.getOCLBuffer();
    cl_mem scale_mem = scale.getOCLBuffer();

    // pass all arguments once
    // memory has to be allocated outside of the pipeline by the user

    CDNN_OCL_kern_exe & kern_exe = ocl_fwd_execs_[0];
    int n_arg = 0;
    ocl_args kern_args;
    if (bot_mem)
      {
	kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &bot_mem);
      }
    n_arg++;			// TT: ?? do we want this in the { } above

    if (top_mem)
      {
	kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &top_mem);
      }
    n_arg++;			// TT: ?? same as above

    if (scale_mem)
      {
	kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &scale_mem);
      }
    n_arg++;			// TT: ?? I guess its ok to have 'gaps' in kern_args

    kern_args[n_arg++] = std::make_pair(sizeof(aDType), &alphaoverarea);
    kern_args[n_arg++] = std::make_pair(sizeof(aDType), &alpha);
    kern_args[n_arg++] = std::make_pair(sizeof(aDType), &beta);

    kern_exe.Build(kern_args);

    return(ret);
  }

  int aDNNodeLRN::RunFwd(const adnn_node_parameters * running_params)
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
			printf("Passed layer: LRN-%s: \"%s\"\n",
				(getNormRegion() == ADNN_LRN_ACROSS_CHANNELS) ? "across channels" : "within channel",
				getName().c_str() );
			printf("%*s" "Arguments: CxWxHxOxB: %dx%dx%dx%dx%d\n", ident, " ", inputs, width, height, outputs, batch_sz);
			if (isPerLayerTiming())
			{
				printf("%*s" "Performance: %6.2f ms\n", ident, " ", processing_time_ / iter);
			}
		}

		return(ret);

	}


	int aDNNodeLRN::RunHostFwd(void)
	{
		int ret = 0;

		aDNNTensor & bot = (aDNNTensor &)getBotFwd();
		aDNNTensor & top = (aDNNTensor &)getTopFwd();
		// take from the list of tensors referred by this node
		aDNNTensor & scale_v = getSlot(getTopNm() + ADNN_SCALE_NM + ADNN_VERIFY_NM);

		std::string top_nm;
		aDNNTensor & top_v = getSlot(getTopNm() + ADNN_VERIFY_NM);


		aDType * bot_ptr = (aDType *)bot.accessTensor(ADNN_MEM_ACCESS_READ);
		aDType * top_ptr = (aDType *)top.accessTensor(ADNN_MEM_ACCESS_READ);
		aDType * scale_v_ptr = (aDType *)scale_v.accessTensor(ADNN_MEM_ACCESS_WRITE);
		aDType * top_v_ptr = (aDType *)top_v.accessTensor(ADNN_MEM_ACCESS_READ);

		int local_area = getLocalArea();
		int pre_pad = (local_area - 1) / 2;
		int pad = local_area - pre_pad - 1;

		aDType alphaoverarea = (aDType)((getNormRegion() == ADNN_LRN_ACROSS_CHANNELS) ? getAlpha() / local_area : getAlpha() / (local_area * local_area));
		aDType beta = (aDType)getBeta();

		int outputs = (int)top.getDim(aDNN_TENSOR_DEPTH);
		int batch = (int)top.getDim(aDNN_TENSOR_BATCH);
		int bot_width = (int)bot.getDim(aDNN_TENSOR_WIDTH);
		int bot_height = (int)bot.getDim(aDNN_TENSOR_HEIGHT);
		int top_width = (int)top.getDim(aDNN_TENSOR_WIDTH);
		int top_height = (int)top.getDim(aDNN_TENSOR_HEIGHT);
		int inputs = (int)bot.getDim(aDNN_TENSOR_DEPTH);
		int bot_stride = (int)bot.getStride(aDNN_TENSOR_WIDTH);
		int bot_channel_stride = (int)bot.getStride(aDNN_TENSOR_HEIGHT);
		int bot_batch_stride = (int)bot.getStride(aDNN_TENSOR_DEPTH);

		int scale_v_stride = (int)scale_v.getStride(aDNN_TENSOR_WIDTH);
		int scale_v_channel_stride = (int)scale_v.getStride(aDNN_TENSOR_HEIGHT);
		int scale_v_batch_stride = (int)scale_v.getStride(aDNN_TENSOR_DEPTH);

		int top_v_stride = (int)top_v.getStride(aDNN_TENSOR_WIDTH);
		int top_v_channel_stride = (int)top_v.getStride(aDNN_TENSOR_HEIGHT);
		int top_v_batch_stride = (int)top_v.getStride(aDNN_TENSOR_DEPTH);

		int top_stride = (int)top.getStride(aDNN_TENSOR_WIDTH);
		int top_channel_stride = (int)top.getStride(aDNN_TENSOR_HEIGHT);
		int	top_batch_stride = (int)top.getStride(aDNN_TENSOR_DEPTH);


		if (bot_ptr && top_ptr && scale_v_ptr && top_v_ptr)
		{

			if (getNormRegion() == ADNN_LRN_ACROSS_CHANNELS)
			{


				for (int b = 0; b < batch; b++)
				{
					for (int j = 0; j < top_height; j++)
					{
						for (int i = 0; i < top_width; i++)
						{
							// c-emulator
							aDType res = 0;
							aDType accum_scale = 0;
							int head = 0;
							aDType bot_val;
							while (head < pad) {
								bot_val = bot_ptr[b*bot_batch_stride + head * bot_channel_stride + j * bot_stride + i];
								accum_scale += bot_val  * bot_val;
								++head;
							}
							// until we reach size, nothing needs to be subtracted
							while (head < local_area) {
								bot_val = bot_ptr[b*bot_batch_stride + head * bot_channel_stride + j * bot_stride + i];
								accum_scale += bot_val  * bot_val;
								aDType scale = (aDType)1. + accum_scale * alphaoverarea;
								scale_v_ptr[b*scale_v_batch_stride + (head - pad) * scale_v_channel_stride + j * scale_v_stride + i] = scale;
								bot_val = bot_ptr[b*bot_batch_stride + (head - pad) * bot_channel_stride + j * bot_stride + i];
								aDType s = pow(scale, -beta);
								aDType c_val = bot_val * s;
								top_v_ptr[b*top_v_batch_stride + (head - pad) * top_v_channel_stride + j * top_v_stride + i] = c_val;
								++head;
							}
							// both add and subtract
							while (head < inputs) {
								bot_val = bot_ptr[b*bot_batch_stride + head * bot_channel_stride + j * bot_stride + i];
								accum_scale += bot_val  * bot_val;
								bot_val = bot_ptr[b*bot_batch_stride + (head - local_area) * bot_channel_stride + j * bot_stride + i];
								accum_scale -= bot_val  * bot_val;
								aDType scale = (aDType)1. + accum_scale * alphaoverarea;
								scale_v_ptr[b*scale_v_batch_stride + (head - pad) * scale_v_channel_stride + j * scale_v_stride + i] = scale;
								aDType s = pow(scale, -beta);
								bot_val = bot_ptr[b*bot_batch_stride + (head - pad) * bot_channel_stride + j * bot_stride + i];
								aDType c_val = bot_val * s;
								top_v_ptr[b*top_v_batch_stride + (head - pad) * top_v_channel_stride + j * top_v_stride + i] = c_val;
								++head;
							}
							// subtract only
							while (head < inputs + pad) {
								bot_val = bot_ptr[b*bot_batch_stride + (head - local_area) * bot_channel_stride + j * bot_stride + i];
								accum_scale -= bot_val  * bot_val;
								aDType scale = (aDType)1. + accum_scale * alphaoverarea;
								scale_v_ptr[b*scale_v_batch_stride + (head - pad) * scale_v_channel_stride + j * scale_v_stride + i] = scale;
								bot_val = bot_ptr[b*bot_batch_stride + (head - pad) * bot_channel_stride + j * bot_stride + i];
								aDType s = pow(scale, -beta);
								aDType c_val = bot_val * s;
								top_v_ptr[b*top_v_batch_stride + (head - pad) * top_v_channel_stride + j * top_v_stride + i] = c_val;
								++head;
							}

						}
					}
				}
			}
			else
			{


				for (int b = 0; b < batch; b++)
				{
					for (int o = 0; o < outputs; o++)
					{
						for (int j = 0; j < top_height; j++)
						{
							for (int i = 0; i < top_width; i++)
							{
								// c-emulator
								aDType scale = 0;
								int hstart = j - pad;
								int wstart = i - pad;
								int hend = std::min(hstart + local_area, bot_height + pad);
								int wend = std::min(wstart + local_area, bot_width + pad);
								int adj_area_size = (hend - hstart) * (wend - wstart);
								hstart = std::max(hstart, 0);
								wstart = std::max(wstart, 0);
								hend = std::min(hend, bot_height);
								wend = std::min(wend, bot_width);
								aDType accum = 0;
								for (int h = hstart; h < hend; ++h)
								{
									for (int w = wstart; w < wend; ++w)
									{
#if 0
										if (b == 0 && o == 0 && j == 0 && i == 0)
										{

											printf("c:%d %d   %f %f\n", i, j, res, bot_ptr[b*bot_batch_stride + o * bot_channel_stride + h * bot_stride + w]);
										}
#endif
										aDType bot_val = bot_ptr[b*bot_batch_stride + o * bot_channel_stride + h * bot_stride + w];
										accum += bot_val * bot_val;

									}
								}

								alphaoverarea = (aDType)getAlpha() / adj_area_size;
								scale = (aDType)1. + accum* alphaoverarea;

								aDType s = pow(scale, -beta);
								aDType bot_val = bot_ptr[b*bot_batch_stride + o * bot_channel_stride + j * bot_stride + i];
								aDType c_val = bot_val * s;
#if 0
								if (i == 9 && j == 4 && o == 0)
								{
									printf("C:lrn: %13.11f %13.11f %13.11f %13.11f\n", c_val, bot_val, s, scale);
								}
#endif

								top_v_ptr[b*top_batch_stride + o * top_channel_stride + j * top_stride + i] = c_val;

							}
						}
					}
				} // (getNormRegion() == ACROSS_CHANNELS)

			}



		}

		top.commitTensor();
		bot.commitTensor();
		scale_v.commitTensor();
		top_v.commitTensor();

		return(ret);
	}

	int aDNNodeLRN::VerifyFwd(void)
	{
		int ret = 0;
		ret = RunHostFwd();

		aDNNTensor & top = (aDNNTensor &)getTopFwd();

		aDNNTensor & top_v = getSlot(getTopNm() + ADNN_VERIFY_NM);


		aDType * top_ptr = (aDType *)top.accessTensor(ADNN_MEM_ACCESS_READ);
		aDType * top_v_ptr = (aDType *)top_v.accessTensor(ADNN_MEM_ACCESS_READ);

		int outputs = (int)top.getDim(aDNN_TENSOR_DEPTH);
		int batch = (int)top.getDim(aDNN_TENSOR_BATCH);

		int top_width = (int)top.getDim(aDNN_TENSOR_WIDTH);
		int top_height = (int)top.getDim(aDNN_TENSOR_HEIGHT);



		int top_v_stride = (int)top_v.getStride(aDNN_TENSOR_WIDTH);
		int top_v_channel_stride = (int)top_v.getStride(aDNN_TENSOR_HEIGHT);
		int top_v_batch_stride = (int)top_v.getStride(aDNN_TENSOR_DEPTH);

		int top_stride = (int)top.getStride(aDNN_TENSOR_WIDTH);
		int top_channel_stride = (int)top.getStride(aDNN_TENSOR_HEIGHT);
		int	top_batch_stride = (int)top.getStride(aDNN_TENSOR_DEPTH);

		double sqr_accum = 0;
		double max_err = -std::numeric_limits<double>::min();
		int max_b = 0, max_o = 0, max_i = 0, max_j = 0;

		for (int b = 0; b < batch; b++)
		{
			for (int o = 0; o < outputs; o++)
			{
				for (int j = 0; j < top_height; j++)
				{
					for (int i = 0; i < top_width; i++)
					{

						aDType c_val = top_v_ptr[b*top_v_batch_stride + o * top_v_channel_stride + j * top_v_stride + i];
						aDType g_val = top_ptr[b*top_batch_stride + o * top_channel_stride + j * top_stride + i];
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

		sqr_accum = sqrt(sqr_accum / ((double)batch *outputs * top_height *top_width));

		int match = 1;

		if (sqr_accum > 0 || std::isnan(sqr_accum) || !std::isfinite(sqr_accum))
		{
			std::cout << "Error in LRN forward propagation: " << getName() + " : " << std::fixed << std::setw(15) << std::setprecision(13) << sqr_accum <<
				" Max err: " << std::fixed << std::setw(15) << std::setprecision(13) << max_err << " at " << max_b << ", " << max_o << ", " << max_i << ", " << max_j << std::endl;

			if (sqr_accum > 1. / 1000000000)
			{

				double allowedEps = 4;

				for (int b = 0; b < batch && match; b++)
				{
					for (int o = 0; o < outputs && match; o++)
					{
						for (int j = 0; j < top_height && match; j++)
						{
							for (int i = 0; i < top_width && match; i++)
							{

								aDType c_val = top_v_ptr[b*top_v_batch_stride + o * top_v_channel_stride + j * top_v_stride + i];
								aDType g_val = top_ptr[b*top_batch_stride + o * top_channel_stride + j * top_stride + i];
								double err = CalculateErr(c_val, g_val);
								if (err > allowedEps || std::isnan(c_val) || std::isnan(g_val) || !std::isfinite(c_val) || !std::isfinite(g_val))
								{
									std::cout << "Difference in LRN forward propagation: " << getName() + " " << err << " too large at " << b << ", " << o << ", " << i << ", " << j << " c_v = " << c_val << " vs g_val = " << g_val << std::endl;
									match = 0;
								}



							}
						}
					}
				}

			}
		}
		if (match)
		{
			std::cout << "Passed varifier: forward propagation layer: LRN: " << getName() << std::endl;
		}


		top.commitTensor();

		top_v.commitTensor();

		return(ret);
	}



	/************************************************************************************************************************
	**
	**			BACKWARD PROPAGATION
	**
	************************************************************************************************************************/

	int aDNNodeLRN::ConstructBwd(void)
	{
		int ret = 0;
		ret = aDNNode::ConstructBwd();


		const aDNNTensor & bot = getBotFwd();
		const aDNNTensor & top = getTopFwd();
		// take from the list of tensors referred by this node
		aDNNTensor & scale = getSlot(getTopNm() + ADNN_SCALE_NM);
		const aDNNTensor & bot_df = getBotDiff();
		const aDNNTensor & top_df = getTopDiff();

		int outputs = (int)top.getDim(aDNN_TENSOR_DEPTH);
		int inputs = (int)bot.getDim(aDNN_TENSOR_DEPTH);
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

		int scale_stride = (int)scale.getStride(aDNN_TENSOR_WIDTH);
		int scale_channel_stride = (int)scale.getStride(aDNN_TENSOR_HEIGHT);
		int	scale_batch_stride = (int)scale.getStride(aDNN_TENSOR_DEPTH);

		int top_df_stride = (int)top_df.getStride(aDNN_TENSOR_WIDTH);
		int top_df_channel_stride = (int)top_df.getStride(aDNN_TENSOR_HEIGHT);
		int	top_df_batch_stride = (int)top_df.getStride(aDNN_TENSOR_DEPTH);

		int bot_df_width = (int)bot_df.getDim(aDNN_TENSOR_WIDTH);
		int bot_df_height = (int)bot_df.getDim(aDNN_TENSOR_HEIGHT);
		int bot_df_stride = (int)bot_df.getStride(aDNN_TENSOR_WIDTH);
		int bot_df_channel_stride = (int)bot_df.getStride(aDNN_TENSOR_HEIGHT);
		int	bot_df_batch_stride = (int)bot_df.getStride(aDNN_TENSOR_DEPTH);


		ADNN_LRN_REGION norm_reg = getNormRegion();
		int local_area = getLocalArea();
		int pre_pad = (local_area - 1) / 2;
		int pad = local_area - pre_pad - 1;

		aDType alpha = (aDType)getAlpha();
		aDType beta = (aDType)getBeta();

		int n_out_pix_horiz;
		int n_out_pix_vert;
		int ocl_group_sz0;
		int ocl_group_sz1;

		if (norm_reg == ADNN_LRN_ACROSS_CHANNELS)
		{
			n_out_pix_horiz = 1;
			n_out_pix_vert = 1;
			ocl_group_sz0 = (bot_df_width <= 8) ? 8 : 16;
			ocl_group_sz1 = (bot_df_height <= 8) ? 8 : 16;

		}
		else
		{
			ocl_group_sz0 = 8;
			ocl_group_sz1 = 8;

			n_out_pix_horiz = (bot_df_width <= 8) ? 1 : (bot_df_width <= 16) ? 2 : 4;
			n_out_pix_vert = (bot_df_height <= 8) ? 1 : (bot_df_height <= 16) ? 2 : 4;;
		}
		int ocl_group_lg2sz0 = (int)ceil(log((double)ocl_group_sz0) / log(2.));
		int ocl_group_lg2sz1 = (int)ceil(log((double)ocl_group_sz1) / log(2.));


		std::string comp_options =
			std::string(" -D ADNN_LRN_KERNEL_SZ=") + std::to_string((long long)local_area)
			+ std::string(" -D ADNN_LRN_N_OUTPUTS=") + std::to_string((long long)outputs)
			+ std::string(" -D ADNN_LRN_N_CHANNELS=") + std::to_string((long long)inputs)
			+ std::string(" -D ADNN_LRN_PAD=") + std::to_string((long long)pad)
			+ std::string(" -D ADNN_LRN_N_HORIZ_OUT_PIX=") + std::to_string((long long)n_out_pix_horiz)
			+ std::string(" -D ADNN_LRN_N_VERT_OUT_PIX=") + std::to_string((long long)n_out_pix_vert)
			+ std::string(" -D ADNN_LRN_GROUP_SZ0=") + std::to_string((long long)ocl_group_sz0)
			+ std::string(" -D ADNN_LRN_GROUP_SZ1=") + std::to_string((long long)ocl_group_sz1)
			+ std::string(" -D ADNN_LRN_GROUP_LG2SZ0=") + std::to_string((long long)ocl_group_lg2sz0)
			+ std::string(" -D ADNN_LRN_GROUP_LG2SZ1=") + std::to_string((long long)ocl_group_lg2sz1)
			+ std::string(" -D ADNN_LRN_BOT_BATCH_STRIDE=") + std::to_string((long long)bot_batch_stride)
			+ std::string(" -D ADNN_LRN_BOT_CHANNEL_STRIDE=") + std::to_string((long long)bot_channel_stride)
			+ std::string(" -D ADNN_LRN_BOT_STRIDE=") + std::to_string((long long)bot_stride)
			+ std::string(" -D ADNN_LRN_TOP_BATCH_STRIDE=") + std::to_string((long long)top_batch_stride)
			+ std::string(" -D ADNN_LRN_TOP_CHANNEL_STRIDE=") + std::to_string((long long)top_channel_stride)
			+ std::string(" -D ADNN_LRN_TOP_STRIDE=") + std::to_string((long long)top_stride)
			+ std::string(" -D ADNN_LRN_BOT_WIDTH=") + std::to_string((long long)bot_width)
			+ std::string(" -D ADNN_LRN_BOT_HEIGHT=") + std::to_string((long long)bot_height)
			+ std::string(" -D ADNN_LRN_TOP_WIDTH=") + std::to_string((long long)top_width)
			+ std::string(" -D ADNN_LRN_TOP_HEIGHT=") + std::to_string((long long)top_height)
			+ std::string(" -D ADNN_LRN_SCALE_BATCH_STRIDE=") + std::to_string((long long)scale_batch_stride)
			+ std::string(" -D ADNN_LRN_SCALE_CHANNEL_STRIDE=") + std::to_string((long long)scale_channel_stride)
			+ std::string(" -D ADNN_LRN_SCALE_STRIDE=") + std::to_string((long long)scale_stride)
			+ std::string(" -D ADNN_LRN_TOPDF_BATCH_STRIDE=") + std::to_string((long long)top_df_batch_stride)
			+ std::string(" -D ADNN_LRN_TOPDF_CHANNEL_STRIDE=") + std::to_string((long long)top_df_channel_stride)
			+ std::string(" -D ADNN_LRN_TOPDF_STRIDE=") + std::to_string((long long)top_df_stride)
			+ std::string(" -D ADNN_LRN_BOTDF_BATCH_STRIDE=") + std::to_string((long long)bot_df_batch_stride)
			+ std::string(" -D ADNN_LRN_BOTDF_CHANNEL_STRIDE=") + std::to_string((long long)bot_df_channel_stride)
			+ std::string(" -D ADNN_LRN_BOTDF_STRIDE=") + std::to_string((long long)bot_df_stride)
			+ std::string(" -D ADNN_LRN_BATCH_SZ=") + std::to_string((long long)n_batchs)



			+ getGenericCompOptions()
			;
		std::string kernel_file = "aDNNLRN.cl";
		std::string kernel_name;
		std::vector<size_t> l_wk;
		std::vector<size_t> g_wk;
		l_wk.push_back(ocl_group_sz0);
		l_wk.push_back(ocl_group_sz1);
		l_wk.push_back(1);

		if (norm_reg == ADNN_LRN_ACROSS_CHANNELS)
		{
			g_wk.push_back(bot_df.getDim(aDNN_TENSOR_WIDTH));
			g_wk.push_back(bot_df.getDim(aDNN_TENSOR_HEIGHT));
			g_wk.push_back(bot_df.getDim(aDNN_TENSOR_BATCH));
			kernel_name = "aDNNLRNAcrossChannelsBwd1";
		}
		else
		{
			int g_wk_width = (int)((bot_df.getDim(aDNN_TENSOR_WIDTH) + ocl_group_sz0 * n_out_pix_horiz - 1) / (ocl_group_sz0 * n_out_pix_horiz));
			int g_wk_height = (int)((bot_df.getDim(aDNN_TENSOR_HEIGHT) + ocl_group_sz1 * n_out_pix_vert - 1) / (ocl_group_sz1 * n_out_pix_vert));

			g_wk.push_back(g_wk_width * ocl_group_sz0);
			g_wk.push_back(g_wk_height * ocl_group_sz1);
			g_wk.push_back(bot_df.getDim(aDNN_TENSOR_DEPTH) * bot_df.getDim(aDNN_TENSOR_BATCH));
			kernel_name = "aDNNLRNWithinChannelBwd";

		}

		CDNN_OCL_kern_exe kern_exe_tr(this, kernel_name, kernel_file, comp_options, 0, &g_wk, &l_wk, 0);

		kern_exe_tr.Construct();

		ocl_bwd_execs_.push_back(kern_exe_tr);



		return(ret);
	}


	int aDNNodeLRN::BuildBwd(void)
	{
		int ret = 0;


		ret = aDNNode::BuildBwd();


		const aDNNTensor & bot = getBotFwd();
		const aDNNTensor & top = getTopFwd();

		// take from the list of tensors referred by this node
		aDNNTensor & scale = getSlot(getTopNm() + ADNN_SCALE_NM);

		const aDNNTensor & bot_df = getBotDiff();
		const aDNNTensor & top_df = getTopDiff();

		cl_mem bot_mem = bot.getOCLBuffer();
		cl_mem top_mem = top.getOCLBuffer();
		cl_mem scale_mem = scale.getOCLBuffer();
		cl_mem bot_df_mem = bot_df.getOCLBuffer();
		cl_mem top_df_mem = top_df.getOCLBuffer();


		ADNN_LRN_REGION norm_reg = getNormRegion();
		int local_area = getLocalArea();
		int pre_pad = (local_area - 1) / 2;
		int pad = local_area - pre_pad - 1;

		aDType alpha = (aDType)getAlpha();
		aDType beta = (aDType)getBeta();


		aDType ratio_dta_bwd = (aDType) 2. * alpha * beta / local_area;



		// pass all arguments once
		int n_arg = 0;
		ocl_args kern_args;
		kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &top_mem);
		n_arg++;
		kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &bot_mem);
		n_arg++;
		kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &top_df_mem);
		n_arg++;
		kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &scale_mem);
		n_arg++;
		kern_args[n_arg] = std::make_pair(sizeof(cl_mem), &bot_df_mem);
		n_arg++;
		kern_args[n_arg] = std::make_pair(sizeof(aDType), &ratio_dta_bwd);
		n_arg++;
		kern_args[n_arg] = std::make_pair(sizeof(aDType), &alpha);
		n_arg++;
		kern_args[n_arg] = std::make_pair(sizeof(aDType), &beta);


		CDNN_OCL_kern_exe & kern0_exe = ocl_bwd_execs_[0];

		kern0_exe.Build(kern_args);

		return(ret);

	}

	int aDNNodeLRN::RunBwd(const adnn_node_parameters * running_params)
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

			int inputs = (int)bot_df.getDim(aDNN_TENSOR_DEPTH);
			int outputs = (int)top_df.getDim(aDNN_TENSOR_DEPTH);
			int batch_sz = (int)bot_df.getDim(aDNN_TENSOR_BATCH);

			iter = (iter <= 0) ? 1 : iter;
			processing_time_ = subtractTimes(e, s);
			int ident = 4;
			printf("Passed layer: LRN back-propagation: \"%s\"\n", getName().c_str());
			printf("%*s" "Arguments: CxWxHxOxB: %dx%dx%dx%dx%d\n", ident, " ", inputs, in_width, in_height, outputs, batch_sz);
			if (isPerLayerTiming())
			{
				printf("%*s" "Performance: %6.2f ms\n", ident, " ", processing_time_ / iter);
			}

		}
		return(ret);

	}

	int aDNNodeLRN::RunHostBwd(void)
	{
		int ret = 0;

		aDNNTensor & bot = (aDNNTensor & )getBotFwd();
		aDNNTensor & top = (aDNNTensor &)getTopFwd();
		// take from the list of tensors referred by this node
		aDNNTensor & scale = getSlot(getTopNm() + ADNN_SCALE_NM);
		aDNNTensor & top_df = (aDNNTensor &)getTopDiff();
		aDNNTensor & bot_df_v = getSlot(getBotDiffNm() + ADNN_VERIFY_NM);



		aDType * bot_ptr = (aDType *)bot.accessTensor(ADNN_MEM_ACCESS_READ);
		aDType * top_ptr = (aDType *)top.accessTensor(ADNN_MEM_ACCESS_READ);
		aDType * scale_ptr = (aDType *)scale.accessTensor(ADNN_MEM_ACCESS_READ);
		aDType * bot_df_v_ptr = (aDType *)bot_df_v.accessTensor(ADNN_MEM_ACCESS_WRITE);
		aDType * top_df_ptr = (aDType *)top_df.accessTensor(ADNN_MEM_ACCESS_READ);


		int outputs = (int)top.getDim(aDNN_TENSOR_DEPTH);
		int inputs = (int)bot.getDim(aDNN_TENSOR_DEPTH);
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

		int scale_stride = (int)scale.getStride(aDNN_TENSOR_WIDTH);
		int scale_channel_stride = (int)scale.getStride(aDNN_TENSOR_HEIGHT);
		int	scale_batch_stride = (int)scale.getStride(aDNN_TENSOR_DEPTH);

		int top_df_stride = (int)top_df.getStride(aDNN_TENSOR_WIDTH);
		int top_df_channel_stride = (int)top_df.getStride(aDNN_TENSOR_HEIGHT);
		int	top_df_batch_stride = (int)top_df.getStride(aDNN_TENSOR_DEPTH);

		int bot_df_v_width = (int)bot_df_v.getDim(aDNN_TENSOR_WIDTH);
		int bot_df_v_height = (int)bot_df_v.getDim(aDNN_TENSOR_HEIGHT);
		int bot_df_v_stride = (int)bot_df_v.getStride(aDNN_TENSOR_WIDTH);
		int bot_df_v_channel_stride = (int)bot_df_v.getStride(aDNN_TENSOR_HEIGHT);
		int	bot_df_v_batch_stride = (int)bot_df_v.getStride(aDNN_TENSOR_DEPTH);


		ADNN_LRN_REGION norm_reg = getNormRegion();
		int local_area = getLocalArea();
		int pre_pad = (local_area - 1) / 2;
		int pad = local_area - pre_pad - 1;

		aDType alpha = (aDType)getAlpha();
		aDType beta = (aDType)getBeta();



		aDType negative_beta = -beta;

		if (norm_reg == ADNN_LRN_ACROSS_CHANNELS)
		{

			aDType ratio_dta_bwd = (aDType) 2. * alpha * beta / local_area;

			for (int b = 0; b < n_batchs; b++)
			{
				for (int j = 0; j < bot_height; j++)
				{
					for (int i = 0; i < bot_width; i++)
					{

						// c-emulator
						int head = 0;
						aDType accum_ratio = 0;
						// accumulate values
						while (head < pad) {

							aDType adder = (top_df_ptr[b*top_df_batch_stride + head * top_df_channel_stride + j * top_df_stride + i]
								* top_ptr[b*top_batch_stride + head * top_channel_stride + j * top_stride + i])
								/ scale_ptr[b*scale_batch_stride + head * scale_channel_stride + j * scale_stride + i];

#if 0
							if (i == 5 && j == 11/* && (head - pad) == 12 */ && b == 10)
							{
								printf("C:a %d %f %f\n",
									head,
									accum_ratio,
									adder
									);
							}
#endif


							accum_ratio += adder;


							++head;
						}
						// until we reach size, nothing needs to be subtracted
						while (head < local_area) {

							aDType adder = (top_df_ptr[b*top_df_batch_stride + head * top_df_channel_stride + j * top_df_stride + i]
								* top_ptr[b*top_batch_stride + head * top_channel_stride + j * top_stride + i])
								/ scale_ptr[b*scale_batch_stride + head * scale_channel_stride + j * scale_stride + i];

#if 0
							if (i == 5 && j == 11/* && (head - pad) == 12 */ && b == 10)
							{
								printf("C:a %d %f %f\n",
									head,
									accum_ratio,
									adder
									);
							}
#endif


							accum_ratio += adder;


							bot_df_v_ptr[b*bot_df_v_batch_stride + (head - pad) * bot_df_v_channel_stride + j * bot_df_v_stride + i] =
								top_df_ptr[b*top_df_batch_stride + (head - pad) * top_df_channel_stride + j * top_df_stride + i]
								* pow(scale_ptr[b*scale_batch_stride + (head - pad) * scale_channel_stride + j * scale_stride + i], negative_beta)
								- ratio_dta_bwd * bot_ptr[b*bot_batch_stride + (head - pad) * bot_channel_stride + j * bot_stride + i] * accum_ratio;

							++head;
						}
						// both add and subtract
						while (head < inputs) {

							aDType adder = top_df_ptr[b*top_df_batch_stride + head * top_df_channel_stride + j * top_df_stride + i]
								* top_ptr[b*top_batch_stride + head * top_channel_stride + j * top_stride + i]
								/ scale_ptr[b*scale_batch_stride + head * scale_channel_stride + j * scale_stride + i];

#if 0
							if (i == 5 && j == 11/* && (head - pad) == 12 */ && b == 10)
							{
								printf("C:a %d %f %f\n",
									head,
									accum_ratio,
									adder
									);
							}
#endif

							accum_ratio += adder;

							aDType subs = (top_df_ptr[b*top_df_batch_stride + (head - local_area) * top_df_channel_stride + j * top_df_stride + i]
								* top_ptr[b*top_batch_stride + (head - local_area) * top_channel_stride + j * top_stride + i])
								/ scale_ptr[b*scale_batch_stride + (head - local_area) * scale_channel_stride + j * scale_stride + i];



							accum_ratio -= subs;

#if 0
							if (i == 3 && j == 0/* && (head - pad) == 12 */ && b == 3)
							{
								printf("C: %d %16.12f %16.12f %16.12f %16.12f %16.12f\n",
									head,
									accum_ratio,
									adder,
									top_df_ptr[b*top_df_batch_stride + head * top_df_channel_stride + j * top_df_stride + i],
									top_ptr[b*top_batch_stride + head * top_channel_stride + j * top_stride + i],
									scale_ptr[b*scale_batch_stride + head * scale_channel_stride + j * scale_stride + i]
									);
							}
#endif


							bot_df_v_ptr[b*bot_df_v_batch_stride + (head - pad) * bot_df_v_channel_stride + j * bot_df_v_stride + i] =
								top_df_ptr[b*top_df_batch_stride + (head - pad) * top_df_channel_stride + j * top_df_stride + i]
								* pow(scale_ptr[b*scale_batch_stride + (head - pad) * scale_channel_stride + j * scale_stride + i], negative_beta)
								- ratio_dta_bwd * bot_ptr[b*bot_batch_stride + (head - pad) * bot_channel_stride + j * bot_stride + i] * accum_ratio;


							++head;
						}
						// subtract only
						while (head < inputs + pad) {

							aDType subs = (top_df_ptr[b*top_df_batch_stride + (head - local_area) * top_df_channel_stride + j * top_df_stride + i]
								* top_ptr[b*top_batch_stride + (head - local_area) * top_channel_stride + j * top_stride + i])
								/ scale_ptr[b*scale_batch_stride + (head - local_area) * scale_channel_stride + j * scale_stride + i];

							accum_ratio -= subs;






							bot_df_v_ptr[b*bot_df_v_batch_stride + (head - pad) * bot_df_v_channel_stride + j * bot_df_v_stride + i] =
								top_df_ptr[b*top_df_batch_stride + (head - pad) * top_df_channel_stride + j * top_df_stride + i]
								* pow(scale_ptr[b*scale_batch_stride + (head - pad) * scale_channel_stride + j * scale_stride + i], negative_beta)
								- ratio_dta_bwd * bot_ptr[b*bot_batch_stride + (head - pad) * bot_channel_stride + j * bot_stride + i] * accum_ratio;


#if 0
							if (i == 5 && j == 11/* && (head - pad) == 12 */ && b == 10)
							{
								printf("C: %d %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f\n",
									head,
									bot_df_v_ptr[b*bot_df_v_batch_stride + (head - pad) * bot_df_v_channel_stride + j * bot_df_v_stride + i],
									top_df_ptr[b*top_df_batch_stride + (head - pad) * top_df_channel_stride + j * top_df_stride + i],
									pow(scale_ptr[b*scale_batch_stride + (head - pad) * scale_channel_stride + j * scale_stride + i], negative_beta),
									scale_ptr[b*scale_batch_stride + (head - pad) * scale_channel_stride + j * scale_stride + i],
									-ratio_dta_bwd_ * bot_ptr[b*bot_batch_stride + (head - pad) * bot_channel_stride + j * bot_stride + i] * accum_ratio,
									bot_ptr[b*bot_batch_stride + (head - pad) * bot_channel_stride + j * bot_stride + i],
									accum_ratio
									);
							}
#endif

							++head;
						}


					}
				}
			}
		}
		else
		{
			for (int b = 0; b < n_batchs; b++)
			{
				for (int o = 0; o < inputs; o++)
				{
					for (int j = 0; j < bot_height; j++)
					{

						for (int i = 0; i < bot_width; i++)
						{
							aDType accum_ratio = 0;

							int hstart = j - pad;
							int wstart = i - pad;
							int hend = std::min(hstart + local_area, top_height + pad);
							int wend = std::min(wstart + local_area, top_width + pad);
							int adj_area_size = (hend - hstart) * (wend - wstart);
							hstart = std::max(hstart, 0);
							wstart = std::max(wstart, 0);
							hend = std::min(hend, top_height);
							wend = std::min(wend, top_width);
							for (int h = hstart; h < hend; ++h)
							{
								for (int w = wstart; w < wend; ++w)
								{
									aDType adder = top_df_ptr[b*top_df_batch_stride + o * top_df_channel_stride + h * top_df_stride + w]
										* top_ptr[b*top_batch_stride + o * top_channel_stride + h * top_stride + w]
										/ scale_ptr[b*scale_batch_stride + o * scale_channel_stride + h * scale_stride + w];

									accum_ratio += adder;

								}
							}

							aDType ratio_dta_bwd = (aDType) 2. * alpha * beta / adj_area_size;

							bot_df_v_ptr[b*bot_df_v_batch_stride + o * bot_df_v_channel_stride + j * bot_df_v_stride + i] =
								top_df_ptr[b*top_df_batch_stride + o * top_df_channel_stride + j * top_df_stride + i]
								* pow(scale_ptr[b*scale_batch_stride + o * scale_channel_stride + j * scale_stride + i], negative_beta)
								- ratio_dta_bwd * bot_ptr[b*bot_batch_stride + o * bot_channel_stride + j * bot_stride + i] * accum_ratio;

						}
					}
				}
			}


		}

		bot.commitTensor();
		top.commitTensor();
		scale.commitTensor();
		bot_df_v.commitTensor();
		top_df.commitTensor();

		return(ret);

	}

	int aDNNodeLRN::VerifyBwd(void)
	{
		int ret = 0;
		ret = RunHostBwd();

		aDNNTensor & bot_df = (aDNNTensor & )getBotDiff();

		aDNNTensor & bot_df_v = getSlot(getBotDiffNm() + ADNN_VERIFY_NM);

		aDType * bot_df_v_ptr = (aDType *)bot_df_v.accessTensor(ADNN_MEM_ACCESS_READ);
		aDType * bot_df_ptr = (aDType *)bot_df.accessTensor(ADNN_MEM_ACCESS_READ);

		int n_outputs = (int)bot_df.getDim(aDNN_TENSOR_DEPTH);
		int n_batchs = (int)bot_df.getDim(aDNN_TENSOR_BATCH);

		int bot_df_width = (int)bot_df.getDim(aDNN_TENSOR_WIDTH);
		int bot_df_height = (int)bot_df.getDim(aDNN_TENSOR_HEIGHT);
		int bot_df_stride = (int)bot_df.getStride(aDNN_TENSOR_WIDTH);
		int bot_df_channel_stride = (int)bot_df.getStride(aDNN_TENSOR_HEIGHT);
		int	bot_df_batch_stride = (int)bot_df.getStride(aDNN_TENSOR_DEPTH);
		int bot_df_v_stride = (int)bot_df_v.getStride(aDNN_TENSOR_WIDTH);
		int bot_df_v_channel_stride = (int)bot_df_v.getStride(aDNN_TENSOR_HEIGHT);
		int	bot_df_v_batch_stride = (int)bot_df_v.getStride(aDNN_TENSOR_DEPTH);

		double sqr_accum = 0;
		double max_err = -std::numeric_limits<double>::min();
		int max_b = 0, max_o = 0, max_i = 0, max_j = 0;

		for (int b = 0; b < n_batchs; b++)
		{
			for (int o = 0; o < n_outputs; o++)
			{
				for (int j = 0; j < bot_df_height; j++)
				{
					for (int i = 0; i < bot_df_width; i++)
					{
						aDType c_val = bot_df_v_ptr[b*bot_df_v_batch_stride + o * bot_df_v_channel_stride + j * bot_df_v_stride + i];
						aDType g_val = bot_df_ptr[b*bot_df_batch_stride + o * bot_df_channel_stride + j * bot_df_stride + i];
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

		sqr_accum = sqrt(sqr_accum / ((double)n_batchs *n_outputs * bot_df_height *bot_df_width));

		int match = 1;

		if (sqr_accum > 0 || std::isnan(sqr_accum) || !std::isfinite(sqr_accum))
		{
			std::cout << "Error in LRN back-propagation " << getName() + " : " << std::fixed << std::setw(15) << std::setprecision(13) << sqr_accum <<
				" Max err: " << std::fixed << std::setw(15) << std::setprecision(13) << max_err << " at " << max_b << ", " << max_o << ", " << max_i << ", " << max_j << std::endl;

			if (sqr_accum > 1. / 1000000000)
			{
				double allowedEps = 4;

				for (int b = 0; b < n_batchs && match; b++)
				{
					for (int o = 0; o < n_outputs && match; o++)
					{
						for (int j = 0; j < bot_df_height && match; j++)
						{
							for (int i = 0; i < bot_df_width && match; i++)
							{
								aDType c_val = bot_df_v_ptr[b*bot_df_v_batch_stride + o * bot_df_v_channel_stride + j * bot_df_v_stride + i];
								aDType g_val = bot_df_ptr[b*bot_df_batch_stride + o * bot_df_channel_stride + j * bot_df_stride + i];
								double err = CalculateErr(c_val, g_val);
								if (err > allowedEps || std::isnan(c_val) || std::isnan(g_val) || !std::isfinite(c_val) || !std::isfinite(g_val))
								{
									std::cout << "Difference in LRN back propagation: " << getName() + " " << err << " too large at " << b << ", " << o << ", " << i << ", " << j << " c_v = " << c_val << " vs g_val = " << g_val << std::endl;
									match = 0;
								}

							}
						}
					}
				}
			}
		}

		bot_df_v.commitTensor();
		bot_df.commitTensor();

		if (match)
		{
			std::cout << "Passed varifier: layer: LRN back-propagation: " << getName() << std::endl;
		}


		return (ret);
	}




} // adnn






