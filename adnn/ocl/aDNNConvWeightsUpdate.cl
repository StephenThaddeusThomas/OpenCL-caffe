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


#include "aDNN.cl.h"

#define ADNN_GROUP_SZ2 1


__attribute__((reqd_work_group_size(ADNN_GROUP_SZ0,ADNN_GROUP_SZ1,ADNN_GROUP_SZ2)))
__kernel void aDNNConvWeightUpdateSGD(
		const __global _FLOAT * weights_df,
		__global _FLOAT * weights,
		__global _FLOAT * weights_hist,
		_FLOAT momentum,
		_FLOAT weights_rate,
		_FLOAT weights_decay,
		_FLOAT bias_rate,
		_FLOAT bias_decay
	   )
{
	int w_p = get_global_id(0); // weight/bias
	int o_p = get_global_id(1); // output id
	_FLOAT rate = (w_p == ADNN_CONV_BIAS_POS) ? bias_rate : weights_rate;
	_FLOAT decay = (w_p == ADNN_CONV_BIAS_POS) ? bias_decay : weights_decay;
	_FLOAT we_hist = weights_hist[o_p * ADNN_CONV_WEIGHTS_DF_HIST_STRIDE + w_p];
	_FLOAT we_df = weights_df[o_p * ADNN_CONV_WEIGHTS_DF_STRIDE + w_p];
	_FLOAT we = weights[o_p * ADNN_CONV_WEIGHTS_STRIDE + w_p];
	_FLOAT we_hist_out;
	_FLOAT we_out;
	annCalculateWeightsUpdate(	&we_hist_out, &we_out,
 								we_hist, we_df, we,
								momentum,
								rate, decay
							);

#if 0
	if (o_p == 0 && w_p == 0)
	{
		printf("K:wet: %13.11f %13.11f %13.11f %13.11f %13.11f %13.11f %13.11f %13.11f\n",
			we_hist_out, we_out,
			we_hist, we_df, we,
			momentum,
			rate, decay

		);
	}

#endif
	weights_hist[o_p * ADNN_CONV_WEIGHTS_DF_HIST_STRIDE + w_p] = we_hist_out;
	weights[o_p * ADNN_CONV_WEIGHTS_STRIDE + w_p] = we_out;
}

__attribute__((reqd_work_group_size(ADNN_GROUP_SZ0,ADNN_GROUP_SZ1,ADNN_GROUP_SZ2)))
__kernel void aDNNWeightUpdateSGD(
		const __global _FLOAT * weights_df,
		__global _FLOAT * weights,
		__global _FLOAT * weights_hist,
		_FLOAT momentum,
		_FLOAT weights_rate,
		_FLOAT weights_decay
	   )
{
	int w_p = get_global_id(0); // weight/bias
	int o_p = get_global_id(1); // output id
	_FLOAT rate = weights_rate;
	_FLOAT decay = weights_decay;
	_FLOAT we_hist = weights_hist[o_p * ADNN_CONV_WEIGHTS_DF_HIST_STRIDE + w_p];
	_FLOAT we_df = weights_df[o_p * ADNN_CONV_WEIGHTS_DF_STRIDE + w_p];
	_FLOAT we = weights[o_p * ADNN_CONV_WEIGHTS_STRIDE + w_p];
	_FLOAT we_hist_out;
	_FLOAT we_out;
	annCalculateWeightsUpdate(	&we_hist_out, &we_out,
 								we_hist, we_df, we,
								momentum,
								rate, decay
							);

#if 0
	if (o_p == 0 && w_p == 0)
	{
		printf("K:wet: %13.11f %13.11f %13.11f %13.11f %13.11f %13.11f %13.11f %13.11f\n",
			we_hist_out, we_out,
			we_hist, we_df, we,
			momentum,
			rate, decay

		);
	}

#endif

	weights_hist[o_p * ADNN_CONV_WEIGHTS_DF_HIST_STRIDE + w_p] = we_hist_out;
	weights[o_p * ADNN_CONV_WEIGHTS_STRIDE + w_p] = we_out;
}
