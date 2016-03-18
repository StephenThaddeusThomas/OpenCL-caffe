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

void ActivationFunction(_FLOAT * res, const _FLOAT* data,
							_FLOAT power,
							_FLOAT alpha,
							_FLOAT beta)
{
#if		ADNN_NRN_OP_ID==ADNN_NEURON_PASTHRU
	ActivationFunction_PassThru(res, data);

#elif	ADNN_NRN_OP_ID==ADNN_NEURON_LOGISTIC
// 1/(1 + exp(-x))  
	ActivationFunction_Sigmoid(res, data);

#elif	ADNN_NRN_OP_ID==ADNN_NEURON_TANH
// (exp(2x) -1) / (exp(2x) + 1)
	ActivationFunction_TanH(res, data, alpha, beta);

#elif	ADNN_NRN_OP_ID==ADNN_NEURON_RELU
	ActivationFunction_ReLU(res, data, alpha);

#elif	ADNN_NRN_OP_ID==ADNN_NEURON_BRELU
	ActivationFunction_BReLU(res, data, alpha);

#elif	ADNN_NRN_OP_ID==ADNN_NEURON_SOFTRELU
//	log(1 + exp(x))
	ActivationFunction_BNLL(res, data);
#elif	ADNN_NRN_OP_ID==ADNN_NEURON_ABS
	ActivationFunction_Abs(res, data);

#elif	ADNN_NRN_OP_ID==ADNN_NEURON_SQUARE
	ActivationFunction_Square(res, data);
	
#elif	ADNN_NRN_OP_ID==ADNN_NEURON_SQR
	ActivationFunction_Sqrt(res, data);

#elif	ADNN_NRN_OP_ID==ADNN_NEURON_POWER
// (shift + scale * x ) ^power

	ActivationFunction_Power(res, data,
							power,
							alpha,
							beta);


#endif
}



__attribute__((reqd_work_group_size(ADNN_NRN_GROUP_SZ0,ADNN_NRN_GROUP_SZ1,ADNN_NRN_GROUP_SZ2)))
__kernel void aDNNNeuron4(
       const __global _FLOAT * bot,
       __global _FLOAT * top,
		_FLOAT power,
		_FLOAT scale,
		_FLOAT shift
	   )
{
	int x = get_global_id(0); // channel x

	_FLOAT data[4];
	_FLOAT response[4];

	*(_FLOAT4 *)data = *(__global _FLOAT4*)&bot[x*4];

	ActivationFunction((_FLOAT *)response,(const _FLOAT*)data, power, scale, shift);

	*(__global _FLOAT4*)&top[x*4] = *(_FLOAT4*)response;
}







__attribute__((reqd_work_group_size(ADNN_NRN_GROUP_SZ0,ADNN_NRN_GROUP_SZ1,ADNN_NRN_GROUP_SZ2)))
__kernel void aDNNNeuron4_Bwd(__global _FLOAT * bot_diff,
							__global  const _FLOAT* top_diff,
							__global const _FLOAT *bot_data,
							__global  const _FLOAT *top_data,
							_FLOAT diff_scale,
							_FLOAT power,
							_FLOAT scale,
							_FLOAT shift	   )
{
	int x = get_global_id(0); // channel x

	_FLOAT bot_diff4[4];
	_FLOAT top_diff4[4];
	_FLOAT bot_data4[4];
	_FLOAT top_data4[4];


#if		ADNN_NRN_OP_ID==ADNN_NEURON_RELU
{

	*(_FLOAT4*)top_diff4 = *(__global _FLOAT4*)&top_diff[x*4];
	*(_FLOAT4*)bot_data4 = *(__global _FLOAT4*)&bot_data[x*4];
	ActivationFunction_ReLU_Diff(bot_diff4, (const _FLOAT*)top_diff4, (const _FLOAT*)bot_data4, scale);
}
#elif	ADNN_NRN_OP_ID==ADNN_NEURON_LOGISTIC
// 1/(1 + exp(-x))  
	*(_FLOAT4*)top_diff4 = *(__global _FLOAT4*)&top_diff[x*4];
	*(_FLOAT4*)top_data4 = *(__global _FLOAT4*)&top_data[x*4];
	ActivationFunction_Sigmoid_Diff(bot_diff4, (const _FLOAT*)top_diff4, (const _FLOAT*)top_data4);
#elif	ADNN_NRN_OP_ID==ADNN_NEURON_TANH
// (exp(2x) -1) / (exp(2x) + 1)

	*(_FLOAT4*)top_diff4 = *(__global _FLOAT4*)&top_diff[x*4];
	*(_FLOAT4*)top_data4 = *(__global _FLOAT4*)&top_data[x*4];
	ActivationFunction_TanH_Diff(bot_diff4, (const _FLOAT*)top_diff4, (const _FLOAT*)top_data4);

#elif	ADNN_NRN_OP_ID==ADNN_NEURON_ABS

	*(_FLOAT4*)top_diff4 = *(__global _FLOAT4*)&top_diff[x*4];
	*(_FLOAT4*)bot_data4 = *(__global _FLOAT4*)&bot_data[x*4];

	ActivationFunction_Abs_Diff(bot_diff, (const _FLOAT*) top_diff4, (const _FLOAT *)bot_data4);
#elif	ADNN_NRN_OP_ID==ADNN_NEURON_POWER
// (shift + scale * x ) ^power

	*(_FLOAT4*)top_diff4 = *(__global _FLOAT4*)&top_diff[x*4];
	*(_FLOAT4*)top_data4 = *(__global _FLOAT4*)&top_data[x*4];
	*(_FLOAT4*)bot_data4 = *(__global _FLOAT4*)&bot_data[x*4];
	ActivationFunction_PowerDiff(bot_diff4, (const _FLOAT*) top_diff4, (const _FLOAT *) top_data4, (const _FLOAT *)bot_data4,
							diff_scale,	power, scale, shift);


#elif	ADNN_NRN_OP_ID==ADNN_NEURON_SOFTRELU
//	log(1 + exp(x))
	*(_FLOAT4*)top_diff4 = *(__global _FLOAT4*)&top_diff[x*4];
	*(_FLOAT4*)bot_data4 = *(__global _FLOAT4*)&bot_data[x*4];
	ActivationFunction_BNLL_Diff(bot_diff4, (const _FLOAT*) top_diff4, (const _FLOAT *)bot_data4);
#endif


	*(__global _FLOAT4*)&bot_diff[x *4] = *(_FLOAT4*)bot_diff4;
}



