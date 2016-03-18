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

#ifndef _ADNN_NEURON_CL_H_
#define _ADNN_NEURON_CL_H_


#if ADNN_ACCEL == ADNN_ACCEL_GPU
#define ADNN_NEURON_PASTHRU		0  //x	
#define ADNN_NEURON_LOGISTIC	ADNN_NEURON_PASTHRU + 1		//	1 / (1 + e^-x)	//Sigmoid
#define ADNN_NEURON_TANH		ADNN_NEURON_LOGISTIC + 1	//	a * tanh( b * x)
#define ADNN_NEURON_RELU		ADNN_NEURON_TANH + 1		//	max(0, x)
#define ADNN_NEURON_BRELU		ADNN_NEURON_RELU + 1		//	min(a, max(0, x))
#define ADNN_NEURON_SOFTRELU	ADNN_NEURON_BRELU + 1		//	log(1 + e^x)   // bonomial normal log likelihood
#define ADNN_NEURON_ABS			ADNN_NEURON_SOFTRELU + 1	//	abs(x)
#define ADNN_NEURON_SQUARE		ADNN_NEURON_ABS + 1			//	x^2
#define ADNN_NEURON_SQR			ADNN_NEURON_SQUARE + 1		//	sqr(x)
#define ADNN_NEURON_LINEAR		ADNN_NEURON_SQR	+ 1			//	a + b * x
#define ADNN_NEURON_POWER		ADNN_NEURON_LINEAR + 1		// (a + b * x ) ^power
#define ADNN_NEURON_TOTAL		ADNN_NEURON_POWER + 1
#endif


#define ADNN_NRN_GROUP_SZ2 1
#ifndef ADNN_NRN_OP_ID
#define ADNN_NRN_OP_ID ADNN_NEURON_PASTHRU
#endif

inline
void ActivationFunction_PassThru(_FLOAT * res, const _FLOAT* data)
{
	for (int i = 0; i <4; i++)
	{
		res[i] = data[i];
	}
}


inline
void ActivationFunction_ReLU(_FLOAT * res, const _FLOAT* data,
							_FLOAT slope)
{

	res[0] = (data[0] > 0) ? data[0] : data[0] * slope;	
	res[1] = (data[1] > 0) ? data[1] : data[1] * slope;	
	res[2] = (data[2] > 0) ? data[2] : data[2] * slope;	
	res[3] = (data[3] > 0) ? data[3] : data[3] * slope;	
}

inline
void ActivationFunction_BReLU(_FLOAT * res, const _FLOAT* data, _FLOAT alpha)
{

	res[0] = (_FLOAT)fmin(alpha, fmax(data[0], 0));;
	res[1] = (_FLOAT)fmin(alpha, fmax(data[1], 0));;
	res[2] = (_FLOAT)fmin(alpha, fmax(data[2], 0));;
	res[3] = (_FLOAT)fmin(alpha, fmax(data[3], 0));;
}

inline
void ActivationFunction_Sigmoid(_FLOAT * res, const _FLOAT* data)
{
	for(int i = 0; i <4; i++)
	{
// 1/(1 + exp(-x))  
		res[i] = (1.f + exp(-data[i]));
	}
}


inline
void ActivationFunction_TanH(_FLOAT * res, const _FLOAT* data, _FLOAT alpha, _FLOAT beta)
{
	for (int i = 0; i <4; i++)
	{
		// (exp(2x) -1) / (exp(2x) + 1)
		res[i] = alpha* tanh(beta * data[i]);
	}
}
inline
void ActivationFunction_Abs(_FLOAT * res, const _FLOAT* data)
{
	for(int i = 0; i <4; i++)
	{
		res[i] = fabs(data[i]); 
	}
}

inline
void ActivationFunction_Square(_FLOAT * res, const _FLOAT* data)
{
	for (int i = 0; i <4; i++)
	{
	
		res[i] = data[i] * data[i];
	}
}

inline
void ActivationFunction_Sqrt(_FLOAT * res, const _FLOAT* data)
{
	for (int i = 0; i <4; i++)
	{

		res[i] = sqrt(data[i]);
	}
}

inline
void ActivationFunction_Linear(_FLOAT * res, const _FLOAT* data, _FLOAT alpha, _FLOAT beta)
{
	for (int i = 0; i <4; i++)
	{
		// (exp(2x) -1) / (exp(2x) + 1)
		res[i] = alpha + beta * data[i];
	}
}

inline
void ActivationFunction_Power(_FLOAT * res, const _FLOAT* data,
							_FLOAT power,
							_FLOAT alpha,
							_FLOAT beta)
{
	for(int i = 0; i <4; i++)
	{
// (shift + scale * x ) ^power
		_FLOAT arg = alpha + data[i] * beta;
		_FLOAT run_arg = (arg == 0) ? 1 : arg;
		res[i] = (arg == 0) ? 0 : pow(run_arg, power);

	}
}

inline
void ActivationFunction_BNLL(_FLOAT * res, const _FLOAT* data)

{
	for(int i = 0; i <4; i++)
	{
//	log(1 + exp(x))
		res[i] = log(1.f + exp(data[i]));
	}
}




/******************************************************************************/
/*									DIFF                                      */
/******************************************************************************/
inline
void ActivationFunction_ReLU_Diff(_FLOAT * bot_diff, const _FLOAT* top_diff, const _FLOAT *bot_data, _FLOAT negative_slope)
{

	for (int i = 0; i < 4; ++i)
	{
		bot_diff[i] = top_diff[i] * (bot_data[i] > 0);
	}
}


inline
void ActivationFunction_TanH_Diff(_FLOAT * bot_diff, const _FLOAT* top_diff, const _FLOAT *top_data)
{
	for(int i = 0; i <4; i++)
	{
// (exp(2x) -1) / (exp(2x) + 1)
		_FLOAT tanh_x = top_data[i]; 
		bot_diff[i] = top_diff[i] * (1 - tanh_x*tanh_x);
	}
}

inline
void ActivationFunction_Sigmoid_Diff(_FLOAT * bot_diff, const _FLOAT* top_diff, const _FLOAT *top_data)
{
	for (int i = 0; i <4; i++)
	{
		// 1/(1 + exp(-x))  
		_FLOAT sigmoid_x = top_data[i];
		bot_diff[i] = top_diff[i] * sigmoid_x * (1.f - sigmoid_x);
	}
}


inline
void ActivationFunction_Abs_Diff(_FLOAT * bot_diff, const _FLOAT* top_diff, const _FLOAT *bot_data)
{
	for (int i = 0; i <4; i++)
	{
		bot_diff[i] = top_diff[i] * ((bot_data >= 0 ) ? 1 : -1);
	}
}


// Compute dy/dx = scale * power * (shift + scale * x)^(power - 1)
//               = diff_scale * y / (shift + scale * x)
inline
void ActivationFunction_Power_Diff(_FLOAT * bot_diff, const _FLOAT* top_diff, const _FLOAT *top_data, const _FLOAT *bot_data,
_FLOAT diff_scale,
_FLOAT power,
_FLOAT scale,
_FLOAT shift)
{

	for (int i = 0; i <4; i++)
	{
		_FLOAT arg = shift + bot_data[i] * scale;
		bot_diff[i] = (arg == 0) ? 0 : diff_scale * top_data[i]/arg;

	}
}

inline
void ActivationFunction_BNLL_Diff(_FLOAT * bot_diff, const _FLOAT* top_diff, const _FLOAT *bot_data)
{
	for (int i = 0; i <4; i++)
	{
		//	(log(1 + exp(x)))' = 1/ (1 + exp(-x))
		bot_diff[i] = top_diff[i] * (1.f + native_exp(-bot_data[i]));
	}
}


#endif