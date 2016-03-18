/*
 * Copyright (c) 2016 AMD Inc.
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

//ADNN_WWT_GRP_SZ0				weights transform group size in dim 0
//ADNN_WWT_GRP_SZ1				weights transform group size in dim 1
//ADNN_WWT_GRP_SZ2				weights transform group size in dim 2
//ADNN_WWT_PEROUTPUT            weights linear over output; otherwise linear per input
//ADNN_WIT_GRP_SZ0				input transform group size in dim 0
//ADNN_WIT_GRP_LG2SZ0			log
//ADNN_WIT_GRP_SZ1				input transform group size in dim 1
//ADNN_WIT_GRP_SZ2				input transform group size in dim 2
//ADNN_WIT_OUTLINEAR            inverse tarsnform layout - linear; otherwise tiled per block1xblock0 block
//ADNN_W_BATCH_SZ				batch size
//ADNN_W_N_ICHNLS               total number of input channels
//ADNN_W_N_OCHNLS               total number of output channels
//ADNN_W_HEIGH                  input height
//ADNN_W_WIDTH					input width
//ADNN_W_ISTRIDE				input stride
//ADNN_W_ICHNL_STRIDE			input channel stride
//ADNN_W_IBATCH_STRIDE          input batch strride
//ADNN_W_BSTRIDE				blocked(trasformed input) stride
//ADNN_W_BCHNL_STRIDE			blocked(trasformed input) channel stride
//ADNN_W_BBATCH_STRIDE			blocked(trasformed input) batch stride
//ADNN_W_TILE1					tile: 2x2 or 4x4
//ADNN_W_TILE0
//ADNN_W_BLOCK0					block size dim 0 4x4 or 6x6
//ADNN_W_BLOCK1					block size dim 1
//ADNN_W_FLTR0					filter size dim 0
//ADNN_W_FLTR1					filter size dim 1
//ADNN_W_PAD1,
//ADNN_W_PAD0
//ADNN_WIT_LCL_N_IN_CHNLS			n of localy kept input channels
//ADNN_WIT_LCL_LOG2N_IN_CHNLS		log2 of n of localy kept input channels
//ADNN_WIT_N_IN_TILES0				n total input tiles dim 0 for > 32x32
//ADNN_WIT_N_IN_TILES1				n total input tiles dim 1 for > 32x32
//ADNN_WIT_IN_TILE_LOG2SZ1          log size of input tile for > 32x32
//ADNN_WIT_IN_TILE_LOG2SZ0          log size of input tile for > 32x32
//ADNN_WIT_IN_PROC_SZ				n of processors reading 1 input channel for < 32x32 :  (1  << (ADNN_WIT_GRP_LG2SZ0 - ADNN_LCL_LOG2N_IN_CHNLS))
//ADNN_WIT_READ_SZ1					length to read dim 1
//ADNN_WIT_READ_SZ0,				lenght to read dim 0
//ADNN_WIT_N_TILEPROCS1           n proc in proc tile dim 1
//ADNN_WIT_N_TILEPROCS0           n proc in proc tile dim 0
//ADNN_WIT_BLKD_TILE_SZ1				size of blocked tiles dim 1 for > 32x32
//ADNN_WIT_BLKD_TILE_SZ0				size of blocked tiles dim 0 for > 32x32
//ADNN_WIT_BIG                  input > 32x32
//ADNN_WIT_IN_SIZE              input block size
//ADNN_WIT_LCL_HEIGHT				local memory height (ADNN_IN_HEIGHT + ADNN_W_PAD1 * 2)
//ADNN_WIT_LCL_WIDTH				local memory width (ADNN_IN_WIDTH + ADNN_W_PAD0 * 2)
//ADNN_WIT_LCL_STRIDE				local memory stride ADNN_LCL_WIDTH
//ADNN_WIT_LCL_SZ					local memory size (ADNN_WIT_LCL_STRIDE * ADNN_LCL_HEIGHT)
//invert transform
//ADNN_WMIT_GRP_SZ0
//ADNN_WMIT_GRP_SZ1
//ADNN_WMIT_GRP_SZ2
//ADNN_WMIT_LCL_N_IN_CHNLS          n of input channels - the  same id but from different batchs
//ADNN_WMIT_LCL_N_OUT_CHNLS
//ADNN_WMIT_IN_PROC_SZ             n of processors reading 1 input channel
//ADNN_W_BHEIGH                  height of the transformed buffer
//ADNN_W_BWIDTH                 width of tarnsformed data buffer in pixels
//ADNN_W_OBATCH_STRIDE
//ADNN_W_OCHNL_STRIDE
//ADNN_W_OSTRIDE
// multiply
//ADNN_WMT_GRP_SZ0			// group size
//ADNN_WMT_GRP_SZ1
//ADNN_WMT_GRP_SZ2
//ADNN_WMT_LCL_N_OUT_CHNLS // n of local output channels
//ADNN_WMT_PIXPER_WKITM    // n input pixels per wk-item
//ADNN_WMT_WKITMSPER_BLOCK // n wk items per block
//WIT_LCL_TFM_WIDTH			 input tile transformed size 0
//WIT_LCL_TFM_HEIGHT        input tile transformed size 1


#define ADNN_W_WEI_CONST_CAP ((1 << 14) - (ADNN_WI_BLOCK1 * ADNN_WI_BLOCK0 + ADNN_WI_BLOCK1 * ADNN_WI_FLTR0  + ADNN_WI_BLOCK1 * ADNN_WI_TILE0 + 256))

// constant
		// mutrix multily convolution
		__constant _FLOAT C[ADNN_W_BLOCK1 * ADNN_W_BLOCK0] =
		{
#if ADNN_ALG_WIN2x2_3x3
			1, 0, 0, 0,
			0, 1, -1, 1,
			-1, 1, 1, 0,
			0, 0, 0, -1
#else

			 4,  0,  0,  0,  0,  0,
			 0, -4,  4, -2,  2,  4,
			-5, -4, -4, -1, -1,  0,
			 0,  1, -1,  2, -2, -5,
			 1,  1,  1,  1,  1,  0,
			 0,  0,  0,  0,  0,  1

#endif
		};

		__constant _FLOAT G[ADNN_W_BLOCK1 * ADNN_W_FLTR0] =
		{
#if ADNN_ALG_WIN2x2_3x3
			1, 0, 0,
			0.5f, 0.5f, 0.5f,
			0.5f, -0.5f, 0.5f,
			0, 0, 1
#else
			(float)(1./4), 0,  0,
			-(float)(1./6.), -(float)(1./6.), -(float)(1./6.),
			-(float)(1./6.), (float)(1./6.), -(float)(1./6.),
			(float)(1./24), (float)(1./12), (float)(1./6),
			(float)(1./24), -(float)(1./12), (float)(1./6),
			0, 0, 1
#endif
		};

		__constant _FLOAT A[ADNN_W_BLOCK1 * ADNN_W_TILE0] =
		{
#if ADNN_ALG_WIN2x2_3x3
			1, 0,
			1, 1,
			1, -1,
			0, -1
#else
			1,  0, 0,  0,
			1,  1, 1,  1, 
			1, -1, 1, -1,
			1,  2, 4,  8,
			1, -2, 4, -8,
			0,  0, 0,  1

#endif
		};


/*************************************************************************************************
**
** weights transform
**
*************************************************************************************************/

	void Weights_Block_Tansform_2x2_3x3Win(
	__global _FLOAT *GxgxG_T, 
#if ADNN_WEI_SZ < ADNN_W_WEI_CONST_CAP
	 __constant _FLOAT *g,
#else
	 __global const _FLOAT *g,
#endif
      __constant _FLOAT * G
	  )
	{

		// Gxg : ((ADNN_W_TILE1 + ADNN_W_PAD1 *2) x ADNN_W_FLTR0) * (ADNN_W_FLTR1 x ADNN_W_FLTR0).T
		// trasnposed

		_FLOAT Gxg[ADNN_W_BLOCK1 * ADNN_W_FLTR0];

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


	}




	void Weights_Block_Tansform_Win(
	__global _FLOAT *GxgxG_T,
#if ADNN_WEI_SZ < ADNN_W_WEI_CONST_CAP
	 __constant _FLOAT *g,
#else
	 __global const _FLOAT *g,
#endif
      __constant _FLOAT * G
	 )
	{

#if ADNN_ELEMENT_WISE_TRANSOFRMS


	Weights_Block_Tansform_2x2_3x3Win(GxgxG_T, g, G);


#else

		_FLOAT Gxg[ADNN_W_BLOCK1 * ADNN_W_FLTR0];

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

/*************************************************************************************************
**
** weights transform kernel
**
*************************************************************************************************/


__attribute__((reqd_work_group_size(ADNN_WWT_GRP_SZ0,ADNN_WWT_GRP_SZ1,ADNN_WWT_GRP_SZ2)))
__kernel void Weights_Tansform_Win
		(__global _FLOAT *t_w,
#if ADNN_WEI_SZ < ADNN_W_WEI_CONST_CAP
      __constant _FLOAT * w __attribute__((max_constant_size (ADNN_WEI_SZ) ))
#else
      __global _FLOAT * w
#endif
		 )
	{
		int o = get_global_id(1);
		int c = get_global_id(0);

		__global _FLOAT *GxgxG_T =

#if ADNN_WWT_PEROUTPUT
		 &t_w[(o * ADNN_W_N_ICHNLS + c) * ADNN_W_BLOCK1 * ADNN_W_BLOCK0];
#else
		 &t_w[(c * ADNN_W_N_OCHNLS + o) * ADNN_W_BLOCK1 * ADNN_W_BLOCK0];
#endif

#if ADNN_WEI_SZ < ADNN_W_WEI_CONST_CAP
		__constant _FLOAT *
#else
		__global const _FLOAT *
#endif
		my_g = &w[(o * ADNN_W_N_ICHNLS + c) * ADNN_W_FLTR1 * ADNN_W_FLTR0];

// block transform
		Weights_Block_Tansform_Win(GxgxG_T, my_g, G);

	}






/*************************************************************************************************
**
** input transform
**
*************************************************************************************************/

	void Input_Block_Tansform_2x2_3x3Win(
//	__global
	 _FLOAT *C_TxdxC,
	 __local _FLOAT *run_d,
	 __constant _FLOAT *C,
	 int out_stride
	 )
	{

		_FLOAT C_Txd[ADNN_W_BLOCK1 * ADNN_W_BLOCK0];
		// data transform

		C_Txd[0] = run_d[0 * ADNN_WIT_LCL_STRIDE + 0] - run_d[0 * ADNN_WIT_LCL_STRIDE + 2];
		C_Txd[1] = run_d[1 * ADNN_WIT_LCL_STRIDE + 0] - run_d[1 * ADNN_WIT_LCL_STRIDE + 2];
		C_Txd[2] = run_d[2 * ADNN_WIT_LCL_STRIDE + 0] - run_d[2 * ADNN_WIT_LCL_STRIDE + 2];
		C_Txd[3] = run_d[3 * ADNN_WIT_LCL_STRIDE + 0] - run_d[3 * ADNN_WIT_LCL_STRIDE + 2];
		C_Txd[4] = run_d[0 * ADNN_WIT_LCL_STRIDE + 1] + run_d[0 * ADNN_WIT_LCL_STRIDE + 2];
		C_Txd[5] = run_d[1 * ADNN_WIT_LCL_STRIDE + 1] + run_d[1 * ADNN_WIT_LCL_STRIDE + 2];
		C_Txd[6] = run_d[2 * ADNN_WIT_LCL_STRIDE + 1] + run_d[2 * ADNN_WIT_LCL_STRIDE + 2];
		C_Txd[7] = run_d[3 * ADNN_WIT_LCL_STRIDE + 1] + run_d[3 * ADNN_WIT_LCL_STRIDE + 2];
		C_Txd[8] = -run_d[0 * ADNN_WIT_LCL_STRIDE + 1] + run_d[0 * ADNN_WIT_LCL_STRIDE + 2];
		C_Txd[9] = -run_d[1 * ADNN_WIT_LCL_STRIDE + 1] + run_d[1 * ADNN_WIT_LCL_STRIDE + 2];
		C_Txd[10] = -run_d[2 * ADNN_WIT_LCL_STRIDE + 1] + run_d[2 * ADNN_WIT_LCL_STRIDE + 2];
		C_Txd[11] = -run_d[3 * ADNN_WIT_LCL_STRIDE + 1] + run_d[3 * ADNN_WIT_LCL_STRIDE + 2];
		C_Txd[12] = run_d[0 * ADNN_WIT_LCL_STRIDE + 1] - run_d[0 * ADNN_WIT_LCL_STRIDE + 3];
		C_Txd[13] = run_d[1 * ADNN_WIT_LCL_STRIDE + 1] - run_d[1 * ADNN_WIT_LCL_STRIDE + 3];
		C_Txd[14] = run_d[2 * ADNN_WIT_LCL_STRIDE + 1] - run_d[2 * ADNN_WIT_LCL_STRIDE + 3];
		C_Txd[15] = run_d[3 * ADNN_WIT_LCL_STRIDE + 1] - run_d[3 * ADNN_WIT_LCL_STRIDE + 3];


		C_TxdxC[0 * out_stride + 0] = C_Txd[0] - C_Txd[2];
		C_TxdxC[0 * out_stride + 1] = C_Txd[1] + C_Txd[2];
		C_TxdxC[0 * out_stride + 2] = -C_Txd[1] + C_Txd[2];
		C_TxdxC[0 * out_stride + 3] = C_Txd[1] - C_Txd[3];
		C_TxdxC[1 * out_stride + 0] = C_Txd[4] - C_Txd[6];
		C_TxdxC[1 * out_stride + 1] = C_Txd[5] + C_Txd[6];
		C_TxdxC[1 * out_stride + 2] = -C_Txd[5] + C_Txd[6];
		C_TxdxC[1 * out_stride + 3] = C_Txd[5] - C_Txd[7];
		C_TxdxC[2 * out_stride + 0] = C_Txd[8] - C_Txd[10];
		C_TxdxC[2 * out_stride + 1] = C_Txd[9] + C_Txd[10];
		C_TxdxC[2 * out_stride + 2] = -C_Txd[9] + C_Txd[10];
		C_TxdxC[2 * out_stride + 3] = C_Txd[9] - C_Txd[11];
		C_TxdxC[3 * out_stride + 0] = C_Txd[12] - C_Txd[14];
		C_TxdxC[3 * out_stride + 1] = C_Txd[13] + C_Txd[14];
		C_TxdxC[3 * out_stride + 2] = -C_Txd[13] + C_Txd[14];
		C_TxdxC[3 * out_stride + 3] = C_Txd[13] - C_Txd[15];


	}




	void Input_Block_Tansform_Win(
//		__global 
		_FLOAT *C_TxdxC,
		__local _FLOAT *run_d,
		__constant _FLOAT *C,
		int out_stride
		)
	{

		// data transform
#if ADNN_ELEMENT_WISE_TRANSOFRMS
		Input_Block_Tansform_2x2_3x3Win(C_TxdxC, run_d, C, out_stride);
#else
		_FLOAT C_Txd[ADNN_W_BLOCK1 * ADNN_W_BLOCK0];

		// C_T x d
		for (int j = 0; j < ADNN_W_BLOCK1; ++j)
		{
			for (int i = 0; i < ADNN_W_BLOCK0; ++i)
			{
				C_Txd[j * ADNN_W_BLOCK0 + i] = 0;
				for (int k = 0; k < ADNN_W_BLOCK0; ++k)
				{
					// C transposed + run_d transposed
					C_Txd[j * ADNN_W_BLOCK0 + i] += C[k * ADNN_W_BLOCK0 + j] * run_d[i * ADNN_WIT_LCL_STRIDE + k];
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
				C_TxdxC[j * out_stride + i] = 0;
				for (int k = 0; k < ADNN_W_BLOCK0; ++k)
				{
					C_TxdxC[j * out_stride + i] += C_Txd[j * ADNN_W_BLOCK0 + k] * C[k * ADNN_W_BLOCK0 + i];
//					printf("dC: j=%d i=%d C_TdI=%d ctv=%f cv=%f\n", j, i, j * ADNN_W_BLOCK0 + k, C_Txd[j * ADNN_W_BLOCK0 + k], C[k * ADNN_W_BLOCK0 + i]);
				}
//				printf("dC: dCI=%d cv=%f\n", j * out_stride + i, C_TxdxC[j * out_stride + i]);
			}
		}
#endif
	}


inline void calculateXYPos(int linPos, int width, int *x, int *y)
{
	(*y) = (int)((_FLOAT)linPos /(_FLOAT)width);
	(*x) = linPos - (*y) * width; 
}

inline int calculateOffset(int stride, int x, int y)
{
	int ret = y * stride + x;
	return(ret);
}

inline void readDataElem(__local _FLOAT *lcl_data, __global _FLOAT * gbl_data, int linPos, int gbl_width, int gbl_stride, int gbl_base, int lcl_stride, int lcl_base, int horiz_pad, int batch)
{
	int x, y;
	calculateXYPos(linPos, gbl_width, &x, &y);
	int gbl_off = calculateOffset(gbl_stride, x, y);
	gbl_off += gbl_base;
	int lcl_off = calculateOffset(lcl_stride, x, y);
// shift along y and x by pad size
	lcl_off += lcl_base + horiz_pad;
	_FLOAT val = gbl_data[gbl_off];
	val = (ADNN_W_BATCH_SZ <= batch) ? 0 : val;
	lcl_data[lcl_off] = val;
}

// split the group into several input vector processors
// each processor reads its own input channel
inline void readData(__local _FLOAT *lcl_data, __global _FLOAT * gbl_data, int lcl_p_id, int lcl_p_stride, int size, int gbl_width, int gbl_stride, int gbl_base, int lcl_stride, int lcl_base, int horiz_pad, int batch, int debug)
{
	
	for(int i = lcl_p_id; i < size; i+= lcl_p_stride)
	{
		readDataElem(lcl_data, gbl_data, i, gbl_width, gbl_stride, gbl_base, lcl_stride, lcl_base, horiz_pad, batch);
	}

}

inline void readDataTile(__local _FLOAT *lcl_data, __global _FLOAT * gbl_data,
						int tile_y, int tile_x,
						int gbl_stride, int gbl_base,
						int lcl_stride, int lcl_base,
						int gbl_height, int gbl_width,
						int lcl_height, int lcl_width,
						int lcl_id1, int lcl_id0,
						int lcl_grp_sz1, int lcl_grp_sz0,
						int fltr_pad1, int fltr_pad0,
						_FLOAT padding_val,
						int batch)
{
			for( int j = lcl_id1; j < lcl_height; j += lcl_grp_sz1)
			{	
				int y_act = (j - fltr_pad1);
				bool invisibleY = (tile_y + y_act < 0) || (tile_y + y_act >= gbl_height);

				int y_gbl_off = y_act * gbl_stride;

				int y_lcl_off = j * lcl_stride;

				for(int i = lcl_id0; i < lcl_width; i += lcl_grp_sz0)
				{
					int x_act = (i - fltr_pad0);
					bool invisibleX = (tile_x + x_act < 0) || (tile_x + x_act >= gbl_width);

					_FLOAT val = gbl_data[gbl_base + y_gbl_off + x_act];

					val = (invisibleX || invisibleY || ADNN_W_BATCH_SZ <= batch)? padding_val : val;
								
					lcl_data[y_lcl_off + i] = val;
				}
			}
}




/*************************************************************************************************
**
** inverse transform
**
*************************************************************************************************/

	// inverse transform

	void Inverse_BlockTransform_2x2_3x3Win(
		__global _FLOAT *run_f, _FLOAT * Elem_wise_Mult, __constant _FLOAT *A, _FLOAT * A_TxEWM)
	{

		A_TxEWM[0] = Elem_wise_Mult[0 * ADNN_W_BLOCK0 + 0] + Elem_wise_Mult[1 * ADNN_W_BLOCK0 + 0] + Elem_wise_Mult[2 * ADNN_W_BLOCK0 + 0];
		A_TxEWM[1] = Elem_wise_Mult[0 * ADNN_W_BLOCK0 + 1] + Elem_wise_Mult[1 * ADNN_W_BLOCK0 + 1] + Elem_wise_Mult[2 * ADNN_W_BLOCK0 + 1];
		A_TxEWM[2] = Elem_wise_Mult[0 * ADNN_W_BLOCK0 + 2] + Elem_wise_Mult[1 * ADNN_W_BLOCK0 + 2] + Elem_wise_Mult[2 * ADNN_W_BLOCK0 + 2];
		A_TxEWM[3] = Elem_wise_Mult[0 * ADNN_W_BLOCK0 + 3] + Elem_wise_Mult[1 * ADNN_W_BLOCK0 + 3] + Elem_wise_Mult[2 * ADNN_W_BLOCK0 + 3];
		A_TxEWM[4] = Elem_wise_Mult[1 * ADNN_W_BLOCK0 + 0] - Elem_wise_Mult[2 * ADNN_W_BLOCK0 + 0] - Elem_wise_Mult[3 * ADNN_W_BLOCK0 + 0];
		A_TxEWM[5] = Elem_wise_Mult[1 * ADNN_W_BLOCK0 + 1] - Elem_wise_Mult[2 * ADNN_W_BLOCK0 + 1] - Elem_wise_Mult[3 * ADNN_W_BLOCK0 + 1];
		A_TxEWM[6] = Elem_wise_Mult[1 * ADNN_W_BLOCK0 + 2] - Elem_wise_Mult[2 * ADNN_W_BLOCK0 + 2] - Elem_wise_Mult[3 * ADNN_W_BLOCK0 + 2];
		A_TxEWM[7] = Elem_wise_Mult[1 * ADNN_W_BLOCK0 + 3] - Elem_wise_Mult[2 * ADNN_W_BLOCK0 + 3] - Elem_wise_Mult[3 * ADNN_W_BLOCK0 + 3];


		// x A
		// transpose output
		run_f[0 * ADNN_W_OSTRIDE + 0] = A_TxEWM[0] + A_TxEWM[1] + A_TxEWM[2];
		run_f[1 * ADNN_W_OSTRIDE + 0] = A_TxEWM[1] - A_TxEWM[2] - A_TxEWM[3];
		run_f[0 * ADNN_W_OSTRIDE + 1] = A_TxEWM[4] + A_TxEWM[5] + A_TxEWM[6];
		run_f[1 * ADNN_W_OSTRIDE + 1] = A_TxEWM[5] - A_TxEWM[6] - A_TxEWM[7];
	}



		void Inverse_BlockTransform_Win(
		__global _FLOAT *run_f, _FLOAT * Elem_wise_Mult, __constant _FLOAT *A, _FLOAT * A_TxEWM)
	{


#if ADNN_ELEMENT_WISE_TRANSOFRMS

		Inverse_BlockTransform_2x2_3x3Win
			(
				run_f, Elem_wise_Mult, A, A_TxEWM
			);
#else

		for (int j = 0; j < ADNN_W_TILE1; ++j)
		{
			for (int i = 0; i < ADNN_W_BLOCK1; ++i)
			{
				A_TxEWM[j * ADNN_W_BLOCK1 + i] = 0;
				for (int k = 0; k < ADNN_W_BLOCK0; ++k)
				{
					// A transposed
					A_TxEWM[j * ADNN_W_BLOCK1 + i] += A[k*ADNN_W_TILE1 + j] * Elem_wise_Mult[k*ADNN_W_BLOCK1 + i];
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


/*************************************************************************************************
**
** convolution 2x2,3x3
**
*************************************************************************************************/
#define ADNN_WIT_LCL_TFM_SIZE (ADNN_WIT_LCL_TFM_WIDTH * ADNN_WIT_LCL_TFM_HEIGHT)

__attribute__((reqd_work_group_size(ADNN_WIT_GRP_SZ0,ADNN_WIT_GRP_SZ1,ADNN_WIT_GRP_SZ2)))
__kernel void Conv_Win2x2_3x3(
								__global _FLOAT * conv_data,
								__global _FLOAT *blocked_d,
								__global _FLOAT * transformed_out,
								 __global const _FLOAT *d,
								 __global const _FLOAT *transformed_weights,
								 _FLOAT padding_val,
								__constant int4 * procs __attribute__((max_constant_size(ADNN_WIT_GRP_SZ0*2))),
								int inputs
								 )
	{

		__local _FLOAT lcl_d[ADNN_WIT_LCL_N_IN_CHNLS * ADNN_WIT_LCL_TFM_SIZE];
		__local _FLOAT lcl_t_w[ADNN_WIT_LCL_N_IN_CHNLS * ADNN_WMT_LCL_N_OUT_CHNLS*ADNN_W_BLOCK1 * ADNN_W_BLOCK0];

		_FLOAT pvt_inp_t[ADNN_W_BLOCK1 * ADNN_W_BLOCK0];

		_FLOAT pvt_ewm_accum[ADNN_WMT_LCL_N_OUT_CHNLS*ADNN_W_BLOCK1 * ADNN_W_BLOCK0];

		int tile_id = get_group_id(0); // tile id if larger than 32x32
		int lcl_id = get_local_id(0); 
//		int c = get_global_id(1); // input channel block id
		int o_id = get_global_id(1); // output id

		int b = get_global_id(2); // batch id
		int gbl_tile_id_y = (int) ((_FLOAT)tile_id / ADNN_WIT_N_IN_TILES0);
		int gbl_tile_id_x = tile_id - mul24(gbl_tile_id_y, ADNN_WIT_N_IN_TILES0);
		int tile_y = (gbl_tile_id_y << ADNN_WIT_IN_TILE_LOG2SZ1);
		int tile_x = (gbl_tile_id_x << ADNN_WIT_IN_TILE_LOG2SZ0);


// input processing
// split the group into ADNN_LCL_N_IN_CHNLS input vector processors
// each processor reads its own input channel
		int in_gbl_base = 0;
		int lcl_base = ADNN_WIT_LCL_STRIDE * ADNN_W_PAD1;
// input processors layout
		int in_proc_id = procs[lcl_id].y;
		int in_thread_id = procs[lcl_id].x;
		int lcl_in1 = procs[lcl_id].w;
		int lcl_in0 = procs[lcl_id].z;

// processing
// data (output) processors layout
// find the my tile processor and my processor in the tile
// my processor tile id along y
		int lcl_tileid_p1 = procs[ADNN_WIT_GRP_SZ0 + lcl_id].w;
// my processor tileid id along x
		int lcl_tileid_p0 = procs[ADNN_WIT_GRP_SZ0 + lcl_id].z;
// my processor id along y
		int lcl_tile1 = procs[ADNN_WIT_GRP_SZ0 + lcl_id].y;
// my processor id along x
		int lcl_tile0 = procs[ADNN_WIT_GRP_SZ0 + lcl_id].x;
	


		int debug = 0;


		for(int i = 0; i < ADNN_WMT_LCL_N_OUT_CHNLS*ADNN_W_BLOCK1 * ADNN_W_BLOCK0; ++i)
		{
			pvt_ewm_accum[i] = 0;
		}
		for(int c = 0; c < inputs/*ADNN_W_N_ICHNLS*/; c+=ADNN_WIT_LCL_N_IN_CHNLS)
		{


#if 1
			for(int i = lcl_id; i < ADNN_WIT_LCL_N_IN_CHNLS * ADNN_WIT_LCL_SZ; i+=ADNN_WIT_GRP_SZ0)
			{
				lcl_d[i] = 0;
			}
			barrier(CLK_LOCAL_MEM_FENCE);
#endif
// read ADNN_LCL_N_IN_CHNLS input channels cooperatively

			int input_chnl_id = (c*ADNN_WIT_LCL_N_IN_CHNLS + in_proc_id);
			in_gbl_base = b * ADNN_W_IBATCH_STRIDE + input_chnl_id * ADNN_W_ICHNL_STRIDE;

#if ADNN_WIT_BIG == 0

			readData(&lcl_d[in_proc_id * ADNN_WIT_LCL_SZ], d, in_thread_id, ADNN_WIT_IN_PROC_SZ, ADNN_WIT_IN_SIZE, ADNN_W_WIDTH, ADNN_W_ISTRIDE, in_gbl_base, ADNN_WIT_LCL_STRIDE, lcl_base, ADNN_W_PAD0, 0, debug);
#else

			in_gbl_base += tile_y * ADNN_W_ISTRIDE + tile_x;
			readDataTile(lcl_d, d,
				tile_y, tile_x,
				ADNN_W_ISTRIDE, in_gbl_base,
				ADNN_WIT_LCL_STRIDE, 0,
				ADNN_W_HEIGH, ADNN_W_WIDTH, 
				ADNN_WIT_LCL_HEIGHT, ADNN_WIT_LCL_WIDTH,
				lcl_in1, lcl_in0,
				ADNN_WIT_READ_SZ1, ADNN_WIT_READ_SZ0,
				ADNN_W_PAD1, ADNN_W_PAD0,
				padding_val, 0);

#endif

			barrier(CLK_LOCAL_MEM_FENCE);

#if 0
			if(input_chnl_id >= ADNN_W_N_ICHNLS)
			{
				return;
			}
#endif
// get weights
			// get weights
			for(int i = lcl_id; i < ADNN_WMT_LCL_N_OUT_CHNLS*ADNN_W_BLOCK1 * ADNN_W_BLOCK0; i+=ADNN_WMT_GRP_SZ0)
			{
				lcl_t_w[i] = transformed_weights[(c*ADNN_WIT_LCL_N_IN_CHNLS *ADNN_W_N_OCHNLS  + o_id * ADNN_WMT_LCL_N_OUT_CHNLS)*ADNN_W_BLOCK1 * ADNN_W_BLOCK0 + i];
			}

			int lcl_chnl_id = in_proc_id;
	
			int out_stride;
			out_stride = ADNN_W_BLOCK0;
			__local _FLOAT * l_d = &lcl_d[lcl_chnl_id * ADNN_WIT_LCL_SZ + lcl_tile1 * ADNN_W_TILE1 * ADNN_WIT_LCL_STRIDE + lcl_tile0 * ADNN_W_TILE0];

#if 0
						if (get_global_id(0) == 0 && c == 1)
						{
							printf("k:it: %d %f %f %f %f\n", 
							lcl_chnl_id * ADNN_WIT_LCL_SZ + lcl_tile1 * ADNN_W_TILE1 * ADNN_WIT_LCL_STRIDE + lcl_tile0 * ADNN_W_TILE0,
							l_d[0],
							l_d[1],
							l_d[2],
							l_d[3]
							);
						}
#endif

// input trasnform
			Input_Block_Tansform_Win(pvt_inp_t, l_d, C, out_stride);
			barrier(CLK_LOCAL_MEM_FENCE);

// keep it locally
			int lcl_t_off = lcl_chnl_id * ADNN_WIT_LCL_TFM_SIZE + lcl_tile1 * ADNN_W_BLOCK1 * ADNN_WIT_LCL_TFM_WIDTH + lcl_tile0 * ADNN_W_BLOCK0;
			for(int j = 0; j < ADNN_W_BLOCK1; ++j, lcl_t_off+=ADNN_WIT_LCL_TFM_WIDTH)
			{
				for(int i = 0; i < ADNN_W_BLOCK0; ++i)
				{
					lcl_d[lcl_t_off + i] = pvt_inp_t[j * ADNN_W_BLOCK0 + i];

				}
			}
			barrier(CLK_LOCAL_MEM_FENCE);


			lcl_t_off = lcl_chnl_id * ADNN_WIT_LCL_TFM_SIZE + lcl_tile1 * ADNN_W_BLOCK1 * ADNN_WIT_LCL_TFM_WIDTH + lcl_tile0 * ADNN_W_BLOCK0;
			int lcl_t_w_off = lcl_chnl_id * ADNN_WMT_LCL_N_OUT_CHNLS * ADNN_W_BLOCK1 * ADNN_W_BLOCK0;
			for(int j = 0; j < ADNN_W_BLOCK1; ++j, lcl_t_off+=ADNN_WIT_LCL_TFM_WIDTH, lcl_t_w_off += ADNN_W_BLOCK0)
			{
				for(int i = 0; i < ADNN_W_BLOCK0; ++i)
				{
				    int lcl_t_w_off2 = lcl_t_w_off;

					for(int o=0; o < ADNN_WMT_LCL_N_OUT_CHNLS; ++o, lcl_t_w_off2 += ADNN_W_BLOCK1 * ADNN_W_BLOCK0)
					{				
						pvt_ewm_accum[(o*ADNN_W_BLOCK1 + j) * ADNN_W_BLOCK0 + i] += lcl_d[lcl_t_off + i] * lcl_t_w[lcl_t_w_off2 + i];

#if 0
						if (get_global_id(0) == 0 && i==0 && j == 0 && o == 0)
						{
							printf("k: %d %d %d %f %f %f\n", 
							(o*ADNN_W_BLOCK1 + j) * ADNN_W_BLOCK0 + i, lcl_t_off + i, lcl_t_w_off2 + i,
							pvt_ewm_accum[(o*ADNN_W_BLOCK1 + j) * ADNN_W_BLOCK0 + i],
							lcl_d[lcl_t_off + i],
							lcl_t_w[lcl_t_w_off2 + i]
							);
						}
#endif
					}
				}
			}

			barrier(CLK_LOCAL_MEM_FENCE);


		}

#if 0
		int tile_stride = (16/ADNN_WIT_N_TILEPROCS0);
		int lcl_chnl_id = in_proc_id;
		int gbl_channel_id = 0;

		int lcl_t_off = lcl_chnl_id * ADNN_WIT_LCL_TFM_SIZE + lcl_tile1 * ADNN_W_BLOCK1 * ADNN_WIT_LCL_TFM_WIDTH + lcl_tile0 * ADNN_W_BLOCK0;

		int b_d_off = b*ADNN_W_BBATCH_STRIDE + gbl_channel_id*ADNN_W_BCHNL_STRIDE;
		int out_stride = ADNN_W_BSTRIDE;
		b_d_off += (gbl_tile_id_y * ADNN_WIT_BLKD_TILE_SZ1 + lcl_tile1 * ADNN_W_BLOCK1) * out_stride + (gbl_tile_id_x * ADNN_WIT_BLKD_TILE_SZ0 + lcl_tile0) * ADNN_W_BLOCK0;

		for(int j = 0; j < ADNN_W_BLOCK1; ++j, b_d_off +=  out_stride, lcl_t_off+=ADNN_WIT_LCL_TFM_WIDTH)
		{
			for(int i = 0; i < ADNN_W_BLOCK0; ++i)
			{
				blocked_d[b_d_off + i] = lcl_d[lcl_t_off + i];
			}
		}


		
		int out_off = b*ADNN_W_OBBATCH_STRIDE + o_id * ADNN_WMT_LCL_N_OUT_CHNLS * ADNN_W_OBCHNL_STRIDE
		+ (gbl_tile_id_y * ADNN_WIT_BLKD_TILE_SZ1 + lcl_tile1 * ADNN_W_BLOCK1) * ADNN_W_OBSTRIDE + (gbl_tile_id_x * ADNN_WIT_BLKD_TILE_SZ0 + lcl_tile0) * ADNN_W_BLOCK0;


		for(int o=0; o < ADNN_WMT_LCL_N_OUT_CHNLS; ++o, out_off += ADNN_W_OBCHNL_STRIDE)
		{				
			int out_off2 = out_off;
			for(int j = 0; j < ADNN_W_BLOCK1; ++j, out_off2+= ADNN_W_OBSTRIDE)
			{
				for(int i = 0; i < ADNN_W_BLOCK0; ++i)
				{
						transformed_out[out_off2 + i] = pvt_ewm_accum[(o*ADNN_W_BLOCK1 + j) * ADNN_W_BLOCK0 + i];
#if 0
						if (get_global_id(0) == 0 && i==1 && j == 0 && o == 0)
						{
							printf("k: %d %d %f %f\n", 
							out_off + i, (o*ADNN_W_BLOCK1 + j) * ADNN_W_BLOCK0 + i,
							transformed_out[out_off2 + i],
							pvt_ewm_accum[(o*ADNN_W_BLOCK1 + j) * ADNN_W_BLOCK0 + i]
							);
						}
#endif
				}
			}
		}


#endif

#if 1
		int conv_data_off = b*ADNN_W_OBATCH_STRIDE + o_id * ADNN_WMT_LCL_N_OUT_CHNLS * ADNN_W_OCHNL_STRIDE
					+ (tile_y + lcl_tile1 * ADNN_W_TILE1) * ADNN_W_OSTRIDE + tile_x + lcl_tile0 * ADNN_W_TILE0;
// inverse transform
		_FLOAT A_TxEWM[ADNN_W_TILE1 * ADNN_W_BLOCK1]; 

		for(int o=0; o < ADNN_WMT_LCL_N_OUT_CHNLS; ++o, conv_data_off += ADNN_W_OCHNL_STRIDE)
		{				
			Inverse_BlockTransform_Win(&conv_data[conv_data_off], &pvt_ewm_accum[o*ADNN_W_BLOCK1 * ADNN_W_BLOCK0], A, A_TxEWM);
		}
#endif


	}
