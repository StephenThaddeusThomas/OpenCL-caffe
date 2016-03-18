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

#ifndef _ANN_WEIGHTS_UPDATE_CL_H
#define _ANN_WEIGHTS_UPDATE_CL_H

inline void annCalculateWeightsUpdate(	_FLOAT * we_hist_out, _FLOAT* we_out,
 										_FLOAT we_hist, _FLOAT we_df, _FLOAT we,
										_FLOAT momentum,
										_FLOAT rate, _FLOAT decay
									)
{
	*we_hist_out = we_hist * momentum + we_df * rate;
	*we_hist_out += decay * rate * we;
	*we_out = we - *we_hist_out;
}

#endif
