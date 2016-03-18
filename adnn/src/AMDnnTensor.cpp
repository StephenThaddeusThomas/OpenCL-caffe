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
   **			aDNNTensor Class
   **
   ************************************************************************************************************************/

  aDNNTensor::aDNNTensor() : ADNNBase()
  {
		setInternal(0);

		data_format_ = ADNN_DF_FP32;
		batch_format_ = ADNN_BF_NCHW;
		size_ = 0;
		size_bytes_ = 0;
		sys_mem_ = 0;
		ocl_mem_ = 0;
		control_bits_ = 0;
		remap_.clear();
		dims_.clear();
		strides_.clear();
		allocated_ = 0;
		context_ = 0;
		n_dims_ = 0;
	}

	aDNNTensor::aDNNTensor(const ADNNBase & lib, const adnn_data_parameters & c_descr) : ADNNBase()
	{
		setParent((void*)&lib);

		const ADNNLib * n_lib = (const ADNNLib *)getParent();

		context_ = n_lib->getContext();

		setInternal(0);

		data_format_ = c_descr.data_format;
		batch_format_ = c_descr.batch_format;
		size_ = 0;
		size_bytes_ = 0;
		sys_mem_ = 0;
		ocl_mem_ = 0;
		control_bits_ = c_descr.control_bits;
		remap_.clear();
		dims_.clear();
		strides_.clear();
		allocated_ = 0;
		int dims = c_descr.n_dims;
		int remap_dims = aDNN_TENSOR_5THDIM;
		if (c_descr.n_dims == 0)
		{
			switch (batch_format_)
			{
			case ADNN_BF_NCHW:
			case ADNN_BF_WHCN:
				dims = 4;

				break;

			case ADNN_BF_NHW:
			case ADNN_BF_WHN:
				dims = 3;
				break;

			case ADNN_BF_NW:
			case ADNN_BF_WN:
				dims = 2;
				break;

			case ADNN_BF_HW:
			case ADNN_BF_WH:
				dims = 2;
				break;

			case ADNN_BF_W:
				dims = 1;
				break;

			default:
				printf("Data error: unknown batch format %d\n", batch_format_);
				break;
			}

		}

		n_dims_ = dims;

		remap_.resize(remap_dims + 1);
		strides_.resize(remap_dims + 1);
		dims_.resize(remap_dims);
		for (int j = 0; j < remap_dims; ++j)
		{
			strides_[j] = 1;
			dims_[j] = 1;
			remap_[j] = j;
		}

		remap_[aDNN_TENSOR_0DIM] = 5;

		switch (batch_format_)
		{
		case ADNN_BF_NCHW:
			remap_[aDNN_TENSOR_WIDTH] = 3;
			remap_[aDNN_TENSOR_HEIGHT] = 2;
			remap_[aDNN_TENSOR_DEPTH] = 1;
			remap_[aDNN_TENSOR_BATCH] = 0;
			dims_[0] = (c_descr.dims[0] == 0) ? 1 : c_descr.dims[0];
			dims_[1] = (c_descr.dims[1] == 0) ? 1 : c_descr.dims[1];
			dims_[2] = (c_descr.dims[2] == 0) ? 1 : c_descr.dims[2];
			dims_[3] = (c_descr.dims[3] == 0) ? 1 : c_descr.dims[3];
			strides_[0] = (c_descr.strides[0] == 0) ? dims_[0] : c_descr.strides[0];
			strides_[1] = (c_descr.strides[1] == 0) ? dims_[1] : c_descr.strides[1];
			strides_[2] = (c_descr.strides[2] == 0) ? dims_[2] : c_descr.strides[2];
			strides_[3] = (c_descr.strides[3] == 0) ? dims_[3] : c_descr.strides[3];

			break;

		case ADNN_BF_NHW:
			remap_[aDNN_TENSOR_WIDTH] = 3;
			remap_[aDNN_TENSOR_HEIGHT] = 2;
			remap_[aDNN_TENSOR_DEPTH] = 1;
			remap_[aDNN_TENSOR_BATCH] = 0;
			dims_[0] = (c_descr.dims[0] == 0) ? 1 : c_descr.dims[0];
			dims_[2] = (c_descr.dims[1] == 0) ? 1 : c_descr.dims[1];
			dims_[3] = (c_descr.dims[2] == 0) ? 1 : c_descr.dims[2];
			strides_[0] = (c_descr.strides[0] == 0) ? dims_[0] : c_descr.strides[0];
			strides_[2] = (c_descr.strides[1] == 0) ? dims_[2] : c_descr.strides[1];
			strides_[3] = (c_descr.strides[2] == 0) ? dims_[3] : c_descr.strides[2];
			break;

		case ADNN_BF_NW:
			remap_[aDNN_TENSOR_WIDTH] = 3;
			remap_[aDNN_TENSOR_HEIGHT] = 2;
			remap_[aDNN_TENSOR_DEPTH] = 1;
			remap_[aDNN_TENSOR_BATCH] = 0;

			dims_[0] = (c_descr.dims[0] == 0) ? 1 : c_descr.dims[0];
			dims_[3] = (c_descr.dims[1] == 0) ? 1 : c_descr.dims[1];
			strides_[0] = (c_descr.strides[0] == 0) ? dims_[0] : c_descr.strides[0];
			strides_[3] = (c_descr.strides[1] == 0) ? dims_[3] : c_descr.strides[1];
			break;

		case ADNN_BF_HW:
			remap_[aDNN_TENSOR_WIDTH] = 3;
			remap_[aDNN_TENSOR_HEIGHT] = 2;
			remap_[aDNN_TENSOR_DEPTH] = 1;
			remap_[aDNN_TENSOR_BATCH] = 0;
			dims_[2] = (c_descr.dims[0] == 0) ? 1 : c_descr.dims[0];
			dims_[3] = (c_descr.dims[1] == 0) ? 1 : c_descr.dims[1];
			strides_[2] = (c_descr.strides[0] == 0) ? dims_[2] : c_descr.strides[0];
			strides_[3] = (c_descr.strides[1] == 0) ? dims_[3] : c_descr.strides[1];


			break;
		case ADNN_BF_W:
			remap_[aDNN_TENSOR_WIDTH] = 3;
			remap_[aDNN_TENSOR_HEIGHT] = 2;
			remap_[aDNN_TENSOR_DEPTH] = 1;
			remap_[aDNN_TENSOR_BATCH] = 0;
			dims_[3] = (c_descr.dims[0] == 0) ? 1 : c_descr.dims[0];
			strides_[3] = (c_descr.strides[0] == 0) ? dims_[3] : c_descr.strides[0];
			break;

		case ADNN_BF_WHCN:
			remap_[aDNN_TENSOR_WIDTH] = 0;
			remap_[aDNN_TENSOR_HEIGHT] = 1;
			remap_[aDNN_TENSOR_DEPTH] = 2;
			remap_[aDNN_TENSOR_BATCH] = 3;

			dims_[0] = (c_descr.dims[0] == 0) ? 1 : c_descr.dims[0];
			dims_[1] = (c_descr.dims[1] == 0) ? 1 : c_descr.dims[1];
			dims_[2] = (c_descr.dims[2] == 0) ? 1 : c_descr.dims[2];
			dims_[3] = (c_descr.dims[3] == 0) ? 1 : c_descr.dims[3];
			strides_[0] = (c_descr.strides[0] == 0) ? dims_[0] : c_descr.strides[0];
			strides_[1] = (c_descr.strides[1] == 0) ? dims_[1] : c_descr.strides[1];
			strides_[2] = (c_descr.strides[2] == 0) ? dims_[2] : c_descr.strides[2];
			strides_[3] = (c_descr.strides[3] == 0) ? dims_[3] : c_descr.strides[3];
			break;

		case ADNN_BF_WHN:

			remap_[aDNN_TENSOR_WIDTH] = 0;
			remap_[aDNN_TENSOR_HEIGHT] = 1;
			remap_[aDNN_TENSOR_DEPTH] = 2;
			remap_[aDNN_TENSOR_BATCH] = 3;


			dims_[0] = (c_descr.dims[0] == 0) ? 1 : c_descr.dims[0];
			dims_[1] = (c_descr.dims[1] == 0) ? 1 : c_descr.dims[1];
			dims_[3] = (c_descr.dims[2] == 0) ? 1 : c_descr.dims[2];
			strides_[0] = (c_descr.strides[0] == 0) ? dims_[0] : c_descr.strides[0];
			strides_[1] = (c_descr.strides[1] == 0) ? dims_[1] : c_descr.strides[1];
			strides_[3] = (c_descr.strides[2] == 0) ? dims_[3] : c_descr.strides[2];

			break;

		case ADNN_BF_WN:

			remap_[aDNN_TENSOR_WIDTH] = 0;
			remap_[aDNN_TENSOR_HEIGHT] = 1;
			remap_[aDNN_TENSOR_DEPTH] = 2;
			remap_[aDNN_TENSOR_BATCH] = 3;

			dims_[0] = (c_descr.dims[0] == 0) ? 1 : c_descr.dims[0];
			dims_[3] = (c_descr.dims[1] == 0) ? 1 : c_descr.dims[1];
			strides_[0] = (c_descr.strides[0] == 0) ? dims_[0] : c_descr.strides[0];
			strides_[3] = (c_descr.strides[1] == 0) ? dims_[3] : c_descr.strides[1];

			break;
		case ADNN_BF_WH:

			remap_[aDNN_TENSOR_WIDTH] = 0;
			remap_[aDNN_TENSOR_HEIGHT] = 1;
			remap_[aDNN_TENSOR_DEPTH] = 2;
			remap_[aDNN_TENSOR_BATCH] = 3;

			dims_[0] = (c_descr.dims[0] == 0) ? 1 : c_descr.dims[0];
			dims_[1] = (c_descr.dims[1] == 0) ? 1 : c_descr.dims[1];
			strides_[0] = (c_descr.strides[0] == 0) ? dims_[0] : c_descr.strides[0];
			strides_[1] = (c_descr.strides[1] == 0) ? dims_[1] : c_descr.strides[1];

			break;

		default:
			printf("Data error: unknown batch format %d\n", batch_format_);
			break;
		}

		// element stride
		int e_stride = 0;
		switch (data_format_)
		{
		case ADNN_DF_UI32:
		case ADNN_DF_I32:
		case ADNN_DF_FP32:
			e_stride = 4;
			break;
		case ADNN_DF_FP64:
			e_stride = 8;
			break;
		case ADNN_DF_UI16:
		case ADNN_DF_I16:
		case ADNN_DF_FP16:
			e_stride = 2;
			break;
		case ADNN_DF_UI8:
		case ADNN_DF_I8:
			e_stride = 1;
			break;

		default:
			printf("Data error: unknown data format %d\n", data_format_);
			break;
		}
		strides_[remap_[aDNN_TENSOR_0DIM]] = e_stride;

		calculate();
	}


	aDNNTensor::aDNNTensor(const aDNNTensor & rh)
	{
		*this = rh;
	}


	aDNNTensor::~aDNNTensor()
	{
	}


	const aDNNTensor & aDNNTensor::operator=(const aDNNTensor & rh)
	{

		*(ADNNBase*)this = *(ADNNBase*)&rh;
		setInternal(rh.getInternal());
		data_format_ = rh.data_format_;
		batch_format_ = rh.batch_format_;

//		size_ = rh.size_;
//		size_bytes_ = rh.size_bytes_;
		control_bits_ = rh.control_bits_;

		sys_mem_ = rh.sys_mem_;
		ocl_mem_ = rh.ocl_mem_;

		allocated_ = rh.allocated_;

		context_ = rh.context_;

		remap_.clear();
		dims_.clear();
		strides_.clear();
		n_dims_ = rh.n_dims_;

		for (int i = 0; i < rh.remap_.size(); ++i)
		{
			remap_.push_back(rh.remap_[i]);
		}

		for (int i = 0; i < rh.dims_.size(); ++i)
		{
			dims_.push_back(rh.dims_[i]);
		}
		for (int i = 0; i < rh.strides_.size(); ++i)
		{
			strides_.push_back(rh.strides_[i]);
		}

		calculate();

		return *this;
	}

	size_t aDNNTensor:: getDim(int dim) const
	{
		assert(dim < remap_.size());
		return(dims_[remap_[dim]]);
	}


	size_t aDNNTensor:: getStride(int dim) const
	{
		size_t ret = 0;
		assert(dim < remap_.size());

		if (dim == aDNN_TENSOR_0DIM)
		{
			ret = strides_[remap_[dim]];
		}
		else
		{
			ret = 1;
			for (int i = aDNN_TENSOR_WIDTH; i <= dim; ++i)
			{
				ret *= strides_[remap_[i]];
			}

		}
		return(ret);
	}


	void aDNNTensor::calculate(void)
	{
		size_ = 0;
		size_bytes_ = 0;


		if (strides_.size() > 1)
		{
			size_ = 1;
			for (int i = aDNN_TENSOR_WIDTH; i < aDNN_TENSOR_5THDIM; ++i)
			{
				size_ *= strides_[remap_[i]];
			}

			size_bytes_ = size_ * strides_[remap_[aDNN_TENSOR_0DIM]];

		}
	}


	int aDNNTensor:: initTensor(const adnn_data_init & data_init, cl_command_queue queue)
	{
		int ret = 0;
		return(ret);
	}

	int aDNNTensor::allocTensor(uint flags, void * data_ptr, cl_context context)
	{
		int ret = 0;
		return(ret);
	}

	int aDNNTensor::getParams(adnn_data_parameters & c_descr) const
	{
		int ret = 0;
		c_descr.n_dims = (int)n_dims_;
		c_descr.data_format = data_format_;
		c_descr.batch_format = batch_format_;

		c_descr.size = size_;
		c_descr.size_bytes = size_bytes_;

		c_descr.sys_mem = sys_mem_;
		c_descr.ocl_mem = ocl_mem_;
		switch (batch_format_)
		{
		case ADNN_BF_NCHW:
			c_descr.dims[0] = dims_[0];
			c_descr.dims[1] = dims_[1];
			c_descr.dims[2] = dims_[2];
			c_descr.dims[3] = dims_[3];
			c_descr.strides[0] = strides_[0];
			c_descr.strides[1] = strides_[1];
			c_descr.strides[2] = strides_[2];
			c_descr.strides[3] = strides_[3];

			break;

		case ADNN_BF_NHW:

			c_descr.dims[0] = dims_[0];
			c_descr.dims[1] = dims_[2];
			c_descr.dims[2] = dims_[3];
			c_descr.strides[0] = strides_[0];
			c_descr.strides[1] = strides_[2];
			c_descr.strides[2] = strides_[3];

			break;

		case ADNN_BF_NW:

			c_descr.dims[0] = dims_[0];
			c_descr.dims[1] = dims_[3];
			c_descr.strides[0] = strides_[0];
			c_descr.strides[1] = strides_[3];

			break;

		case ADNN_BF_HW:

			c_descr.dims[0] = dims_[2];
			c_descr.dims[1] = dims_[3];
			c_descr.strides[0] = strides_[2];
			c_descr.strides[1] = strides_[3];

			break;
		case ADNN_BF_W:
			c_descr.dims[0] = dims_[3];
			c_descr.strides[0] = strides_[3];
			break;

		case ADNN_BF_WHCN:

			c_descr.dims[0] = dims_[0];
			c_descr.dims[1] = dims_[1];
			c_descr.dims[2] = dims_[2];
			c_descr.dims[3] = dims_[3];
			c_descr.strides[0] = strides_[0];
			c_descr.strides[1] = strides_[1];
			c_descr.strides[2] = strides_[2];
			c_descr.strides[3] = strides_[3];


			break;

		case ADNN_BF_WHN:

			c_descr.dims[0] = dims_[0];
			c_descr.dims[1] = dims_[1];
			c_descr.dims[2] = dims_[3];
			c_descr.strides[0] = strides_[0];
			c_descr.strides[1] = strides_[1];
			c_descr.strides[2] = strides_[3];

			break;

		case ADNN_BF_WN:

			c_descr.dims[0] = dims_[0];
			c_descr.dims[1] = dims_[3];
			c_descr.strides[0] = strides_[0];
			c_descr.strides[1] = strides_[3];


			break;
		case ADNN_BF_WH:

			c_descr.dims[0] = dims_[0];
			c_descr.dims[1] = dims_[1];
			c_descr.strides[0] = strides_[0];
			c_descr.strides[1] = strides_[1];

			break;

		default:
			printf("Data error: unknown batch format %d\n", batch_format_);
			break;
		}


		return(ret);
	}

	void * aDNNTensor::accessTensor(unsigned int flags, cl_command_queue queue)
	{
		void * ret = NULL;
		return(ret);
	}

	void aDNNTensor::commitTensor(void)
	{

	}

	const aDNNTensor & aDNNTensor::mul2(
		size_t c_cols, size_t c_rows,
		aDNNTensor &  tA, size_t a_cols, size_t a_rows,
		aDNNTensor & tB, size_t b_cols, size_t b_rows,
		int traspose_A, int transpose_B, double alpha,double beta,
		cl_command_queue queue
		)
	{
		return(*this);
	}



}; // adnn

