// File:adnn_conv_layer.cpp
// 1603140256 : CLEANED ready to compile, but need to work on AdnnConvLayer
// 1603140254 : MERGED .part1 and .part2 ==> ,merge  THIS STILL HAS _gemm methods - to cut next
// Path:/c/AMD/MLopen/hybrid/stage4/src
// Date:160313
// From:base_conv_layers.cpp
// Path:/c/AMD/MLopen/caffe/src/caffe/layers
// Host:sys76
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////9012345678901234567890123456789012345678901234567890123456789012345678901234567890

#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
//#include "caffe/util/im2col.hpp"
//#include "caffe/util/math_functions.hpp"
//#include "caffe/vision_layers.hpp" replaced by adnn_layers.hpp
#include "caffe/common.hpp"

#include "/c/AMD/MLopen/hybrid/stage4/inc/adnn_layers.hpp"

// NOTES:
// - blobs_[0] holds the filter weights
// - blobs_[1] holds the biases (optional)
// NOTE: CODE is ONLY GPU - CPU methods are empty

// BRIDGING between Caffe and the yet refactored ADNN - (when refactored we won't need these, but will have a true 'Setuplayer', 'Forward_gpu' and 'Backward_gpu' 
// TT: 1603100356 This was tested in Stage3 (but it was adnn_run_conv_layer)
// TT: 1603140213 In Stage4 we are going to be using the Blobs ALSO there is now a Forward and Backward version

extern    alib_obj caffe::adnn_lib_object;
template <typename T> int adnn_init_layer_train(alib_obj lib_handle, const caffe::Blob<T>*src, caffe::Blob<T>*des);
extern  int adnn_run_foward();
extern  int adnn_run_backward();

template <typename T> int adnn_setup_conv_layer(alib_obj lib_handle, >>>TODO<<<<);  // NEW Stage4

namespace caffe {

// >> NOTE: This usese amdDevice.Context <<<< need to replace with the Adnn context (which is held in _adnn_lib_paramters, but there are no methods to extract it)
// It IS defined as 1 on line 97 of common.hpp in caffe/include/caffe
// >> SO for now undefining the packing scheme
#undef use_packing_scheme
#ifdef use_packing_scheme
template <typename Dtype> size_t AdnnConvolutionLayer<Dtype>::subtop_mem_size = sizeof(Dtype);
template <typename Dtype> size_t AdnnConvolutionLayer<Dtype>::trans_mem_size = sizeof(Dtype);
template <typename Dtype> cl_mem AdnnConvolutionLayer<Dtype>::subTopMem = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, BaseConvolutionLayer<Dtype>::subtop_mem_size, NULL, NULL);
template <typename Dtype> cl_mem AdnnConvolutionLayer<Dtype>::transMem = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, BaseConvolutionLayer<Dtype>::trans_mem_size, NULL, NULL);
template <typename Dtype> void   Alloc_public_tmp_mem(size_t subtop_size, size_t trans_size)
{
  if (subtop_size > AdnnConvolutionLayer < Dtype > ::subtop_mem_size)
    {
      // TT: this was (still is) ConvolutionLayer FIX if we need it for ADNN - moving on
      ConvolutionLayer < Dtype > ::subtop_mem_size = subtop_size;
      clReleaseMemObject(ConvolutionLayer < Dtype > ::subTopMem);
      ConvolutionLayer < Dtype > ::subTopMem = clCreateBuffer(amdDevice.Context,CL_MEM_READ_WRITE, BaseConvolutionLayer < Dtype > ::subtop_mem_size, NULL, NULL);
    }

  if (trans_size > ConvolutionLayer < Dtype > ::trans_mem_size)
    {
    ConvolutionLayer < Dtype > ::trans_mem_size = trans_size;
    clReleaseMemObject(ConvolutionLayer < Dtype > ::transMem);
    ConvolutionLayer < Dtype > ::transMem = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, BaseConvolutionLayer < Dtype > ::trans_mem_size, NULL, NULL);
    }
}
#endif
  
template <typename Dtype>
void AdnnConvolutionLayer<Dtype>::ocl_setup()
{
  M_ = num_output_ / group_;
  K_ = conv_in_channels_ * kernel_w_ * kernel_h_ / group_;
  N_ = height_out_ * width_out_;
#ifdef use_packing_scheme
  size_t subtop_size = (size_t)((M_ * group_) * N_ * global_packing_N * sizeof(Dtype));
  size_t trans_size = (size_t)((K_ * group_ )* N_ * global_packing_N * sizeof(Dtype));
  Alloc_public_tmp_mem<Dtype>(subtop_size, trans_size);
#endif
}

template <typename Dtype>
void AdnnConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  // TT: removed ALL of the checks (we don't get these errors in cifar10 tests)
  // TT: want to keep this clean so we can figure out the MainFlow which is setting up for Adnn

  
  if (conv_param.has_kernel_size())
    {
      kernel_h_ = kernel_w_ = conv_param.kernel_size();
    }
   else
     {
       kernel_h_ = conv_param.kernel_h();
       kernel_w_ = conv_param.kernel_w();
     }

  if (!conv_param.has_pad_h())
    {
      pad_h_ = pad_w_ = conv_param.pad();
    }
  else
    {
      pad_h_ = conv_param.pad_h();
      pad_w_ = conv_param.pad_w();
    }
  
  if (!conv_param.has_stride_h())
    {
      stride_h_ = stride_w_ = conv_param.stride();
    }
  else
    {
      stride_h_ = conv_param.stride_h();
      stride_w_ = conv_param.stride_w();
    }
  
  // Configure output channels and groups.
  channels_   = bottom[0]->channels();
  num_output_ = this->layer_param_.convolution_param().num_output();
  group_      = this->layer_param_.convolution_param().group();
  bias_term_  = this->layer_param_.convolution_param().bias_term();
  
  // if (reverse_dimensions()) returns False by definition in adnn_layer.hpp so removed first stanza where found
  conv_out_channels_ = num_output_; // second stanze
  conv_in_channels_  = channels_;   // second stanza
  
  if (this->blobs_.size() > 0)
    {
      LOG(INFO) << "Skipping parameter initialization";
    }
  else
    {
      if (bias_term_)
	{
	  this->blobs_.resize(2);
	}
      else
	{
	  this->blobs_.resize(1);
	}
      
      // Initialize and fill the weights:
      // output channels x input channels per-group x kernel height x kernel width
      // [0] is filter weights
      this->blobs_[0].reset(new Blob<Dtype>(conv_out_channels_, conv_in_channels_ / group_, kernel_h_, kernel_w_));
      
      shared_ptr < Filler<Dtype> > weight_filler( GetFiller<Dtype> (this->layer_param_.convolution_param().weight_filler()));
      
      weight_filler->Fill(this->blobs_[0].get());

      // If necessary, initialize and fill the biases.
      if (bias_term_)
	{
	  vector<int> bias_shape(1, num_output_);
	  this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
	  
	  shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype> (this->layer_param_.convolution_param().bias_filler()));
	  
	  bias_filler->Fill(this->blobs_[1].get());
	}
    }
  
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  ///// Stage4 - creating a new function to 'setup' adnn
  int adnn_setup_conv_layer( >>>> LIST OF PARAMETER NEEDED TODO <<<< ); // see line 211 for 'template'
}

template <typename Dtype>
void AdnnConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  num_    = bottom[0]->num();
  height_ = bottom[0]->height();
  width_  = bottom[0]->width();
  
  // Shape the tops.
  compute_output_shape();
  
  for (int top_id = 0; top_id < top.size(); ++top_id)
    {
      top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
    }
  
  //if (reverse_dimensions())
  conv_in_height_       = height_;
  conv_in_width_        = width_;
  conv_out_spatial_dim_ = height_out_ * width_out_;
  
  kernel_dim_    = conv_in_channels_  * kernel_h_    * kernel_w_;
  weight_offset_ = conv_out_channels_ * kernel_dim_  / group_ / group_;
  col_offset_    = kernel_dim_        * conv_out_spatial_dim_ / group_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  
  col_buffer_.Reshape(1, kernel_dim_, height_out_, width_out_);   // 1x1_ issue   // Where col_buffer_ 
  
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  if (bias_term_)
    {
      vector<int> bias_multiplier_shape(1, height_out_ * width_out_);
      
      bias_multiplier_.Reshape(bias_multiplier_shape);  // where bias_multiplier

      caffe_set(bias_multiplier_.count(), Dtype(1), bias_multiplier_.mutable_cpu_data());     // <<<<< ??? caffe_set 
    }
  
  //initializa OpenCL kernels and cl_mem objects
  ocl_setup();

  // Stage4 : Need to communicate the new parameters
  // TODO : create new method 
}

  ////////////// END of the BaseConvolutionLayer (base_convo_layer.cpp) adnn_conv_layer.cpp.part1 /////////////
  /////////////  START of ConvolutionLayer (conv_layer.cpp) adnn_conv_layer.cpp.part2             /////////////
  // File:adnn_conv_layer.cpp.part2
  // root:/repo/stt/OpenCL-caffe --> work: /c/AMD/MLopen/caffe/
  // From:/c/AMD/MLopen/caffe/src/caffe/layers/conv_layer.cpp
  // host:thaddeus-nn  (Ubuntu 15.10 with Fury X R9 AMD GPU)
  // date:160219 
  // See Junli.notes in /c/AMD/notes 

template <typename Dtype>
void AdnnConvolutionLayer<Dtype>::compute_output_shape()
{
  this->height_out_ = (this->height_ + 2 * this->pad_h_ - this->kernel_h_) / this->stride_h_ + 1;
  this->width_out_ = (this->width_ + 2 * this->pad_w_ - this->kernel_w_)   / this->stride_w_ + 1;
}

template <typename Dtype>
void AdnnConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
}

template <typename Dtype>
void AdnnConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
}

// This is modeled after conv_layer.cpp Forward_gpu_batched, but I really hacked it down
// ISSUE : we may not want to be calling adnn_init_layer_train within the loop
// OR (and perhaps better we break up adnn_init_layer_train|infer into two functions
// The first adnn_create_layer_train will do all the contruction of empty items
// Then we call adnn_init_layer_train to initialize it with blob data  (we called adnn_init_weight before)
// BUT for now keeping it like this - to initialize for Each blob - then call run_forward
template <typename Dtype>
void AdnnConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom_src, const vector<Blob<Dtype>*>& top_des)
{
  adnn_init_weight(blobs_[0]);

  for (int i = 0; i < bottom.size(); ++i)             // For all the Input Blobs
    {
      adnn_init_layer_train(caffe::adnn_lib_object, src[i], des[i]);
      std::cout << "TT: Forward_gpu Stage4 Calling Adnn_Run_Forward"<< std::endl;
      int rc=adnn_run_forward(adnn_lib_object,bottom[i],top[i]);    // OBVIOUSLY WRONG
      std::cout << "TT: Forward_gpu Stage4 Return Adnn_Run_Forward =" << rc << std::endl;
    }
}

template <typename Dtype>
void AdnnConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
  const Dtype* weight      = this->blobs_[0]->cpu_data();
        Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();

	for (int i = 0; i < top.size(); ++i)
	  {
	     std::cout << "TT: Backward_gpu Stage4 Calling Adnn_Run_Backward"<< std::endl;
	     int rc=adnn_run_backward(adnn_lib_object,bottom,top,propogate_down);   // Stage4 NEW MEHTOD NAME
	     std::cout << "TT: Backward_gpu Stage4 Return Adnn_Run_Backward =" << rc << std::endl;
	  }
}

  //CPU ONLY MODE
  //STUB_GPU(ConvolutionLayer);

INSTANTIATE_CLASS (AdnnConvolutionLayer);

}  // namespace caffe
