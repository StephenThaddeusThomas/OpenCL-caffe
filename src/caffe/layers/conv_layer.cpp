// Modified for Stage5 
// This is the mutilated file for testing Stage3 and Stage4
// See /c/AMD/MLopen/hybrid/starg4/caffe-conv_layer.cpp.notes for questions 
// Stage4 changed - not passing vector<Blob> just single blobs
// root:/repo/stt/OpenCL-caffe --> work: /c/AMD/MLopen/caffe/
// path:src/caffe/layers/conv_layer.cpp
// file:conv_layer.cpp
// host:thaddeus-nn  (Ubuntu 15.10 with Fury X R9 AMD GPU)
// date:160219 
// Search TT:Jinlu for Notes from today's meeting
// See Junli.notes in /c/AMD/notes 
// 1603160358 : cleaned out test stuff fron Junli - see conv_layer.cpp.stage3 for previous

#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

extern    alib_obj caffe::adnn_lib_object;

// TT:1603100356 Stage3  this had vectors
// Stage3 : template <typename T> int adnn_run_conv_layer(alib_obj lib_handle, const std::vector<caffe::Blob<T>*>&src, const std::vector<caffe::Blob<T>*>&des);
// TT:1603160351 Stage4  this usis a single blob.  The 'bridge' will call the appropriate methods

extern void adnn_init_weight(int chan, int height, int width) ;
extern void adnn_init_bias(int chan, int height, int width);
extern int  adnn_init_layer_infer(alib_obj lib_handle);
template<typename T> int  adnn_run_forward(const caffe::Blob<T> *src_blob, caffe::Blob<T> *des_blob);
extern int adnn_run_backward();
template <typename T> int adnn_term_layer(alib_obj lib_handle,caffe::Blob<T> *top_blob);

// Stage4.9 
extern int adnn_setup_layer(alib_obj lib_handle,
			    int kernel_h, int kernel_w,
			    int stride_h, int stride_w,
			    int num, int channels,
			    int pad_h, int pad_w,
			    int height, int width,
			    int num_output);

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape()
{
  this->height_out_ = (this->height_ + 2 * this->pad_h_ - this->kernel_h_) / this->stride_h_ + 1;
  this->width_out_ = (this->width_ + 2 * this->pad_w_ - this->kernel_w_)   / this->stride_w_ + 1;
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  // First call the base class methoed - we need the values to send to ADNN
  BaseConvolutionLayer<Dtype>::LayerSetUp(bottom,top);
  
  // Stage4.9: Setup the Layer in the ADNN Framework. It will use these to PrepareConvNode
  // In Stage4.8 the following call was at the end of BaseConvolutionLayer - but was being called numerious times

    // PATCH - 1603180548 - added this to see if I can find appropriate values to send instead of num_, width_ and height_ 
    std::cout << "ConvolutionLayer DUMP"
	      << "\n\t kernel_h_  :" << this->kernel_h_
	      << "\n\t kernel_w_  :" << this->kernel_w_
	      << "\n\t stride_h_  :" << this->stride_h_
	      << "\n\t stride_w_  :" << this->stride_w_
	      << "\n\t num_       :" << this->num_
	      << "\n\t channels_  :" << this->channels_
	      << "\n\t pad_h_     :" << this->pad_h_
	      << "\n\t pad_w_     :" << this->pad_w_
	      << "\n\t height_    :" << this->height_
	      << "\n\t widht_     :" << this->width_
	      << "\n\t num_output :" << this->num_output_
	      << "\n\t group_     :" << this->group_
	      << "\n\t height_out_     : " << this->height_out_
	      << "\n\t width_out_      : " << this->width_out_
	      << "\n\t bias_term_      : " << this->bias_term_
      // private
	      << "\n\t conv_out_channels_    :" << this->conv_out_channels_
	      << "\n\t conv_in_channels_     :" << this->conv_in_channels_
	      << "\n\t conv_out_spatial_dim_ :" << this->conv_out_spatial_dim_
	      << "\n\t conv_in_height_       :" << this->conv_in_height_
	      << "\n\t conv_in_width_        :" << this->conv_in_width_
	      << "\n\t kernel_dim_           :" << this->kernel_dim_
	      << std::endl;

  // Stage4: Use the values calculated in the Base LayerSetup to configure ADNN Node
  // NOTE: that adnn_setup_layer may be ignoring num_ 
  int rc=adnn_setup_layer(caffe::adnn_lib_object, this->kernel_h_, this->kernel_w_,
			  this->stride_h_, this->stride_w_, this->num_, this->channels_,
			  this->pad_h_, this->pad_w_, this->height_, this->width_, this->num_output_);

  // new Stage5: with insight from ALEX, see ConvParamMapping.notes 160318
  adnn_init_bias(1,  this->height_out_,  this->width_out_);
  adnn_init_weight(this->conv_in_channels_, this->kernel_h_, this->kernel_w_);

  // new Stage5: this was called in Forward_gpu  (V3) also not sending any blobs 
  rc=adnn_init_layer_infer(adnn_lib_object);  
  std::cout << "RC:"<< rc << std::endl;
}
  
template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
#ifdef  ORIGINAL_JUNLI_CODE
#error  "We don't want this = we want adnn "

  const Dtype* weight = this->blobs_[0]->cpu_data();

  for (int i = 0; i < bottom.size(); ++i)
    {
      const Dtype* bottom_data = bottom[i]->cpu_data();
      Dtype* top_data = top[i]->mutable_cpu_data();
      
      for (int n = 0; n < this->num_; ++n)
	{
	  this->forward_cpu_gemm(bottom_data + bottom[i]->offset(n), weight, top_data + top[i]->offset(n));  // vision_layers.hpp
	  if (this->bias_term_)
	    {
	      const Dtype* bias = this->blobs_[1]->cpu_data();
	      this->forward_cpu_bias(top_data + top[i]->offset(n), bias);
	    }
	}
    }
  // CHECK_BLOB_DATA(top[0],20, "top[0]");
#else
    // NOTE : Alex only wants INFER (or Forward) - 1603160416 - saving this for later
  std::cout << "TT: File:"<<__FILE__<<':'<<__LINE__<<" Forward_cpu Stage4 NOOP "<< std::endl;
#endif
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
#ifdef  ORIGINAL_JUNLI_CODE
#error  "We don't want this = we want to run adnn "

  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();

  for (int i = 0; i < top.size(); ++i)
    {
      const Dtype* top_diff = top[i]->cpu_diff();
      const Dtype* bottom_data = bottom[i]->cpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
      
      // Bias gradient, if necessary.
      if (this->bias_term_ && this->param_propagate_down_[1])
	{
	  Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
	  for (int n = 0; n < this->num_; ++n)
	    {
	      this->backward_cpu_bias(bias_diff, top_diff + top[i]->offset(n));
	    }
	}
      
      if (this->param_propagate_down_[0] || propagate_down[i])
	{
	  for (int n = 0; n < this->num_; ++n)
	    {
	      // gradient w.r.t. weight. Note that we will accumulate diffs.
	      if (this->param_propagate_down_[0])
		{
		  this->weight_cpu_gemm(bottom_data + bottom[i]->offset(n), top_diff + top[i]->offset(n), weight_diff);
		}
	      // gradient w.r.t. bottom data, if necessary.
	      if (propagate_down[i])
		{
		  this->backward_cpu_gemm(top_diff + top[i]->offset(n), weight, bottom_diff + bottom[i]->offset(n));
		}
	    }
	}
    }
#else
  // NOTE : Alex only wants INFER (or Forward) - 1603160416 - saving this for later
  std::cout << "TT: File:"<<__FILE__<<':'<<__LINE__<<" Backward_cpu Stage4 NOOP "<< std::endl;
#endif
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  // 160316 : Stage4.Infer.1.fail.txt First Run FAILED
  // Going to test a couple of versions
  // V1: commenting out the INNER items
  // V2: comment out the OUTER items
  // V3: see SetupLayer()
  
  int n,rc=0,max=bottom.size();
  for(n=0;n<max;n++)
    {
      std::cout << "TT: File:"<<__FILE__<<':'<<__LINE__<< " Forward_gpu Stage5 V3 ["<<n<<"] Calling Init and RunForward for ["<< max << "] blobs" << std::endl;
      rc=adnn_run_forward(bottom[n],top[n]);
      std::cout << "TT: Forward_gpu Stage5  n["<<n<<"] Return Adnn_Run_Conv_Layer =" << rc << std::endl;
    }
  //rc=adnn_term_layer(adnn_lib_object,top[n]); // should this be in loop, or out, if out, what [N] should it use
  //I don't think I need to do this as I set the point of mutuable from top blob to node_sink - BUT NOT SURE
 }

// is the 'top' the 'source' and 'bottom' the sink for Backward ??
template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
  /***********************************************************************************
  // NEED TO DISCUSS WITH ALX
  int rc=0,n=0,max=top.size();
  adnn_init_bias(this->blobs_[1]);
  adnn_init_weight(this->blobs_[0]);
  rc=adnn_init_layer_train(adnn_lib_object,bottom[n],top[n]);      //  HERE or....
  for(n=0;n<max;n++)
    {
      rc=adnn_init_layer_train(adnn_lib_object,bottom[n],top[n]);  // ... HERE

      // Bias gradient, if necessary.
      if (this->bias_term_ && this->param_propagate_down_[1])
	Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();  //  ??? What should I do here
      
      std::cout << "TT: Backward_gpu Stage4 n["<<n<<"] Calling Adnn_Run_Backward for "<< max << " blobs" << std::endl;
      rc=adnn_run_backward();
      std::cout << "TT: Backward_gpu Stage4  n["<<n<<"] Return Adnn_Run_Backward rc =" << rc << std::endl;
      
    }
  // ?? Should this be here? or in the loop, should it use the Top or Bottom, and if out of loop, what [N] ??
  rc=adnn_term_layer(adnn_lib_object,top[0]);   
  ************************************************************************************/
  std::cout << "TT: File:"<<__FILE__<<':'<<__LINE__<<" Backward_GPU Stage4 NOOP "<< std::endl;
}

INSTANTIATE_CLASS (ConvolutionLayer);

}  // namespace caffe
