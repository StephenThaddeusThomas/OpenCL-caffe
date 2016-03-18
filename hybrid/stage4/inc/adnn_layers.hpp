// File:adnn_layers.hpp
// Path:/c/AMD/MLopen/hybrid/stage4
// From:vision_layers.hpp  (only the first two classes - may others hacked out)
// RefactorStage4.notes - copy this file to hybrid/stage4/(inc/)adnn_layers.hpp [160313]
// Path:/c/AMD/MLopen/caffe/inlcude/caffe
// Date:160310

// Revisions:
// draft1 : combined BaseConvolutionLayer and ConvolutionLayer into one file  TOP:BaseConvolutionLayer, BOT:ConvolutionLayer
// draft2 : cut out comments and reformated so I can think (see hardcopy for notes what we might want back)
// draft3 : merged the TOP and BOT parts together (public with public, protected and protected, etc) cleaned up

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TT: this is the conv_layer.cpp that we have modified in Stage3
// I should look at building the true C++ ADNN_layer : public ConvolutionLayer 


#ifndef ADNN_LAYERS_DEF
#define ADNN_LAYERS_DEF

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
//TT#include "caffe/common_layers.hpp"
//TT#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
//TT#include "caffe/loss_layers.hpp"
//TT#include "caffe/neuron_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

// This class is a hybrid of base_convolution_layter and convolutions_layers
template <typename Dtype>
class AdnnConvolutionLayer : public Layer<Dtype>
{
public:

  virtual inline const char* type() const { return "AdnnConvolution"; }

  explicit AdnnConvolutionLayer(const LayerParameter& param) : Layer<Dtype>(param)  { }
  virtual ~AdnnConvolutionLayer() { }

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,    const vector<Blob<Dtype>*>& top);

  virtual inline int MinBottomBlobs() const { return 1;  }
  virtual inline int MinTopBlobs() const    { return 1;  }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }

protected:
  
  /* These Forward|Backward_cpu|gpu need to be kept as part of interface TT 
  ** for DVLP/TEST I am putting the same calls adnn_run_conv_layer_forward|_backward 
  ** in BOTH the _cpu and _gpu calls. (obviously binding this code to a GPU based platform) 
  ** Could have it exit() on _cpu calls TBD
  */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& bot);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& bot);

  void compute_output_shape();  // Compute height_out_ and width_out_ from other parameters.
  void ocl_setup();             // opencl related setup

  // reverse_dimensions should return true iff we are implementing deconv, so
  // that conv helpers know which dimensions are which.
  bool reverse_dimensions() { return false ; }   // was: virtual and =0;
  
  /* OK, now these are all the parameters that should be set */
  /* key // 'sp' = setup means this variable is sent to adnn_setup_layer
  int kernel_h_, kernel_w_;	// sp
  int stride_h_, stride_w_;	// sp
  int num_;			// ?? what is this ??
  int channels_;		// sp
  int pad_h_, pad_w_;		// sp
  int height_, width_;		// sp
  int group_;			// not sent to setup
  int num_output_;		// sp
  int height_out_, width_out_;
  bool bias_term_;

  int conv_out_channels_;
  int conv_in_channels_;
  int conv_out_spatial_dim_;
  int conv_in_height_;
  int conv_in_width_;
  int kernel_dim_;

  Blob<Dtype> col_buffer_;
  Blob<Dtype> bias_multiplier_;

  //opencl related data structures
  int opt_num2;
  int M_, N_, K_;
  int weight_offset_;
  int col_offset_;
  int output_offset_;
  int top_offset_;
  int top_offset_opt;
  int bottom_offset_;

  static cl_mem subTopMem, transMem;
  static size_t subtop_mem_size, trans_mem_size;

//bool is_1x1_;     <<<< they are using this in calculation: "is_1x1_ * bottom_offset_ + col_offset_ * g," to call caffe_gpu_gemm (355 in previous draft  
};

}  // namespace caffek  <<<< adnn ??? 

#endif  // ADNN_LAYERS_DEF 
