/* File:AdnnSetupLayer.cpp
** Path:/c/AMD/MLopen/hybrid/stage4/src/
** Date:1603140354
*/
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <CL/opencl.h>
#include "/c/AMD/MLopen/adnn/inc/AMDnn.h"

// LIBRARY GLOBALS.  Using adnn_ instead of namespace
// These ints get set from call by BaseConvolutionLayer::SetupLayers
// These are the default values (from aLibADNNDriver.cpp)

  int  adnn_n_categories = 100;
  int  adnn_batch_sz = 100;
  int  adnn_input_channels = 3;
  int  adnn_input_w = 32;
  int  adnn_input_h = 32;
  int  adnn_n_output_features = 32;  // same as n_output_featuremaps
  int  adnn_filter_sz = 3;
  int  adnn_stride = 1;

// These need to be set somehow (can be set via running main4.cpp)
  int  adnn_max_iterations = 400;
  int  adnn_per_layer_iters = 1000;
  bool adnn_conv_only = true;    // enabled but not used
  bool adnn_do_training = true;
  bool adnn_per_layer_timing = false;
  bool adnn_net_timing = true;
  bool adnn_verify = true;       // 160315-enabled

// These could be members of the AdnnConvolutionLayer
  adnn_control_params      adnn_layer_control;
  adnn_filter1D_parameters adnn_filter_params;
  adnn_update_params       adnn_update_train_params;     // only need for Training

// Phase1 : Setup 
// This is called by AdnnConvolutionLayer::SetupLayer()

// don't need this to be templated
// template <typename D>
int adnn_setup_layer(alib_obj lib_handle,
		     int kernel_h, int kernel_w,
		     int stride_h, int stride_w,
		     int num, int channels,
		     int pad_h, int pad_w,
		     int height, int width,
		     int num_output)
{
  std::cout << "AdnnSetupLayer: BEFORE \n\t n_categories:"<<adnn_n_categories
	    << "\n\t batch_sz:"<<adnn_batch_sz
	    << "\n\t input_channels:"<<adnn_input_channels
	    << "\n\t input_w:"<<adnn_input_w
	    << "\n\t input_h:"<<adnn_input_h
	    << "\n\t n_output_features:"<<adnn_n_output_features
	    << "\n\t filter_sz:" << adnn_filter_sz
	    << "\n\t stride:" << adnn_stride
	    << std::endl;

  // Step1.1: Init Variables (non-structures)
  // Set the parameters, from values determined in AdnnConvolution::SetupLayers (adnn_conv_layer.cpp)

  // PATCH - 1603180548 - commented out the 3 parameters that don't get set correctly
  // PATCH - added in printout to conv_layer.cpp to see what all the available parameters are 
  adnn_n_categories      = num_output;
  //adnn_batch_sz          = num;
  adnn_input_channels    = channels;
  //adnn_input_w           = width;
  //adnn_input_h           = height;
  adnn_n_output_features = num_output;
  adnn_filter_sz         = kernel_h;	// Caffe has kernel_h and kernel_w, only using one TBD
  adnn_stride            = stride_h;	// Caffe has stride_h and stride_w, only using h

  std::cout << "AdnnSetupLayer: AFTER \n\t n_categories:"<<adnn_n_categories
	    << "\n\t batch_sz:"<<adnn_batch_sz
	    << "\n\t input_channels:"<<adnn_input_channels
	    << "\n\t input_w:"<<adnn_input_w
	    << "\n\t input_h:"<<adnn_input_h
	    << "\n\t n_output_features:"<<adnn_n_output_features
	    << "\n\t filter_sz:" << adnn_filter_sz
	    << "\n\t stride:" << adnn_stride
	    << std::endl;
  
  // These stanza came from Stage3/src/AdnnConvLayer.cpp  adnn_run_conv_layer_backward()

  // Step1.2: InitParmaters (structures)
  if(adnn_verify) {  adnn_max_iterations = adnn_per_layer_iters = 1; }

  memset(&adnn_layer_control, 0, sizeof(adnn_control_params));
  adnn_layer_control.per_layer_iter      = 1;
  adnn_layer_control.per_layer_messages  = (adnn_net_timing == false);
  adnn_layer_control.per_layer_timing    = adnn_per_layer_timing;
  adnn_layer_control.per_layer_iter      = adnn_per_layer_iters;
  adnn_layer_control.debug_level         = (adnn_verify) ? 1 : 0;

  adnn_filter_params.size   = adnn_filter_sz;
  adnn_filter_params.pad    = (adnn_filter_sz-1) / 2;
  adnn_filter_params.stride = adnn_stride;
  
  memset(&adnn_update_train_params, 0, sizeof(adnn_update_params));
  adnn_update_train_params.weights_lr.policy = ADNN_LP_LINEAR;
  adnn_update_train_params.weights_lr.base   = 0.001;
  adnn_update_train_params.bias_lr.policy    = ADNN_LP_FIXED;
  adnn_update_train_params.bias_lr.base      = 0.001;
  adnn_update_train_params.weights_momentum  = 0.9;
  adnn_update_train_params.bias_momentum     = 0;
  adnn_update_train_params.weights_decay     = 0.004;

  return(0);
}



