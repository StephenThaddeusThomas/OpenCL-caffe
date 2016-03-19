/* File:main5.cpp
** Path:/c/AMD/MLopen/hybrid/stage4/src/
** Date:1603181914
*/
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <CL/opencl.h>
#include "/c/AMD/MLopen/adnn/inc/AMDnn.h"  // replace with <aDNN/AMDnn.h>
#include "/c/AMD/MLopen/caffe/include/caffe/blob.hpp"  // replace with <caffe/blob.hpp>
#include "/c/AMD/MLopen/caffe/include/caffe/common.hpp"

#if defined(TEST_TRAIN) && defined(TEST_INFER)
#error "Both TRAIN and INFER are defined"
#elif !defined(TEST_TRAIN) && !defined(TEST_INFER)
#error "Neither TRAIN nor INFER are define - need only one"
#endif

// defined in AdnnSetupLayer.cpp
extern int adnn_setup_layer(alib_obj lib_handle, int kernel_h, int kernel_w, int stride_h, int stride_w, int num, int channels,
			    int pad_h, int pad_w, int height, int width, int num_output);
extern void adnn_init_bias(int chan, int height, int width);
extern void adnn_init_weight(int chan, int height, int width);


#ifdef TEST_TRAIN
// defined in AdnnConvLayerTrain.cpp
template <typename T> int adnn_run_forward(const caffe::Blob<T> *src, caffe::Blob<T> *des);
extern int adnn_init_layer_train(alib_obj lib_handle);
#endif
  
#ifdef TEST_INFER
// define in AdnnConvLayerInfer.cpp
extern int adnn_init_layer_infer(alib_obj lib_handle);
template <typename T> int adnn_run_forward(const caffe::Blob<T> *src, caffe::Blob<T> *des);
extern int adnn_run_backward(alib_obj lib_handle);
#endif

// assign TRAIN or INFER  (Stage3 entry function, called from conv_layer.cpp in caffe/src/caffe/layers/)
template <typename T> int adnn_run_conv_layer(alib_obj lib_handle, const caffe::Blob<T> *src, caffe::Blob<T> *des);

template <typename D> int adnn_term_layer(alib_obj lib_handle,caffe::Blob<D> *top_blob);
extern int adnn_cleanup_layer(alib_obj lib_handle);


// assign TRAIN or INFER


// Stage4 : (intermediate) - making these global for the bridge library (so prefixed with adnn_)
// These eventually will be replaced by the protected members of  AdnnConvolutionLayer  (adnn_layers.hpp)
// These are extern here (main4.cpp) they live in adnn_setup_layer.cpp

extern int
  adnn_n_categories,
  adnn_max_iterations,
  adnn_per_layer_iters,
  adnn_batch_sz,
  adnn_input_channels,
  adnn_input_w,
  adnn_input_h,
  adnn_n_output_features,
  adnn_filter_sz,
  adnn_stride;
extern bool
  adnn_conv_only,
  adnn_do_training,
  adnn_per_layer_timing,
  adnn_net_timing,
  adnn_verify;


static void Usage(void)
{
  printf("Arguments:\n");
  printf("%*s" "-h" "%*s" "help.\n", 4, " ", 9, " ");
  printf("%*s" "-v" "%*s" "verify each layer.\n", 4, " ", 9, " ");
  printf("%*s" "-ni" "%*s" "# of iterations, default=400, max = 430, if -lt==true, it's set to 1.\n", 4, " ", 8, " ");
  printf("%*s" "-li" "%*s" "# of per layer iterations, default=1000, if -nt==true, it's set to 1.\n", 4, " ", 8, " ");
  printf("%*s" "-bz" "%*s" "# of batches, default=100.\n", 4, " ", 8, " ");
  printf("%*s" "-iw" "%*s" "input width, default=32.\n", 4, " ", 8, " ");
  printf("%*s" "-ih" "%*s" "input height, default=32.\n", 4, " ", 8, " ");
  printf("%*s" "-ic0" "%*s" "n input channels in 0 layer, default=3.\n", 4, " ", 8, " ");
  printf("%*s" "-oc0" "%*s" "n outputchnnels in 0 layer, default=32.\n", 4, " ", 8, " ");
  printf("%*s" "-fs" "%*s" "filter size, default=3.\n", 4, " ", 8, " ");
  printf("%*s" "-sd" "%*s" "convolution stride, default=1.\n", 4, " ", 8, " ");
  printf("%*s" "-l" "%*s" "per layer timing is true, default=false.\n", 4, " ", 8, " ");
  printf("%*s" "-n" "%*s" "net timing is false, default=true, if set it has a priority over the per layer timing.\n", 4, " ", 8, " ");
  printf("%*s" "-cnv" "%*s" "single convolution layer. default = full application.\n", 4, " ", 8, " ");
  printf("%*s" "-fw" "%*s" "forward propagation only. default = forward an dbackward propagations.\n", 4, " ", 8, " ");
  exit(0);
}

// For testing
int main(int argc, char* argv[])
{
  for (int i = 1; i < argc; i++)
    {
      std::string arg_prefix(argv[i]);
      if (arg_prefix == "-ni" && i < argc - 1)
	{
	  adnn_max_iterations = std::stoi(std::string(argv[++i]));
	  adnn_max_iterations = (adnn_max_iterations <= 0) ? 1 : (adnn_max_iterations > 430) ? 430 : adnn_max_iterations;
	}
      else if (arg_prefix == "-bz" && i < argc - 1)
	{
	  adnn_batch_sz = std::stoi(std::string(argv[++i]));
	  adnn_batch_sz = (adnn_batch_sz <= 0) ? 1 : adnn_batch_sz;
	}
      else if (arg_prefix == "-li" && i < argc - 1)
	{
	  adnn_per_layer_iters = std::stoi(std::string(argv[++i]));
	  adnn_per_layer_iters = (adnn_per_layer_iters <= 0) ? 1 : adnn_per_layer_iters;
	}
      else if (arg_prefix == "-ic0" && i < argc - 1)
	{
	  adnn_input_channels = std::stoi(std::string(argv[++i]));
	  adnn_input_channels = (adnn_input_channels <= 0) ? 1 : adnn_input_channels;
	}
      else if (arg_prefix == "-oc0" && i < argc - 1)
	{
	  adnn_n_output_features = std::stoi(std::string(argv[++i]));
	  adnn_n_output_features = (adnn_n_output_features <= 0) ? 1 : adnn_n_output_features;
	}
      else if (arg_prefix == "-iw" && i < argc - 1)
	{
	  adnn_input_w = std::stoi(std::string(argv[++i]));
	  adnn_input_w = (adnn_input_w <= 0) ? 1 : adnn_input_w;
	}
      else if (arg_prefix == "-ih" && i < argc - 1)
	{
	  adnn_input_h = std::stoi(std::string(argv[++i]));
	  adnn_input_h = (adnn_input_h <= 0) ? 1 : adnn_input_h;
	}
      else if (arg_prefix == "-fs" && i < argc - 1)
	{
	  adnn_filter_sz = std::stoi(std::string(argv[++i]));
	  adnn_filter_sz = (adnn_filter_sz <= 0) ? 3 : adnn_filter_sz;
	}
      else if (arg_prefix == "-sd" && i < argc - 1)
	{
	  adnn_stride = std::stoi(std::string(argv[++i]));
	  adnn_stride = (adnn_stride <= 0) ? 1 : adnn_stride;
	}
      else if (arg_prefix == "-v")
	{
	  adnn_verify = true;
	}
      else if (arg_prefix == "-cnv")
	{          
	  adnn_conv_only = true;
	}
      else if (arg_prefix == "-fw")
	{
	  adnn_do_training = false;
	}
      else if (arg_prefix == "-h" && i < argc - 1)
	{
	  Usage();
	}
      else if (arg_prefix == "-l")
	{
	  adnn_per_layer_timing = true;
	}
      else if (arg_prefix == "-n")
	{
	  adnn_net_timing = false;
	}
      else
	{
	  printf("Unrecognized parameter: \"%s\".\n", arg_prefix.c_str());
	  Usage();
	}
    }

  // 1. Need to create Bottom Blob and Top Blob
  // 2. Need to create the ADNN LIbrary
  adnn_lib_parameters  lib_params;
  memset(&lib_params, 0, sizeof(adnn_lib_parameters));

  lib_params.accel_type = CL_DEVICE_TYPE_GPU;
  lib_params.ocl_kernels_path = "../aLibDNN";     // <<<<< make a command line arg, or part of a config file
  alib_obj lib_handle = ADNNLibCreate(&lib_params);
  printf("Created ADNN library %s\n", ADNNLibGetName(lib_handle));
  
  // 3. call  adnn_run_conv_layer(lib, src, des)
  // Stage3 : these where vectors, now just single Blobs (not initialized/loaded)
  const caffe::Blob<float> src_blob;
        caffe::Blob<float> des_blob;
  int status=adnn_run_conv_layer(lib_handle,&src_blob,&des_blob);

  // 4. Need to delete ADNN Library
  ADNNLibDestroy(&lib_handle);

  return(0);
}
