// File:main.cpp (mini-main.cpp) runs only one convolution layer, either in training(forward/back) or inference (forward only)

#include <cstdio>       // this defines malloc also, so commented out the malloc.h below
#include <iomanip>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <vector>
#include <map>
#include <string>
#include <CL/opencl.h>
#include "../inc/AMDnn.h"
#include "../inc/AMDnn.hpp"   // one of these may be gone

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


extern int
ADNNSingleConvLayer(alib_obj aLib,
		    const adnn_control_params *layer_control,
		    const adnn_filter1D_parameters *filter_params,
		    int batch_sz,
		    int input_channels,
		    int input_h,
		    int input_w,
		    int n_output_featuremaps,
		    bool training,
		    adnn_update_params *pupdate_params);

#define TEST_CORRECT 0
int main(int argc, char* argv[])
{
  int n_categories = 100;
  int max_iterations = 400;
  int per_layer_iters = 1000;
  int batch_sz = 100;
  int input_channels = 3;
  int input_w = 32;
  int input_h = 32;
  bool per_layer_timing = false;
  bool net_timing = true;
  bool verify = false;
  int n_output_features = 32;
  int filter_sz = 3;
  int stride = 1;
  bool conv_only = false;
  bool do_training = true;

  for (int i = 1; i < argc; i++)
    {
      std::string arg_prefix(argv[i]);
      if (arg_prefix == "-ni" && i < argc - 1)
	{
	  max_iterations = std::stoi(std::string(argv[++i]));
	  max_iterations = (max_iterations <= 0) ? 1 : (max_iterations > 430) ? 430 : max_iterations;
	}
      else if (arg_prefix == "-bz" && i < argc - 1)
	{
	  batch_sz = std::stoi(std::string(argv[++i]));
	  batch_sz = (batch_sz <= 0) ? 1 : batch_sz;
	}
      else if (arg_prefix == "-li" && i < argc - 1)
	{
	  per_layer_iters = std::stoi(std::string(argv[++i]));
	  per_layer_iters = (per_layer_iters <= 0) ? 1 : per_layer_iters;
	}
      else if (arg_prefix == "-ic0" && i < argc - 1)
	{
	  input_channels = std::stoi(std::string(argv[++i]));
	  input_channels = (input_channels <= 0) ? 1 : input_channels;
	}
      else if (arg_prefix == "-oc0" && i < argc - 1)
	{
	  n_output_features = std::stoi(std::string(argv[++i]));
	  n_output_features = (n_output_features <= 0) ? 1 : n_output_features;
	}
      else if (arg_prefix == "-iw" && i < argc - 1)
	{
	  input_w = std::stoi(std::string(argv[++i]));
	  input_w = (input_w <= 0) ? 1 : input_w;
	}
      else if (arg_prefix == "-ih" && i < argc - 1)
	{
	  input_h = std::stoi(std::string(argv[++i]));
	  input_h = (input_h <= 0) ? 1 : input_h;
	}
      else if (arg_prefix == "-fs" && i < argc - 1)
	{
	  filter_sz = std::stoi(std::string(argv[++i]));
	  filter_sz = (filter_sz <= 0) ? 3 : filter_sz;
	}
      else if (arg_prefix == "-sd" && i < argc - 1)
	{
	  stride = std::stoi(std::string(argv[++i]));
	  stride = (stride <= 0) ? 1 : stride;
	}
      else if (arg_prefix == "-v")
	{
	  verify = true;
	}
      else if (arg_prefix == "-cnv")
	{          
	  conv_only = true;
	}
      else if (arg_prefix == "-fw")
	{
	  do_training = false;
	}
      else if (arg_prefix == "-h" && i < argc - 1)
	{
	  Usage();
	}
      else if (arg_prefix == "-l")
	{
	  per_layer_timing = true;
	}
      else if (arg_prefix == "-n")
	{
	  net_timing = false;
	}
      else
	{
	  printf("Unrecognized parameter: \"%s\".\n", arg_prefix.c_str());
	  Usage();
	}      
    }

  int status;
  max_iterations = (/*!net_timing || */verify) ? 1 : max_iterations;
  // TEMP	
  // max_iterations = 100;

  per_layer_iters = (verify) ? 1 : per_layer_iters;

  adnn_lib_parameters  lib_params;
  memset(&lib_params, 0, sizeof(adnn_lib_parameters));

  lib_params.accel_type = CL_DEVICE_TYPE_GPU;
  lib_params.ocl_kernels_path = "/c/AMD/MLopen/adnn/ocl";     // <<<<< make a command line arg, or part of a config file
  alib_obj aLib = ADNNLibCreate(&lib_params);

  printf("Created ADNN library {%s}\n", ADNNLibGetName(aLib));

  adnn_control_params layer_control;
  memset(&layer_control, 0, sizeof(adnn_control_params));
  layer_control.per_layer_iter = 1;
  layer_control.per_layer_messages = (net_timing == false);
  layer_control.per_layer_timing = per_layer_timing;
  layer_control.per_layer_iter = per_layer_iters;
  layer_control.debug_level = (verify) ? 1 : 0;
  
  adnn_filter1D_parameters f_params;
  f_params.size = filter_sz;
  f_params.pad = (filter_sz-1) / 2;
  f_params.stride = stride;

  adnn_update_params update_params;
  memset(&update_params, 0, sizeof(adnn_update_params));
  update_params.weights_lr.policy = ADNN_LP_LINEAR;
  update_params.weights_lr.base = 0.001;
  update_params.bias_lr.policy = ADNN_LP_FIXED;
  update_params.bias_lr.base = 0.001;
  update_params.weights_momentum = 0.9;
  update_params.bias_momentum = 0;
  update_params.weights_decay = 0.004;

  status = ADNNSingleConvLayer(aLib,
			       &layer_control,
			       &f_params,
			       batch_sz,
			       input_channels,
			       input_h,
			       input_w,
			       n_output_features,
			       do_training,
			       &update_params);

  /*  free(conv_filter_params);
  free(conv_featuremaps);
  free(neuron_params);
  free(pooling_filter_params);
  free(pooling_methods);
  free(LRN_params);
  */
  ADNNLibDestroy(&aLib);
}
// END
