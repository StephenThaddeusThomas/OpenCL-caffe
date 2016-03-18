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
  lib_params.ocl_kernels_path = "../aLibDNN";     // <<<<< make a command line arg, or part of a config file
  alib_obj aLib = ADNNLibCreate(&lib_params);

  printf("Created ADNN library %s\n", ADNNLibGetName(aLib));

  adnn_control_params layer_control;
  memset(&layer_control, 0, sizeof(adnn_control_params));
  layer_control.per_layer_iter = 1;
  layer_control.per_layer_messages = (net_timing == false);
  layer_control.per_layer_timing = per_layer_timing;
  layer_control.per_layer_iter = per_layer_iters;
  layer_control.debug_level = (verify) ? 1 : 0;
  
  adnn_filter1D_parameters f_params;
  adnn_filter1D_parameters pooling_f_params;

  adnn_neuron_parameters neuron1_params;
  adnn_lrn_parameters lrn_params;

  f_params.size = filter_sz;
  f_params.pad = (filter_sz-1) / 2;
  f_params.stride = stride;

  neuron1_params.power = 0;
  neuron1_params.alpha = 1;
  neuron1_params.beta = 1;
  neuron1_params.type = ADNN_NEURON_RELU; // ADNN_NEURON_SOFTRELU; // ADNN_NEURON_TANH; // ADNN_NEURON_LOGISTIC;

  pooling_f_params.pad = 0;
  pooling_f_params.size = 3;
  pooling_f_params.stride = 2;

  ADNN_POOLING_METHOD pooling_method = ADNN_POOLING_MAX; //ADNN_POOLING_AVE; 

  lrn_params.region = ADNN_LRN_WITHIN_CHANNEL;; // ADNN_LRN_ACROSS_CHANNELS; // 
  lrn_params.kernel_sz = 3;
  lrn_params.alpha = 0.001;
  lrn_params.beta = 0.75;
  ADNN_NODE_TYPE softmax_type = (do_training) ? ADNN_NODE_SOFTMAX_COST_CROSSENTROPY : ADNN_NODE_SOFTMAX;

  adnn_update_params update_params;
  memset(&update_params, 0, sizeof(adnn_update_params));

  update_params.weights_lr.policy = ADNN_LP_LINEAR;
  update_params.weights_lr.base = 0.001;
  update_params.bias_lr.policy = ADNN_LP_FIXED;
  update_params.bias_lr.base = 0.001;
  update_params.weights_momentum = 0.9;
  update_params.bias_momentum = 0;
  update_params.weights_decay = 0.004;

#if 1
  // filter setting
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

  if (conv_only)
    {
      exit(0);
    }

  status = ADNNSingleNeuronLayer(aLib,
				 &layer_control,
				 &neuron1_params,
				 batch_sz,
				 input_channels,
				 input_h,
				 input_w,
				 do_training);

  status = ADNNSinglePoolingLayer(aLib,
				  &layer_control,
				  &pooling_f_params,
				  pooling_method,
				  batch_sz,
				  input_channels,
				  input_h,
				  input_w,
				  do_training);

  status = ADNNSingleLRNLayer(aLib,
			      ADNN_NODE_RESP_NORM,
			      &layer_control,
			      &lrn_params,
			      batch_sz,
			      input_channels,
			      input_h,
			      input_w,
			      do_training);
  
  status = ADNNFullConnectLayer(aLib,
				&layer_control,
				batch_sz,
				input_channels,
				input_h,
				input_w,
				n_categories,
				do_training);

  status = ADNNSoftMaxLayer(aLib,
			    &layer_control,
			    softmax_type,
			    batch_sz,
			    n_categories,
			    do_training);


  status = ADNNOpenVXBindingConvLayer(aLib,
				      batch_sz,
				      input_channels,
				      input_h,
				      input_w,
				      n_output_features);
#endif

#if 1
  // triple buffering
  status = ADNNSingleConvLayer3plBuffering(aLib,
					   &layer_control,
					   &f_params,
					   3, //max_iterations,
					   batch_sz,
					   input_channels,
					   input_h,
					   input_w,
					   n_output_features);
//	exit(0);
#endif

  int n_conv_filter_params = 3;
  adnn_filter1D_parameters *conv_filter_params = (adnn_filter1D_parameters *)malloc(sizeof(adnn_filter1D_parameters) *n_conv_filter_params);
  int *conv_featuremaps = (int*)malloc(sizeof(int) * n_conv_filter_params);
  int n_neuron_params = 3;
  adnn_neuron_parameters *neuron_params = (adnn_neuron_parameters *)malloc(sizeof(adnn_neuron_parameters) * n_neuron_params);
  int n_pooling_filter_parames = 2;
  adnn_filter1D_parameters *pooling_filter_params = (adnn_filter1D_parameters *)malloc(sizeof(adnn_filter1D_parameters) * n_pooling_filter_parames);
  ADNN_POOLING_METHOD *pooling_methods = (ADNN_POOLING_METHOD *)malloc(sizeof(ADNN_POOLING_METHOD) * n_pooling_filter_parames);

  for (int i = 0; i < n_conv_filter_params; ++i)
    {
      conv_filter_params[i] = f_params;
    }
  conv_featuremaps[0] = 32;
  conv_featuremaps[1] = 32;
  conv_featuremaps[2] = 64;

  for (int i = 0; i < n_neuron_params; ++i)
    {
      neuron_params[i] = neuron1_params;
    }
  
  neuron_params[1].type = ADNN_NEURON_TANH;
  neuron_params[2].type = ADNN_NEURON_LOGISTIC;
  
  for (int i = 0; i < n_pooling_filter_parames; ++i)
    {
      pooling_filter_params[i] = pooling_f_params;
    }
  
  pooling_methods[0] = ADNN_POOLING_MAX;
  pooling_methods[1] = ADNN_POOLING_AVE;

  int n_lrn_parameters = 3;
  adnn_lrn_parameters *LRN_params = (adnn_lrn_parameters*)malloc(sizeof(adnn_lrn_parameters) * n_lrn_parameters);
  for (int i = 0; i < n_lrn_parameters; ++i)
    {
      LRN_params[i] = lrn_params;
    }

#if 1
  status = ADNNcuDNNBinding(aLib,
			    &layer_control,
			    n_conv_filter_params,
			    conv_filter_params,
			    conv_featuremaps,
			    n_neuron_params,
			    neuron_params,
			    n_pooling_filter_parames,
			    pooling_filter_params,
			    pooling_methods,
			    n_lrn_parameters,
			    LRN_params,
			    max_iterations,
			    batch_sz,
			    input_channels,
			    input_h,
			    input_w,
			    n_categories,
			    do_training	);
//	exit(0);
#endif

  adnn_net_parameters net_params;
  memset(&net_params, 0, sizeof(adnn_net_parameters));

  adnn_control_params net_control;
  memset(&net_control, 0, sizeof(adnn_control_params));

  net_control.per_layer_timing = net_timing;
  net_control.per_layer_messages = net_timing;

  // net is a node
  net_params.name = "CIFAR10";

  net_params.control = net_control;

  net_params.update_params = update_params;

  ADNNCAFFEEBinding(aLib,
		    &net_params,
		    &layer_control,
		    n_conv_filter_params,
		    conv_filter_params,
		    conv_featuremaps,
		    n_neuron_params,
		    neuron_params,
		    n_pooling_filter_parames,
		    pooling_filter_params,
		    pooling_methods,
		    n_lrn_parameters,
		    LRN_params,
		    max_iterations,
		    batch_sz,
		    input_channels,
		    input_h,
		    input_w,
		    n_categories,
		    do_training	);

  free(conv_filter_params);
  free(conv_featuremaps);
  free(neuron_params);
  free(pooling_filter_params);
  free(pooling_methods);
  free(LRN_params);
  
  ADNNLibDestroy(&aLib);
}
// END
