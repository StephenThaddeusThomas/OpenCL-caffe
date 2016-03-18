// called by main() 
// from 3175
static int ADNNCAFFEEBinding(alib_obj aLib,
			     const adnn_net_parameters *net_params,
			     const adnn_control_params *layer_control,
			     int n_conv_filter_params,
			     const adnn_filter1D_parameters *conv_filter_params,
			     int *conv_featuremaps,
			     int n_neuron_params,
			     const adnn_neuron_parameters *neuron_params,
			     int n_pooling_filter_parames,
			     const adnn_filter1D_parameters *pooling_filter_params,
			     ADNN_POOLING_METHOD *pooling_method,
			     int n_lrn_parameters,
			     const adnn_lrn_parameters *LRN_params,
			     int max_iterations,
			     int batch_sz,
			     int input_channels,
			     int input_h,
			     int input_w,
			     int n_categories,
			     bool training = false)
{
  int status = 0;
  int n_nodes = 13;

  printf("*********************************************************************************************************************************\n\n");
  printf("ADNN : building a %s propagation pipepline  with %s net consisting of %d layers with double buffering (CAFFE binding)\n", ((training) ? "forward/backward" : "forward"), net_params->name, n_nodes);
  printf("ADNN : please, wait...\n");
  printf("**********************************************************************************************************************************\n\n");

  int n_data_objs = 20;

  adata_obj *data_objs = (adata_obj *)malloc(sizeof(adata_obj) * n_data_objs);
  anode_obj *nodes = (anode_obj *)malloc(sizeof(anode_obj) * n_nodes);
  memset(data_objs, 0, sizeof(adata_obj) * n_data_objs);
  memset(nodes, 0, sizeof(anode_obj) * n_nodes);

  adata_obj data_src_objs[4] = { 0, 0, 0, 0 };
  adata_obj data_sink_objs[2] = { 0, 0 };

  // conv 1
  status = PrepareConvNode(aLib,
			   layer_control,
			   &conv_filter_params[0],
			   batch_sz,
			   input_channels,
			   input_h,
			   input_w,
			   conv_featuremaps[0],
			   "conv1",
			   "conv1_src",
			   ADNN_ED_SOURCE,
			   &data_src_objs[0], // external src
			   &data_objs[0], // sink1
			   &data_objs[1], // weights1
			   &data_objs[2], // bias1
			   &nodes[0]);      // conv1

  // source double buffering

  data_src_objs[1] = ADNNDataClone(data_src_objs[0], true);

  // neuron1

  status = PrepareNeuronNode(aLib,
			     layer_control,
			     &neuron_params[0],
			     // input is a prev output
			     0,
			     0,
			     0,
			     0,
			     "neuron1",
			     "conv1",
			     ADNN_ED_INTERNAL,
			     &data_objs[0], // src2 == sink1
			     &data_objs[3],  // sink2
			     &nodes[1]);        // neuron 1
			     
  // pooling 1
  status = PreparePoolingNode(aLib,
			      ADNN_NODE_POOLING,
			      pooling_method[0],
			      &pooling_filter_params[0],
			      layer_control,
			      0,
			      0,
			      0,
			      0,
			      "pooling1",
			      "neuron1",
			      ADNN_ED_INTERNAL,
			      &data_objs[3],  // src == prev sink
			      &data_objs[4],  // sink
			      &nodes[2]);       // pooling1
  // lrn 1
  status = PrepareLRNode(aLib,
			 ADNN_NODE_RESP_NORM,
			 layer_control,
			 &LRN_params[0],
			 0,
			 0,
			 0,
			 0,
			 "lrn1",
			 "pooling1",
			 ADNN_ED_INTERNAL,
			 &data_objs[4], // src == prev sink
			 &data_objs[5], // sink
			 &nodes[3]);      // lrn 1
  // conv 2
  status = PrepareConvNode(aLib,
			   layer_control,
			   &conv_filter_params[1],
			   0,
			   0,
			   0,
			   0,
			   conv_featuremaps[1],
			   "conv2",
			   "lrn1",
			   ADNN_ED_INTERNAL,
			   &data_objs[5],  // src == prev sink
			   &data_objs[6],  // sink
			   &data_objs[7], // weights1
			   &data_objs[8], // bias1
			   &nodes[4]);      // conv2
  // neuron2
  status = PrepareNeuronNode(aLib,
			     layer_control,
			     &neuron_params[1],
			     // input is a prev output
			     0,
			     0,
			     0,
			     0,
			     "neuron2",
			     "conv2",
			     ADNN_ED_INTERNAL,
			     &data_objs[6], // src == prev sink
			     &data_objs[9], // sink2
			     &nodes[5]);      // neuron 2
  // poooling2
  status = PreparePoolingNode(aLib,
			      ADNN_NODE_POOLING,
			      pooling_method[1],
			      &pooling_filter_params[1],
			      layer_control,
			      0,
			      0,
			      0,
			      0,
			      "pooling2",
			      "neuron2",
			      ADNN_ED_INTERNAL,
			      &data_objs[9],  // src == prev sink
			      &data_objs[10], // sink
			      &nodes[6]);       // poolin1
  // lrn 2
  status = PrepareLRNode(aLib,
			 ADNN_NODE_RESP_NORM,
			 layer_control,
			 &LRN_params[1],
			 0,
			 0,
			 0,
			 0,
			 "lrn2",
			 "pooling2",
			 ADNN_ED_INTERNAL,
			 &data_objs[10], // src == prev sink
			 &data_objs[11], // sink
			 &nodes[7]);      // lrn 2
  // conv 3
  status = PrepareConvNode(aLib,
			   layer_control,
			   &conv_filter_params[2],
			   0,
			   0,
			   0,
			   0,
			   conv_featuremaps[2],
			   "conv3",
			   "lrn2",
			   ADNN_ED_INTERNAL,
			   &data_objs[11],  // src == prev sink
			   &data_objs[12],  // sink
			   &data_objs[13], // weights1
			   &data_objs[14], // bias1
			   &nodes[8]);      // conv2
  // neuron3
  status = PrepareNeuronNode(aLib,
			     layer_control,
			     &neuron_params[2],
			     // input is a prev output
			     0,
			     0,
			     0,
			     0,
			     "neuron3",
			     "conv3",
			     ADNN_ED_INTERNAL,
			     &data_objs[12], // src == prev sink
			     &data_objs[15], // sink
			     &nodes[9]);        // neuron 2
  // lrn 3
  status = PrepareLRNode(aLib,
			 ADNN_NODE_RESP_NORM,
			 layer_control,
			 &LRN_params[2],
			 0,
			 0,
			 0,
			 0,
			 "lrn3",
			 "neuron3",
			 ADNN_ED_INTERNAL,
			 &data_objs[15], // src == prev sink
			 &data_objs[16], // sink
			 &nodes[10]);      // lrn 3
  // fully connected
  status = PrepareFullyConnectNode(aLib,
				   layer_control,
				   0,
				   0,
				   0,
				   0,
				   n_categories,
				   "fully_connect1",
				   "lrn3",
				   ADNN_ED_INTERNAL,
				   &data_objs[16],  // src == prev sink
				   &data_objs[17],  // sink
				   &data_objs[18],  // weights
				   &data_objs[19],  // bias
				   &nodes[11]);      // fully connected1

  // softmax with cross entropy cost
  status = PrepareSoftMaxNode(aLib,
			      ADNN_NODE_SOFTMAX_COST_CROSSENTROPY,
			      layer_control,
			      0,
			      n_categories,
			      "soft_max",
			      "fully_connect1",
			      ADNN_ED_INTERNAL,
			      &data_objs[17],
			      "labels",
			      &data_src_objs[2],   // input labels
			      (training) ? NULL : &data_sink_objs[0],  // external sink
			      &nodes[12]      // soft max
			      );

  // labels double buffering
  
  data_src_objs[3] = ADNNDataClone(data_src_objs[2], true);

  // sink double buffering
  if (!training)
    {
      data_sink_objs[1] = ADNNDataClone(data_sink_objs[0], true);
    }

  const adnn_net_parameters * cifar_net_params = net_params;

  anet_obj cifar_net = ADNNCreate(aLib, cifar_net_params);

  status = ADNNodesAdd(cifar_net, n_nodes, nodes);

  // connect node and verify net
  status = ADNNConnect(cifar_net);

  // plan execution
  if (training)
    {
      status = ADNNConstructTraining(cifar_net);
    }
  else
    {
      status = ADNNConstruct(cifar_net);
    }

  // allocate data before building exectution path

  for (int i = 0; i < n_data_objs; ++i)
    {
      status |= ADNNDataAllocate(data_objs[i], 0);     // ?? do these return bit-othogonal return types 
    }

  for (int i = 0; i < 4; ++i)
    {
      status |= ADNNDataAllocate(data_src_objs[i], 0);
    }

  for (int i = 0; !training && i < 2; ++i)
    {
      status |= ADNNDataAllocate(data_sink_objs[i], 0);
    }

  // buld execution path
  if (training)
    {
      status = ADNNBuildTraining(cifar_net);
    }
  else
    {
      status = ADNNBuild(cifar_net);
    }

  printf("*********************************************************************************************************************************\n\n");
  printf("ADNN : running %d iterations of a %s propagation pipepline  with %s net consisting of %d layers with double buffering (CAFFE binding)\n", max_iterations, ((training) ? "forward/backward" : "forward"), net_params->name, n_nodes);
  printf("ADNN : please, wait...\n");
  printf("**********************************************************************************************************************************\n\n");

  adnn_data_init_parameters init_weights;
  adnn_data_init_parameters init_bias;
  memset(&init_weights, 0, sizeof(adnn_data_init_parameters));
  memset(&init_bias, 0, sizeof(adnn_data_init_parameters));

  init_weights.init_distr = ADNN_WD_GAUSSIAN;
  init_weights.std = 0.01;

  init_bias.init_distr = ADNN_WD_CONSTANT;
  init_bias.mean = 1;

  // TEM SKIP bias
  // initilize (or upload) weights
  status |= ADNNDataInit(data_objs[1], &init_weights);
  // initilize (or upload) weights
  status |= ADNNDataInit(data_objs[6], &init_weights);
  // initilize (or upload) weights
  status |= ADNNDataInit(data_objs[11], &init_weights);

  // fully con init
  // initilize (or upload) weights
  status |= ADNNDataInit(data_objs[18], &init_weights);
  // initilize (or upload) bias
  status |= ADNNDataInit(data_objs[19], &init_bias);

  // initialize labels once (TEMP)
  adnn_data_init_parameters init_labels;
  memset(&init_labels, 0, sizeof(adnn_data_init_parameters));
  init_labels.init_distr = ADNN_WD_CATEGORIES;

  // initilize (or upload) labels for testing
  status = ADNNDataInit(data_src_objs[2], &init_labels);
  status = ADNNDataInit(data_src_objs[3], &init_labels);

  // dynamic run-time parameters 
  adnn_node_parameters  net_run_param[2];
  memset(net_run_param, 0, sizeof(adnn_node_parameters) * 2);

  // net source 
  adnn_node_parameters  *conv_src_param = &net_run_param[0];
  conv_src_param->name = "conv1";
  conv_src_param->n_input_nodes = 1;
  conv_src_param->inputs[0].data = data_src_objs[0];

  // net sink
  adnn_node_parameters  *conv_sink_param = &net_run_param[1];
  conv_sink_param->name = "soft_max";
  // net second external input
  conv_sink_param->n_input_nodes = 2;
  conv_sink_param->inputs[1].data = data_src_objs[2];
  if (!training)
    {
      conv_sink_param->n_output_nodes = 1;
      conv_sink_param->outputs[0].data = data_sink_objs[0];
    }

  adnn_data_parameters src_data_params;
  memset(&src_data_params, 0, sizeof(adnn_data_parameters));
  status = ADNNDataAccess(data_src_objs[0], 0, ADNN_MEM_ACCESS_WRITE_DESTRUCT, &src_data_params);

  // initialize with something
  for (size_t j = 0; j < src_data_params.size; ++j)
    {
      ((float*)src_data_params.sys_mem)[j] = (float)((double)rand() * (1.0 / RAND_MAX));
    }

  status = ADNNDataCommit(data_src_objs[0]);

  memset(&src_data_params, 0, sizeof(adnn_data_parameters));
  status = ADNNDataAccess(data_src_objs[1], 0, ADNN_MEM_ACCESS_WRITE_DESTRUCT, &src_data_params);
  // initialize with something

  for (size_t j = 0; j < src_data_params.size; ++j)
    {
      ((float*)src_data_params.sys_mem)[j] = (float)((double)rand() * (1.0 / RAND_MAX));   // reusing data_param 
    }

  status = ADNNDataCommit(data_src_objs[1]);

  // run net forward propogation with double buffering input and output
  for (int i = 0; i < max_iterations; ++i)
    {
      // download result
      if (!training && i > 0)
	{
	  adnn_data_parameters out_data_params;
	  memset(&out_data_params, 0, sizeof(adnn_data_parameters));
	  status = ADNNDataAccess(data_sink_objs[(i - 1) % 2], 0, ADNN_MEM_ACCESS_READ, &out_data_params);

	  // move the data out here
	  // .... = ((float*)out_data_params.sys_mem)[i] 

	  status = ADNNDataCommit(data_sink_objs[(i - 1) % 2]);
	}

      // run forward propagation
      // send new input
      conv_src_param->inputs[0].data = data_src_objs[(i % 2)];
      // send new labels
      conv_sink_param->inputs[1].data = data_src_objs[2 + (i % 2)];

      if (!training)
	{
	  // store result
	  conv_sink_param->outputs[0].data = data_sink_objs[(i % 2)];
	}

      if (training)
	{
	  status = ADNNRunTraining(cifar_net, 2, net_run_param);
	}
      else
	{
	  status = ADNNRunInference(cifar_net, 2, net_run_param);
	}

      printf("*********************************************************************************************************************************\n\n");
      printf("ADNN : run %d iteration\n", i+1);
      printf("**********************************************************************************************************************************\n\n");
    }

  // clean up
  for (int i = 0; i < n_data_objs; ++i)
    {
      if (data_objs[i])
	{
	  ADNNDataDestroy(&data_objs[i]);
	}
    }

  for (int i = 0; i < 4; ++i)
    {
      if (data_src_objs[i])
	{
	  ADNNDataDestroy(&data_src_objs[i]);
	}
    }

  for (int i = 0; i < 2; ++i)
    {
      if (data_sink_objs[i])
	{
	  ADNNDataDestroy(&data_sink_objs[i]);
	}
    }

  for (int i = 0; i < n_nodes; ++i)
    {
      if (nodes[i])
	{
	  ADNNodeDestroy(&nodes[i]);
	}
    }

  free(data_objs);
  free(nodes);
  
  status = ADNNDestroy(&cifar_net);
  
  return(status);
}
