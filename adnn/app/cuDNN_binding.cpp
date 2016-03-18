--------------------------------------------------------------------------------------------------------------------------------
// Line:2243
static int
ADNNcuDNNBinding(alib_obj aLib,
		 const adnn_control_params *layer_control,
		 int n_conv_filter_params,      const adnn_filter1D_parameters *conv_filter_params,
		 int *conv_featuremaps,
		 int n_neuron_params,           const adnn_neuron_parameters *neuron_params,
		 int n_pooling_filter_parames,	const adnn_filter1D_parameters *pooling_filter_params,
		 ADNN_POOLING_METHOD *pooling_method,
		 int n_lrn_parameters,   	const adnn_lrn_parameters *LRN_params,
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

  printf("****************************************************************************************************************\n\n");
  printf("ADNN : building a %s propagation pipepline with %d stand-alone layers with double buffering (cuDNN binding)\n",
  	       ((training) ? "forward/backward" : "forward"), n_nodes);
  printf("ADNN : please, wait...\n");
  printf("****************************************************************************************************************\n\n");

  int n_data_objs = 20;
  
  adata_obj *data_objs = (adata_obj *)malloc(sizeof(adata_obj) * n_data_objs);
  anode_obj *nodes = (anode_obj *)malloc(sizeof(anode_obj) * n_nodes);
  adnn_node_parameters *node_params = (adnn_node_parameters *)malloc(sizeof(adnn_node_parameters) * n_nodes);
  adata_obj *data_diffs = (adata_obj *)malloc(sizeof(adata_obj) * n_nodes);
  
  memset(data_objs, 0, sizeof(adata_obj) * n_data_objs);
  memset(data_diffs, 0, sizeof(adata_obj) * n_nodes);
  memset(nodes, 0, sizeof(anode_obj) * n_nodes);
  memset(node_params, 0, sizeof(adnn_node_parameters) * n_nodes);

  adata_obj data_src_objs[4] = { 0, 0, 0, 0 };
  adata_obj data_sink_objs[2] = { 0, 0 };

  adata_obj node_bot_df = 0;
  adata_obj node_top_df = 0;
  adata_obj node_weights_df = 0;
  adata_obj *node_bias_df = 0;

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
			   &data_objs[0],     // sink1
			   &data_objs[1],     // weights1
			   &data_objs[2],     // bias1
			   0,		      // conv1
			   0,		      // bot_df
			   0,		      // top_df
			   0,		      // weights_df
			   0,		      // bias_df
			   false,	      // training
			   true,	      // inference
			   &node_params[0]);  // conv1 params forward
		
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
			     ADNN_ED_SOURCE,
			     &data_objs[0],  // src2 == sink1
			     &data_objs[3],  // sink2
			     0,		     // neuron 1
			     0,		     // bot_df
			     0,		     // top_df
			     false,	     // training
			     true,	     // inference
			     &node_params[1]); // neuron 1 params forward

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
			      ADNN_ED_SOURCE,
			      &data_objs[3],  // src == prev sink
			      &data_objs[4],  // sink
			      0,	      // pooling 1
			      0,	      //bot_df
			      0,	      //top_df
			      false,	      // training
			      true,	      // inference
			      &node_params[2] // pooling 1 params forward
			      );

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
			 ADNN_ED_SOURCE,
			 &data_objs[4],	      // src == prev sink
			 &data_objs[5],	      // sink
			 0,		      // lnr 1
			 0,		      //bot_df
			 0,		      //top_df
			 false,		      // training
			 true,		      // inference
			 &node_params[3]      // lnr 1 params forward
			 );

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
			   ADNN_ED_SOURCE,
			   &data_objs[5],     // src == prev sink
			   &data_objs[6],     // sink
			   &data_objs[7],     // weights1
			   &data_objs[8],     // bias1
			   0,		      // conv2
			   0,		      //bot_df
			   0,		      //top_df
			   0,		      //weights_df
			   0,		      //bias_df
			   false,	      // training
			   true,	      // inference
			   &node_params[4]);  // conv2 params forward
	
  // neuron2S
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
		ADNN_ED_SOURCE,
		&data_objs[6], // src == prev sink
		&data_objs[9], // sink2
		0,        // neuron 2
		0,		//bot_df
		0,		//top_df
		false,  // training
		true,   // inference
		&node_params[5] // neuron 2 params forward
		);

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
		ADNN_ED_SOURCE,
		&data_objs[9],  // src == prev sink
		&data_objs[10], // sink
		0,        // pooling 2
		0,		//bot_df
		0,		//top_df
		false,  // training
		true,   // inference
		&node_params[6] // pooling 2 params forward
		);

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
		ADNN_ED_SOURCE,
		&data_objs[10], // src == prev sink
		&data_objs[11], // sink
		0,        // lnr 2
		0,		//bot_df
		0,		//top_df
		false,  // training
		true,   // inference
		&node_params[7] // lnr 2 params forward
		);

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
		ADNN_ED_SOURCE,
		&data_objs[11],  // src == prev sink
		&data_objs[12],  // sink
		&data_objs[13], // weights3
		&data_objs[14], // bias3
		0,     // conv3
		0,		//bot_df
		0,		//top_df
		0,		//weights_df
		0,		//bias_df
		false,  // training
		true,   // inference
		&node_params[8] // conv2 params forward
		);

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
		ADNN_ED_SOURCE,
		&data_objs[12], // src == prev sink
		&data_objs[15], // sink
		0,        // neuron 3
		0,		//bot_df
		0,		//top_df
		false,  // training
		true,   // inference
		&node_params[9] // neuron 3 params forward
		);


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
		ADNN_ED_SOURCE,
		&data_objs[15], // src == prev sink
		&data_objs[16], // sink
		0,        // lnr 3
		0,		//bot_df
		0,		//top_df
		false,  // training
		true,   // inference
		&node_params[10] // lnr 3 params forward
		);

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
		ADNN_ED_SOURCE,
		&data_objs[16],  // src == prev sink
		&data_objs[17],  // sink
		&data_objs[18],  // weights
		&data_objs[19],  // bias
		0,     // fully connected1
		0,		//bot_df
		0,		//top_df
		0,		//weights_df
		0,		//bias_df
		false,  // training
		true,   // inference
		&node_params[11] // fully connected params forward
		);


	// softmax with cross entropy cost
	// there is no sink data
	// the output is bottom difference
	// both inference and training are true
	status = PrepareSoftMaxNode(aLib,
		ADNN_NODE_SOFTMAX_COST_CROSSENTROPY,
		layer_control,
		0,
		n_categories,
		"soft_max",
		"fully_connect1",
		ADNN_ED_SOURCE,
		&data_objs[17],
		"labels",
		&data_src_objs[2],   // input labels
		0,		// external sink  - no sink. we are sending deltas as bottom differences up stream
		0,       // soft max
		&data_diffs[12],		//bot_df
		0,		//top_df
		true,  // training
		true,   // inference
		&node_params[12] //  softmax params forward
		);


	// labels double buffering

	data_src_objs[3] = ADNNDataClone(data_src_objs[2], true);


// at this point we connect all nodes of the stend-alone set by data running down-stream
// now we start connecting data differences up-stream

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
		ADNN_ED_SOURCE,
// data has been already created with down-stream
// we send src as a layout reference
        &data_objs[16],  // src == prev sink
		&data_objs[17],  // sink
		&data_objs[18],  // weights
		&data_objs[19],  // bias
		0,     // fully connected1
// new bottom data diff 
		&data_diffs[11],		//bot_df
// reference from down stream node
		&data_diffs[12],		//top_df
// weights and biad differences will eb created implicitly as slots
		0,		//weights_df
		0,		//bias_df
		true,  // training
		false,   // inference
		&node_params[11] // fully connected params forward
		);

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
		ADNN_ED_SOURCE,
		&data_objs[15], // src == prev sink
		&data_objs[16], // sink
		0,        // lnr 3
		// new bottom data diff 
		&data_diffs[10],		//bot_df
		// reference from down stream node
		&data_diffs[11],		//top_df
		true,  // training
		false,   // inference
		&node_params[10] // lnr 3 params forward
		);


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
		ADNN_ED_SOURCE,
		&data_objs[12], // src == prev sink
		&data_objs[15], // sink
		0,        // neuron 3
		// new bottom data diff 
		&data_diffs[9],		//bot_df
		// reference from down stream node
		&data_diffs[10],		//top_df
		true,  // training
		false,   // inference
		&node_params[9] // neuron 3 params forward
		);



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
		ADNN_ED_SOURCE,
		&data_objs[11],  // src == prev sink
		&data_objs[12],  // sink
		&data_objs[13], // weights3
		&data_objs[14], // bias3
		0,     // conv3
		// new bottom data diff 
		&data_diffs[8],		//bot_df
		// reference from down stream node
		&data_diffs[9],		//top_df
		// weights and bias differences will be created implicitly as slots
		0,		//weights_df
		0,		//bias_df
		true,  // training
		false,   // inference
		&node_params[8] // fully connected params forward
		);


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
		ADNN_ED_SOURCE,
		&data_objs[10], // src == prev sink
		&data_objs[11], // sink
		0,        // lnr 2
		// new bottom data diff 
		&data_diffs[7],		//bot_df
		// reference from down stream node
		&data_diffs[8],		//top_df
		true,  // training
		false,   // inference
		&node_params[7] // lnr 3 params forward
		);


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
		ADNN_ED_SOURCE,
		&data_objs[9],  // src == prev sink
		&data_objs[10], // sink
		0,        // pooling 2
		// new bottom data diff 
		&data_diffs[6],		//bot_df
		// reference from down stream node
		&data_diffs[7],		//top_df
		true,  // training
		false,   // inference
		&node_params[6] // lnr 3 params forward
		);

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
		ADNN_ED_SOURCE,
		&data_objs[6], // src == prev sink
		&data_objs[9], // sink2
		0,        // neuron 2
		// new bottom data diff 
		&data_diffs[5],		//bot_df
		// reference from down stream node
		&data_diffs[6],		//top_df
		true,  // training
		false,   // inference
		&node_params[5] // lnr 3 params forward
		);

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
		ADNN_ED_SOURCE,
		&data_objs[5],  // src == prev sink
		&data_objs[6],  // sink
		&data_objs[7], // weights2
		&data_objs[8], // bias2
		0,     // conv2
		// new bottom data diff 
		&data_diffs[4],		//bot_df
		// reference from down stream node
		&data_diffs[5],		//top_df
		// weights and bias differences will be created implicitly as node's internal slots
		0,		//weights_df
		0,		//bias_df
		true,  // training
		false,   // inference
		&node_params[4] // conv2 params forward
		);


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
		ADNN_ED_SOURCE,
		&data_objs[4], // src == prev sink
		&data_objs[5], // sink
		0,        // lnr 1
		// new bottom data diff 
		&data_diffs[3],		//bot_df
		// reference from down stream node
		&data_diffs[4],		//top_df
		true,  // training
		false,   // inference
		&node_params[3] // lnr 1 params forward
		);

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
		ADNN_ED_SOURCE,
		&data_objs[3],  // src == prev sink
		&data_objs[4],  // sink
		0,        // pooling 1
		// new bottom data diff 
		&data_diffs[2],		//bot_df
		// reference from down stream node
		&data_diffs[3],		//top_df
		true,  // training
		false,   // inference
		&node_params[2] // pooling 1 params forward
		);


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
		ADNN_ED_SOURCE,
		&data_objs[0], // src2 == sink1
		&data_objs[3],  // sink2
		0,        // neuron 1
		// new bottom data diff 
		&data_diffs[1],		//bot_df
		// reference from down stream node
		&data_diffs[2],		//top_df
		true,  // training
		false,   // inference
		&node_params[1] // neuron 1 params forward
		);

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
// references
		&data_src_objs[0], // external src
		&data_objs[0], // sink1
		&data_objs[1], // weights1
		&data_objs[2], // bias1
		0,      // conv1
		// new bottom data diff 
		&data_diffs[0],		//bot_df
		// reference from down stream node
		&data_diffs[1],		//top_df
		// weights and bias differences will be created implicitly as node's internal slots
		0,		//weights_df
		0,		//bias_df
		true,  // training
		false,   // inference
		&node_params[0] // conv1 params forward
		);


	// plan execution
	for (int i = 0; i < n_nodes; ++i)
	{
		nodes[i] = ADNNodeCreate(aLib, &node_params[i]);
		assert(nodes[i]);
	}

	// plan execution
	for (int i = 0; i < n_nodes; ++i)
	{
		if (training)
		{
			status |= ADNNodeConstructTraining(nodes[i]);
		}
		else
		{
			status |= ADNNodeConstruct(nodes[i]);
		}
	}

	// allocate data before building exectution path
	for (int i = 0; i < n_data_objs; ++i)
	{
		status |= ADNNDataAllocate(data_objs[i], 0);
	}

	for (int i = 0; i < 4; ++i)
	{
		status |= ADNNDataAllocate(data_src_objs[i], 0);
	}

	if (training)
	{
		for (int i = 0; i < n_nodes; ++i)
		{
			status |= ADNNDataAllocate(data_diffs[i], 0);
		}
	}
	else
	{
		for (int i = 0; i < 2; ++i)
		{
			status |= ADNNDataAllocate(data_sink_objs[i], 0);
		}
	}

	// buld execution path
	for (int i = 0; i < n_nodes; ++i)
	{
		if (training)
		{
			status |= ADNNodeBuildTraining(nodes[i]);
		}
		else
		{
			status |= ADNNodeBuild(nodes[i]);
		}
	}

printf("******************************************************************************************************************\n\n");
printf("ADNN : running %d iterations of a %s propagation pipepline with %d stand-alone layers with double buffering\
	       (cuDNN binding)\n", max_iterations, ((training) ? "forward/backward" : "forward"), n_nodes);
printf("ADNN : please, wait...\n");
printf("******************************************************************************************************************\n\n");

	adnn_data_init_parameters init_weights;
	adnn_data_init_parameters init_bias;
	memset(&init_weights, 0, sizeof(init_weights));
	memset(&init_bias, 0, sizeof(init_bias));

	init_weights.init_distr = ADNN_WD_GAUSSIAN;
	init_weights.std = 0.01;

	init_bias.init_distr = ADNN_WD_CONSTANT;
	init_bias.mean = 0.01;
// conv init
	// TEM SKIP bias for conv for now
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


	adnn_node_parameters  net_run_param[2];
	memset(net_run_param, 0, sizeof(adnn_node_parameters) * 2);

	adnn_node_parameters  *conv_src_param = &net_run_param[0];
	conv_src_param->name = "conv1";
	conv_src_param->n_input_nodes = 1;

	adnn_node_parameters  *conv_sink_param = &net_run_param[1];
	// second external input

	conv_sink_param->name = "soft_max";
	conv_sink_param->n_input_nodes = 2;
	conv_sink_param->inputs[1].data = data_src_objs[2];
	if (!training)
	{
		conv_sink_param->n_output_nodes = 1;
		conv_sink_param->outputs[0].data = data_sink_objs[0];
	}
	// run forward propagation
	// with dynamicall updated prameters
	// initial source upload

	adnn_data_parameters src_data_params;
	memset(&src_data_params, 0, sizeof(adnn_data_parameters));
	status = ADNNDataAccess(data_src_objs[0], 0, ADNN_MEM_ACCESS_WRITE_DESTRUCT, &src_data_params);
	// initialize with something

	for (size_t j = 0; j < src_data_params.size; ++j)
	{
		((float*)src_data_params.sys_mem)[j] = (float)((double)rand() * (1.0 / RAND_MAX));
	}

	status = ADNNDataCommit(data_src_objs[0]);

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

		// upload source
		adnn_data_parameters src_data_params;
		memset(&src_data_params, 0, sizeof(adnn_data_parameters));
		status = ADNNDataAccess(data_src_objs[(i + 1) % 2], 0, ADNN_MEM_ACCESS_WRITE_DESTRUCT, &src_data_params);
		// initialize with something

		for (size_t j = 0; j < src_data_params.size; ++j)
		{
			((float*)src_data_params.sys_mem)[j] = (float)((double)rand() * (1.0 / RAND_MAX));
		}
		status = ADNNDataCommit(data_src_objs[(i + 1) % 2]);

		// run forward propagation

		// send new input
		conv_src_param->inputs[0].data = data_src_objs[(i % 2)];
		// send new labels
		conv_sink_param->inputs[1].data = data_src_objs[2 + (i % 2)];
		// store result
		if (!training)
		{
			conv_sink_param->outputs[0].data = data_sink_objs[(i % 2)];
		}
		status |= ADNNodeRunInference(nodes[0], conv_src_param);

		for (int j = 1; j < n_nodes-1; ++j)
		{
			status |= ADNNodeRunInference(nodes[j], NULL);
		}

		status |= ADNNodeRunInference(nodes[n_nodes-1], conv_sink_param);

		if (training)
		  {		  
		    for (int j = n_nodes - 2; j >= 0; --j)
		      {
			status |= ADNNodeRunTraining(nodes[j], NULL);
		      }

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


	for (int i = 0; nodes && i < n_nodes; ++i)
	{
		if (data_diffs[i])
		{
			ADNNDataDestroy(&data_diffs[i]);
		}

		if (nodes[i])
		{
			ADNNodeDestroy(&nodes[i]);
		}
	}

	if (node_params)
	{
		free(node_params);
	}
	if (data_objs)
	{
		free(data_objs);
	}
	if (nodes)
	{
		free(nodes);
	}
	return(status);
}
// end
