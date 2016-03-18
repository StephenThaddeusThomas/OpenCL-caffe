//File:PrepareFullConnectNode.cpp
//From:aDNNDriver.cpp line:667

static int PrepareFullyConnectNode(alib_obj aLib,
	const adnn_control_params *layer_control,
	int batch_sz,
	int input_channels,
	int input_h,
	int input_w,
	int n_categories,
	const char * node_name,
	const char *input_name,
	ADNN_EDGE_DIR_TYPE input_edge_type,
	adata_obj *node_src,
	adata_obj *node_sink,
	adata_obj *node_weights,
	adata_obj *node_bias,
	anode_obj *node,
	adata_obj *node_bot_df = NULL,
	adata_obj *node_top_df = NULL,
	adata_obj *node_weights_df = NULL,
	adata_obj *node_bias_df = NULL,
	bool training = false,
	bool inference = true,
	adnn_node_parameters *pnode_params = NULL )
{
  int status = 0;

  // convolution layer definition

  adnn_node_parameters node_params;
  if (pnode_params)
    {
      node_params = *pnode_params;
    }
  else
    {
      memset(&node_params, 0, sizeof(adnn_node_parameters));
    }

  node_params.type = ADNN_NODE_FULLY_CONNECT;
	
  if (node_name)
    {
      node_params.name = node_name;
    }

  if (layer_control)
    {
      node_params.control = *layer_control;
    }

  // input
  node_params.n_input_nodes = (node_params.n_input_nodes == 0) ? 1 : node_params.n_input_nodes;

  if (input_name)
    {
      node_params.inputs[0].name = input_name;
    }

  node_params.inputs[0].edge_type = input_edge_type;

  // src
  adnn_data_parameters node_src_params;
  memset(&node_src_params, 0, sizeof(adnn_data_parameters));
  node_src_params.data_format = ADNN_DF_FP32;
  node_src_params.batch_format = ADNN_BF_NCHW;
  // strides set to 0 means it's a packed tensor

  node_src_params.dims[0] = batch_sz;
  node_src_params.dims[1] = input_channels;
  node_src_params.dims[2] = input_h;
  node_src_params.dims[3] = input_w;
  
  if (inference && node_src)
    {
      if (!*node_src)
	{
	  *node_src = ADNNDataCreate(aLib, &node_src_params);
	}
      node_params.inputs[0].data = *node_src;

      ADNNDataInspect(*node_src, &node_src_params);

      batch_sz = (batch_sz == 0) ? (int)node_src_params.dims[0] : batch_sz;

      // weights
      adnn_data_parameters node_weights_params;
      memset(&node_weights_params, 0, sizeof(adnn_data_parameters));
      node_weights_params.batch_format = ADNN_BF_HW;
      node_weights_params.dims[0] = n_categories;
      node_weights_params.dims[1] = node_src_params.dims[1] * node_src_params.dims[2] * node_src_params.dims[3];

      if (node_weights && !*node_weights)
	{
	  if (!*node_weights)
	    {
	      *node_weights = ADNNDataCreate(aLib, &node_weights_params);
	    }
	  node_params.inputs[0].weights = *node_weights;
	}

      // bias
      adnn_data_parameters node_bias_params;
      memset(&node_bias_params, 0, sizeof(adnn_data_parameters));
      node_bias_params.batch_format = ADNN_BF_W;
      node_bias_params.dims[0] = n_categories;

      if (node_bias)
	{
	  if (!*node_bias)
	    {
	      *node_bias = ADNNDataCreate(aLib, &node_bias_params);
	    }
	  node_params.inputs[0].bias = *node_bias;
	}
    }

  if (training)
    {
      if (node_bot_df)
	{
	  if (node_src && *node_src && !*node_bot_df)
	    {
	      *node_bot_df = ADNNDataClone(*node_src, true);
	    }
	  node_params.inputs[0].data_diff = *node_bot_df;
	}
      if (node_weights_df)
	{
	  if (node_weights && *node_weights && !*node_weights_df)
	    {
	      *node_weights_df = ADNNDataClone(*node_weights, true);
	    }
	  node_params.inputs[0].weights_diff = *node_weights_df;
	}
      if (node_bias_df)
	{
	  if (node_bias && *node_bias && !*node_bias_df)
	    {
	      *node_bias_df = ADNNDataClone(*node_bias, true);
	    }
	  node_params.inputs[0].bias_diff = *node_bias_df;
	}
    }

  node_params.n_output_nodes = (node_params.n_output_nodes == 0) ? 1 : node_params.n_output_nodes;

  if (node_name)
    {
      node_params.outputs[0].name = node_name;
    }

  // output
  // TODO: CHECK LAYOUT. CURRENTLY assumes NCHW
  adnn_data_parameters node_sink_params;

  memset(&node_sink_params, 0, sizeof(adnn_data_parameters));
  node_sink_params.data_format = ADNN_DF_FP32;
  node_sink_params.batch_format = ADNN_BF_NW;
  node_sink_params.dims[0] = batch_sz;
  node_sink_params.dims[1] = n_categories;

  if (inference && node_sink)
    {
      if (!*node_sink)
	{
	  *node_sink = ADNNDataCreate(aLib, &node_sink_params);
	}
      node_params.outputs[0].data = *node_sink;
    }
  if (training && node_top_df)
    {
      if (node_sink && *node_sink && !*node_top_df)
	{
	  *node_top_df = ADNNDataClone(*node_sink, true);
	}
      node_params.outputs[0].data_diff = *node_top_df;
    }

  // parameters are all fully copied
  // they can go out off scope
  if (pnode_params)
    {
      *pnode_params = node_params;
    }

  if (node)
    {
	*node = ADNNodeCreate(aLib, &node_params);
    }
  return(status);
}
//END
