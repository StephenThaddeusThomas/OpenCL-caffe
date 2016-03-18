// File:PrepareNeuronNode.cpp
// From:aDNNDriver.cpp line: 540

static int
PrepareNeuronNode(alib_obj aLib,
	const adnn_control_params *node_control,
	const adnn_neuron_parameters *neuron_params,
	int batch_sz,
	int input_channels,
	int input_h,
	int input_w,
	const char *node_name,
	const char *input_name,
	ADNN_EDGE_DIR_TYPE input_edge_type,
	adata_obj *node_src,
	adata_obj *node_sink,
	anode_obj *node,
	adata_obj *node_bot_df = NULL,
	adata_obj *node_top_df = NULL,
	bool training = false,
	bool inference = true,
	adnn_node_parameters *pnode_params = NULL )
{
  int status = 0;
  adata_obj data_objs[2] = { 0, 0 };
  adnn_node_parameters  neuron_param;

  if (pnode_params)
    {
      neuron_param = *pnode_params;
    }
  else
    {
      memset(&neuron_param, 0, sizeof(neuron_param));
    }
 
  // neuron node definition
  neuron_param.type = ADNN_NODE_NEURON;
  if (node_name)
    {
      neuron_param.name = node_name;
    }
  if (node_control)
    {
      neuron_param.control = *node_control;
    }
  // neuron specific parameters
  if (neuron_params)
    {
      neuron_param.neuron_params = *neuron_params;
    }
  // input
  neuron_param.n_input_nodes = (neuron_param.n_input_nodes == 0) ? 1 : neuron_param.n_input_nodes;
  if (input_name)
    {
      neuron_param.inputs[0].name = input_name;
    }
  neuron_param.inputs[0].edge_type = input_edge_type;

  adnn_data_parameters node_src_params;
  memset(&node_src_params, 0, sizeof(adnn_data_parameters));
  node_src_params.data_format = ADNN_DF_FP32;
  node_src_params.batch_format = ADNN_BF_NCHW;
  // strides set to 0 means it's a packed tensor

  node_src_params.dims[0] = batch_sz;
  node_src_params.dims[1] = input_channels;
  node_src_params.dims[2] = input_h;
  node_src_params.dims[3] = input_w;

  // src
  if (inference && node_src)
    {
      if (!*node_src)
	{
	  *node_src = ADNNDataCreate(aLib, &node_src_params);
	}
      neuron_param.inputs[0].data = *node_src;
    }

  if (training && node_bot_df)
    {
      if (node_src && *node_src && !*node_bot_df)
	{
	  *node_bot_df = ADNNDataClone(*node_src, true);
	}
      neuron_param.inputs[0].data_diff = *node_bot_df;
    }

  // output
  neuron_param.n_output_nodes = (neuron_param.n_output_nodes == 0)? 1 : neuron_param.n_output_nodes;
  neuron_param.outputs[0].name = node_name;

  // TODO: CHECK LAYOUT. CURRENTLY assumes NCHW
  if (inference && node_src && *node_src && node_sink)
    {
      if (!*node_sink)
	{
	  *node_sink = ADNNDataClone(*node_src, true);
	}
      neuron_param.outputs[0].data = *node_sink;
    }
  if (training && node_sink && *node_sink && node_top_df)
    {
      if (!*node_top_df)
	{
	  *node_top_df = ADNNDataClone(*node_sink, true);
	}
      neuron_param.outputs[0].data_diff = *node_top_df;
    }

  // parameters are all fully copied
  // they can go out off scope
  if (pnode_params)
    {
      *pnode_params = neuron_param;
    }

  if (node)
    {
      *node = ADNNodeCreate(aLib, &neuron_param);
    }
  
  return (status);
}
// END
