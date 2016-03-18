// File:PrepareConvNode.cpp
// From:aDNNDriver.cpp line:338
// Path:aDNN/app

#include <cstdio>       // this defines malloc also, so commented out the malloc.h below
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <string>
#include <CL/opencl.h>
#include "../inc/AMDnn.h"
#include <vector>
#include <map>
#include <cassert>
#include "../inc/AMDnn.hpp"   // one of these may be gone

//static
int PrepareConvNode(alib_obj aLib, //         1    ?? 8 or 12 parameters can be passed by register
	const adnn_control_params *layer_control, // 2    the rest need to be put on stack (or ?? ) and thus incur cost
	const adnn_filter1D_parameters *f_params, // 3    IS there a way we can group these into a structure and pass
        int batch_sz,				  // 4    a pointer to the structure ?? 
	int input_channels,			  // 5
	int input_h,				  // 6
	int input_w,				  // 7
	int n_output_featuremaps,		  // 8
	const char * node_name,			  // 9
	const char *input_name,			  // 10
	ADNN_EDGE_DIR_TYPE input_edge_type,	  // 11
	adata_obj *node_src,			  // 12
	adata_obj *node_sink,			  // 13
	adata_obj *node_weights,		  // 14
	adata_obj *node_bias,			  // 15
	anode_obj *node,			  // 16
	adata_obj *node_bot_df,		  // 17
	adata_obj *node_top_df,		  // 18
	adata_obj *node_weights_df,	  // 19
	adata_obj *node_bias_df,		  // 20
	bool training,			  // 21
	bool inference,			  // 22
        adnn_node_parameters *pnode_params) // 23
{
  int status = 0;

  // convolution layer definition

  adnn_node_parameters node_params;
  if (!pnode_params)
    {
      memset(&node_params, 0, sizeof(adnn_node_parameters));
    }
  else
    {
      node_params = *pnode_params;
    }

  node_params.type = ADNN_NODE_CONV;
  
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
      node_params.inputs[0].data = *node_src;

      ADNNDataInspect(*node_src, &node_src_params);

      // filter setting
      node_params.inputs[0].filter_params.n_dims = 2;   // TT: why is this hard-coded 
      node_params.inputs[0].filter_params.filter[0] = node_params.inputs[0].filter_params.filter[1] = *f_params;

      // weights
      adnn_data_parameters node_weights_params;
      memset(&node_weights_params, 0, sizeof(adnn_data_parameters));
      node_weights_params.batch_format = ADNN_BF_HW;

      node_weights_params.dims[0] = n_output_featuremaps;
      node_weights_params.dims[1] = node_src_params.dims[1] * f_params->size * f_params->size; // + BIAS TO DO: SEPARATE Bias

      if (node_weights)
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

      node_bias_params.dims[0] = n_output_featuremaps;

      if (inference && node_bias)
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
	  if (!*node_bot_df)
	    {
	      *node_bot_df = ADNNDataClone(*node_src, true);
	    }
	  node_params.inputs[0].data_diff = *node_bot_df;
	}
      if (node_weights_df)
	{
	  if (!*node_weights_df)
	    {
	      *node_weights_df = ADNNDataClone(*node_weights, true);
	    }
	  node_params.inputs[0].weights_diff = *node_weights_df;
	}
      if (node_bias_df)
	{
	  if (!*node_bias_df)
	    {
	      *node_bias_df = ADNNDataClone(*node_bias, true);
	    }
	  node_params.inputs[0].bias_diff = *node_bias_df;
	}
    }

  // output
  node_params.n_output_nodes = (node_params.n_output_nodes == 0) ? 1 : node_params.n_output_nodes;
  if (node_name)
    {
      node_params.outputs[0].name = node_name;
    }

  // TO DO: CHECK LAYOUT. CURRENTLY assumes NCHW  <<<< assume this means N (batchnumber) Channel/ColorsComponent Height Width
  if (inference && node_src && *node_src && node_sink)
    {
      adnn_data_parameters node_sink_params;
      ADNNDataInspect(*node_src, &node_sink_params);
      // packed    <<<<---- TT: should we check stride first 
      for (int i = 0; i < 4; ++i)
	{
	  node_sink_params.strides[i] = 0;
	}

      int out_width = ((int)node_sink_params.dims[3] + 2 * f_params->pad - f_params->size) / f_params->stride + 1;
      int out_height = ((int)node_sink_params.dims[2] + 2 * f_params->pad - f_params->size) / f_params->stride + 1;
      
      node_sink_params.dims[1] = n_output_featuremaps;
      node_sink_params.dims[2] = out_height;
      node_sink_params.dims[3] = out_width;

      if (!*node_sink)
	{
	  *node_sink = ADNNDataCreate(aLib, &node_sink_params);
	}
      node_params.outputs[0].data = *node_sink;
    }

  if (training && node_sink && * node_sink && node_top_df)
    {
      if (!*node_top_df)
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
// END
