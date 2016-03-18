// File: RunSingleConvLayerTrain.cpp
// This has BOTH forward and back - compare to RunSingleConvLayerInfer.cpp which is only forward
// from aLibDNNDriver.cpp line 1909

#include <cstdio>       // this defines malloc also, so commented out the malloc.h below
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <string>
#include <CL/opencl.h>
#include "../src/AMDnn.h"
#include "../src/AMDnn.hpp"   // one of these may be gone

static int
run_single_convLayer_training(alib_obj aLib,
		    const adnn_control_params *layer_control,
		    const adnn_filter1D_parameters *filter_params,
		    int batch_sz,
		    int input_channels,
		    int input_h,
		    int input_w,
		    int n_output_featuremaps,
		    adnn_update_params *pupdate_params)
{
  int status = 0;

  printf("***************************************************************************************************\n\n");
  printf("TEST : build and run a forward/backward propagation pipeline with a single stand-alone convolutional layer\n");
  printf("****************************************************************************************************\n");

  adata_obj node_src = 0, node_sink = 0, node_weights = 0, node_bias = 0;
  adata_obj node_bot_df = 0, node_top_df = 0, node_weights_df = 0, node_bias_df = 0;
  anode_obj node;
  adnn_node_parameters node_params;

  memset(&node_params, 0, sizeof(adnn_node_parameters));

  status = PrepareConvNode(aLib,
			   layer_control,
			   filter_params,
			   batch_sz,
			   input_channels,
			   input_h,
			   input_w,
			   n_output_featuremaps,
			   "conv_node",
			   "conv_src",
			   ADNN_ED_SOURCE,
			   &node_src,
			   &node_sink,
			   &node_weights,
			   &node_bias,
			   (training) ? 0 : &node,  // conv1
			   0,			    // bot_df
			   0,			    // top_df
			   0,			    // weights_df
			   0,			    // bias_df
			   false,		    // training
			   true,		    // inference
			   (training) ? &node_params : 0 );

  if (pupdate_params)
    {
      node_params.update_params = *pupdate_params;
    }

  status = PrepareConvNode(aLib,
			   layer_control,
			   filter_params,
			   batch_sz,
			   input_channels,
			   input_h,
			   input_w,
			   n_output_featuremaps,
			   "conv_node",
			   "conv_src",
			   ADNN_ED_SOURCE,
			   &node_src,
			   &node_sink,
			   &node_weights,
			   &node_bias,
			   &node,
			   &node_bot_df,
			   &node_top_df,
			   &node_weights_df,
			   &node_bias_df,
			   true,     // training
			   false,    // inference
			   &node_params);
  
  // conctruct an execution plan
  status = ADNNodeConstructTraining(node);
  
  // allocate data before the node build
  status = ADNNDataAllocate(node_src, 0);
  status = ADNNDataAllocate(node_weights, 0);
  status = ADNNDataAllocate(node_bias, 0);
  status = ADNNDataAllocate(node_sink, 0);

  // buld execution path
  status = ADNNDataAllocate(node_bot_df, 0);
  status = ADNNDataAllocate(node_weights_df, 0);
  status = ADNNDataAllocate(node_bias_df, 0);
  status = ADNNDataAllocate(node_top_df, 0);
  
  status = ADNNodeBuildTraining(node);
  
  // initialization operator
  adnn_data_init_parameters init_weights;
  adnn_data_init_parameters init_bias;
  memset(&init_weights, 0, sizeof(init_weights));
  memset(&init_bias, 0, sizeof(init_bias));

  init_weights.init_distr = ADNN_WD_GAUSSIAN;
  init_weights.std = 0.01;

  init_bias.init_distr = ADNN_WD_CONSTANT;
  init_bias.mean = 0.01;

  // initilize (or upload) weights
  status = ADNNDataInit(node_weights, &init_weights);
  // initilize (or upload) bias
  status = ADNNDataInit(node_bias, &init_bias);

  // initialize top_df
  adnn_data_init_parameters init_top_df;
  memset(&init_top_df, 0, sizeof(adnn_data_init_parameters));
  init_top_df.init_distr = ADNN_WD_GAUSSIAN;
  init_top_df.std = 0.01;
  status = ADNNDataInit(node_top_df, &init_top_df);
  
  // upload source
  adnn_data_parameters data_params;
  memset(&data_params, 0, sizeof(adnn_data_parameters));
  status = ADNNDataAccess(node_src, 0, ADNN_MEM_ACCESS_WRITE_DESTRUCT, &data_params);

  // initialize with something
  for (size_t i = 0; i < data_params.size; ++i)
    {
      ((float*)data_params.sys_mem)[i] = (float)((double)rand() * (1.0 / RAND_MAX));
    }

  status = ADNNDataCommit(node_src);

  // run forward propagation
  status = ADNNodeRunInference(node, NULL);
  // run backward propagation
  status = ADNNodeRunTraining(node, NULL);

  // download output
  memset(&data_params, 0, sizeof(adnn_data_parameters));
  status = ADNNDataAccess(node_sink, 0, ADNN_MEM_ACCESS_READ, &data_params);

  // move the data out here
  status = ADNNDataCommit(node_sink);

  // clean up
  status = ADNNDataDestroy(&node_src);
  status = ADNNDataDestroy(&node_sink);
  status = ADNNDataDestroy(&node_weights);
  status = ADNNDataDestroy(&node_bias);
  
  status = ADNNDataDestroy(&node_bot_df);
  status = ADNNDataDestroy(&node_weights_df);
  status = ADNNDataDestroy(&node_bias_df);
  status = ADNNDataDestroy(&node_top_df);
    
  status = ADNNodeDestroy(&node);

  return(status);
}

