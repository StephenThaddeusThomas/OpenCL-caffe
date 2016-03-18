// File: RunSingleConvLayerInfer.cpp
// This only has Forward convolution
// This is step one of the mini test 
// from aLibDNNDriver.cpp line 1909

// NOTE: This and RunSingleConvLayerTrain both came from the same set of sources
// These are training files. This INFER version runs a FORWARD ONLY Network
// The TRAIN runs FORWARD and BACKWARE with an UPDATE
// These versions are simplier since the multiple levesl of if(training) have been removed
// USE THIS FILE for NOTES common to both (they both do FORWARD) 

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

extern int PrepareConvNode(alib_obj aLib, //         1    ?? 8 or 12 parameters can be passed by register
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
	adata_obj *node_bot_df = NULL,		  // 17
	adata_obj *node_top_df = NULL,		  // 18
	adata_obj *node_weights_df = NULL,	  // 19
	adata_obj *node_bias_df = NULL,		  // 20
	bool training = false,			  // 21
	bool inference = true,			  // 22
        adnn_node_parameters *pnode_params = NULL); // 23
			   

int
ADNNSingleConvLayer(alib_obj aLib,
		    const adnn_control_params *layer_control,
		    const adnn_filter1D_parameters *filter_params,
		    int batch_sz,
		    int input_channels,
		    int input_h,
		    int input_w,
		    int n_output_featuremaps,
		    bool training,
		    adnn_update_params *pupdate_params)
{
  int status = 0;
  
  printf("***********************************************************************************************\n");
  printf("TEST : build and run forward propagation pipeline with a single stand-alone convolutional layer\n");
  printf("***********************************************************************************************\n");

  adata_obj node_src = 0, node_sink = 0, node_weights = 0, node_bias = 0;
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
			   &node,  // conv1
			   0,	   // bot_df
			   0,	   // top_df
			   0,	   // weights_df
			   0,	   // bias_df
			   false,  // training
			   true,   // inference
			   0);

  status = ADNNodeConstruct(node);

  // allocate data before the node build
  status = ADNNDataAllocate(node_src, 0);
  status = ADNNDataAllocate(node_weights, 0);
  status = ADNNDataAllocate(node_bias, 0);
  status = ADNNDataAllocate(node_sink, 0);
  status = ADNNodeBuild(node);

  /** NOTES ************************************************************************************************
      Alex came by to describe what's going on here (phew)
      Basically since his Framework is used for testing it therefore has no real external data to run against
      So, it generates its own Internal data.  There are two sections where this is done.  The first is the
      1. Implicit Initialization and the 
      2. Explicit Initialization 
      What I need to ask his is how we get the data back out. 
  ******************************************************************************************************** */

  // 1. IMPLICITE
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

  // 2. EXPLICIT 
  // upload source
  // To do this we populate the data_params object by calling ADNNDataAccess
  // HOW can we SET the size of the data buffers sys_mem 
  adnn_data_parameters data_params;  // holds metadata and pointers data (or actual buffer) 
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
  
  status = ADNNodeDestroy(&node);

  return(status);
}

