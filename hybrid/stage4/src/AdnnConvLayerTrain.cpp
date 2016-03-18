// [1] Version4.6 -- REMOVED vector<Blob>  now just passing blob
// File:AdnnConvLayerTrain.cpp
// See AdnnConvLayer.cpp,1 for pre deletions and pre insertions of 
// Stage4 
// Date:160310
// Delta: Cut out Usage and have FIXED parameters (the defaults)
// The main() has become adnn_run_conv_layer(alib_obj lib_handle, vector<Blob<Dtype>*>&src, vector<Blob<Dtype>*>&des); 
// From:main.cpp (mini-main.cpp) runs only one convolution layer, either in training(forward/back) or inference (forward only)

#include <cstdio>       // this defines malloc also, so commented out the malloc.h below
#include <iomanip>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <vector>
#include <map>
#include <string>
#include <CL/opencl.h>
#include "/c/AMD/MLopen/adnn/inc/AMDnn.h"
#include <vector>
#include <map>
#include <cassert>
#include "/c/AMD/MLopen/adnn/inc/AMDnn.hpp"   // one of these may be gone
#include "/c/AMD/MLopen/caffe/include/caffe/blob.hpp"
#include "/c/AMD/MLopen/caffe/include/caffe/common.hpp"
#include <TensorBlob.hpp>

// Stage4 : these are global for the Bridge, see AdnnSetupLayer.cpp (main4.cpp)
extern int
  adnn_n_categories,
  adnn_max_iterations,
  adnn_per_layer_iters,
  adnn_batch_sz,
  adnn_input_channels,
  adnn_input_w,
  adnn_input_h,
  adnn_n_output_features,
  adnn_filter_sz3,
  adnn_stride1;
extern bool
  adnn_conv_only,
  adnn_do_training,
  adnn_per_layer_timing,
  adnn_net_timing,
  adnn_verify;

extern int PrepareConvNode(alib_obj aLib,         // 1    ?? 8 or 12 parameters can be passed by register
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

// These would be class members and created during constructor
static  adata_obj node_src = 0, node_sink = 0, node_weights = 0, node_bias = 0;
static  adata_obj node_bot_df = 0, node_top_df = 0, node_weights_df = 0, node_bias_df = 0;
static  anode_obj node;

extern adnn_control_params      adnn_layer_control;
extern adnn_filter1D_parameters adnn_filter_params;
extern adnn_update_params       adnn_update_train_params;     // only need for Training

// Defined in AdnnSetupLayer.cpp
// Data extracted from Caffe's Blobs
// AdnnSetupLayer.cpp:92:32: warning: variable templates only available with -std=c++14 or -std=gnu++14
// see AdnnSetupLayer.c++14

const float * bias_data;   int bias_cnt=0;  
const float * weight_data; int weight_cnt=0;

// SOMEWHERE in caffe::net we need to call this with (blobs_[1])

template<typename DType> void adnn_init_bias(const caffe::Blob<DType> *blob)
{
  // The first component in a blob is the Bias
  bias_data = blob[0]->gpu_data();
  bias_cnt  = blob[0]->count();
}

template<typename DType> void adnn_init_weight(const caffe::Blob<DType> *blob)
{
  // The second component in a blob is the weight
  weight_data = blob[1]->gpu_data();
  weight_cnt  = blob[1]->count();
}

template <typename D> int adnn_init_layer_train(alib_obj lib_handle, const caffe::Blob<D> *src_blob, caffe::Blob<D> *des_blob)
{
  int status = 0;
  int size   = 0;
  cl_command_queue adnn_queue;
  
  printf("***********************************************************************************************\n");
  printf("STAGE4.2 : build  forward/backward propagation pipeline with a single stand-alone convolutional layer\n");
  printf("***********************************************************************************************\n");

  adnn_node_parameters node_params;
  memset(&node_params, 0, sizeof(adnn_node_parameters));

  // Step2: Prepare
  status = PrepareConvNode(lib_handle,
			   &adnn_layer_control,
			   &adnn_filter_params,
			   adnn_batch_sz,
			   adnn_input_channels,
			   adnn_input_h,
			   adnn_input_w,
			   adnn_n_output_features,
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
			   (adnn_node_parameters*)NULL);

  // Step3 : Allocate  
  status = ADNNodeConstruct(node);

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

  // initialize top_df
  adnn_data_init_parameters init_top_df;
  memset(&init_top_df, 0, sizeof(adnn_data_init_parameters));
  init_top_df.init_distr = ADNN_WD_GAUSSIAN;
  init_top_df.std = 0.01;
  status = ADNNDataInit(node_top_df, &init_top_df);

  if(!weight_cnt)
    {
      // initialization operator
      adnn_data_init_parameters init_weights;
      memset(&init_weights, 0, sizeof(init_weights));
      init_weights.init_distr = ADNN_WD_GAUSSIAN;
      init_weights.std = 0.01;
      // initilize (or upload) weights
      status = ADNNDataInit(node_weights, &init_weights);
    }
  else
    {
      adnn::aDNNTensor *weight_tensor = static_cast<adnn::aDNNTensor*>(node_weights);
      void *weight_ptr = weight_tensor->accessTensor(ADNN_MEM_ACCESS_WRITE,adnn_queue);
      memcpy(weight_ptr,(const void*)weight_data,weight_cnt);   /* dest,src,bytes */
    }
  
  if(!bias_cnt)
    {
      adnn_data_init_parameters init_bias;
      memset(&init_bias, 0, sizeof(init_bias));
      init_bias.init_distr = ADNN_WD_CONSTANT;
      init_bias.mean = 0.01;
      status = ADNNDataInit(node_bias, &init_bias);
    }
  else
    {
      adnn::aDNNTensor *bias_tensor = static_cast<adnn::aDNNTensor*>(node_bias);
      void *bias_ptr = bias_tensor->accessTensor(ADNN_MEM_ACCESS_WRITE,adnn_queue);
      memcpy(bias_ptr,(const void*)bias_data,bias_cnt);
    }

  // If we are passed data in the Input Blob (src_blob (the bottom))
  // Then we are copying from the Blob into the Tensor
  // Else we generate some random ness
  // NOTE: I can't find a good way to determine if the blob was filled.
  // .Count_ *= shape[i] in blob.cpp(31) 
  if(src_blob->count()>=1||src_blob->num_axes()>0)
    {
      std::cout << "TT: We have Blob data" << std::endl;
      (void)copy(src_blob,static_cast<adnn::aDNNTensor*>(node_src));  // gpu_data()
      (void)copy(des_blob,static_cast<adnn::aDNNTensor*>(node_sink));  // mutable_gpu_data
    }
  else
    {
      std::cout << "TT: We have EMPTY Blob data" << std::endl;
      adnn_data_parameters data_params;
      memset(&data_params, 0, sizeof(adnn_data_parameters));
      status = ADNNDataAccess(node_src, 0, ADNN_MEM_ACCESS_WRITE_DESTRUCT, &data_params);

      // initialize with something, anything, just be random
      for (size_t i = 0; i < data_params.size; ++i)
	{
	  ((float*)data_params.sys_mem)[i] = (float)((double)rand() * (1.0 / RAND_MAX));
	}
    }
  status = ADNNDataCommit(node_src);

  // OK, now we are all ready to go with Forward and Backward propogations
      
  return(status);
}

// To support tests in Stage3 (where caffe/src/caffe/layes/conv_layer.cpp, calls ...
template <typename T> int adnn_run_conv_layer(alib_obj lib_handle, const caffe::Blob<T> *src, caffe::Blob<T> *des)
{
  return(adnn_init_layer_train(lib_handle,src,des));
}

// Step__F: Run Forward   (in both Infer and Training)
// 160315 removed template and alib_obj, as neither are used, only static (eventually class) var 'node' is needed
// template <typename D>
int adnn_run_forward(/*alib_obj lib_handle*/)
{
  printf("***********************************************************************************************\n");
  printf("STAGE4t : run forward propagation pipeline with a single stand-alone convolutional layer\n");
  printf("***********************************************************************************************\n");
  // run forward propagation
  return(ADNNodeRunInference(node, NULL));
}

// Step__B: Run Backward  (only in Training)
// 160315 removed template and alib_obj, as neither are used
// template <typename D>
int adnn_run_backward(/*alib_obj lib_handle*/)
{
  printf("***********************************************************************************************\n");
  printf("STAGE4t : run backward propagation pipeline with a single stand-alone convolutional layer\n");
  printf("***********************************************************************************************\n");
  // run backward propagation
  return(ADNNodeRunTraining(node, NULL));
}

template <typename D> int adnn_term_layer(alib_obj lib_handle,caffe::Blob<D> *top_blob)
{
  int status;
  adnn_data_parameters data_params;  // holds metadata and pointers data (or actual buffer) 

  (void)copy(static_cast<adnn::aDNNTensor*>(node_sink),top_blob);

  memset(&data_params, 0, sizeof(adnn_data_parameters));
  status = ADNNDataAccess(node_sink, 0, ADNN_MEM_ACCESS_READ, &data_params);
  for (size_t i = 0; i < data_params.size; ++i)
    {
      std::cout << ((float*)data_params.sys_mem)[i];
    }
  status = ADNNDataCommit(node_sink);
}

// Step__ : Destroy or Term
// clean up
//template <typename D>
int adnn_cleanup_layer(alib_obj lib_handle)
{
  int status; 
  status = ADNNDataDestroy(&node_src);
  status += ADNNDataDestroy(&node_sink);
  status += ADNNDataDestroy(&node_weights);
  status += ADNNDataDestroy(&node_bias);

  status += ADNNDataDestroy(&node_bot_df);
  status += ADNNDataDestroy(&node_weights_df);
  status += ADNNDataDestroy(&node_bias_df);
  status += ADNNDataDestroy(&node_top_df);
    
  status += ADNNodeDestroy(&node);

  return(status);
}

template int adnn_init_layer_train(alib_obj lib_handle, const caffe::Blob<float>  *src, caffe::Blob<float>  *des);
template int adnn_init_layer_train(alib_obj lib_handle, const caffe::Blob<double> *src, caffe::Blob<double> *des);
template int adnn_run_conv_layer(alib_obj lib_handle, const caffe::Blob<float> *src, caffe::Blob<float> *des);
template int adnn_run_conv_layer(alib_obj lib_handle, const caffe::Blob<double> *src, caffe::Blob<double> *des);

// END
