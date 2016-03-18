// [1] Version4.6 -- REMOVED vector<Blob>  now just passing blob
// 160316 : Had to put in some boost::shared_ptr<>   NEED to do for Train
// File:AdnnConvLayerInfer.cpp
// See AdnnConvLayer.cpp,1 for pre deletions and pre insertions of 
// Stage4.7 see notes in Evolution /c/AMD/MLopen/hybrid/Evolution.notes 
// Date:160315
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
  adnn_filter_sz,
  adnn_stride;
extern bool
  adnn_conv_only,
  adnn_do_training,
  adnn_per_layer_timing,
  adnn_net_timing,
  adnn_verify;

extern const adnn_control_params      adnn_layer_control;
extern const adnn_filter1D_parameters adnn_filter_params;
//extern       cl_command_queue         adnn_cl_queue;       // do we want created and initialized once, or pull it from library 

extern int PrepareConvNode(alib_obj aLib,         // 1    ?? 8 or 12 parameters can be passed by register
	const adnn_control_params *layer_control, // 2    the rest need to be put on stack (or ?? ) and thus incur cost
	const adnn_filter1D_parameters *f_params, // 3    IS there a way we can group these into a structure and pass
        int batch_sz,				  // 4    a pointer to the structure ?? 
	int input_channels,			  // 5
	int input_h,				  // 6
	int input_w,				  // 7
	int n_output_featuremaps,		  // 8
	const char *node_name,			  // 9
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
static adata_obj node_src = 0, node_sink = 0, node_weights = 0, node_bias = 0; // Tensors
static anode_obj node;            // equavalent to a 'layer' in Caffe 

// Defined in AdnnSetupLayer.cpp
// Data extracted from Caffe's Blobs
// AdnnSetupLayer.cpp:92:32: warning: variable templates only available with -std=c++14 or -std=gnu++14
// see AdnnSetupLayer.c++14

const double * bias_data;   int bias_cnt=0;  
const double * weight_data; int weight_cnt=0;   // FUDGE

// 160316 : change to use boost::shared_ptr 
template<typename D> void adnn_init_bias(boost::shared_ptr< caffe::Blob<D> >& blob)
{
  std::cout << "AdnnInitLayerInfer : AdnnInitBiase  FIXME" << std::endl;
  // The first component in a blob_ is the Bias
  //  bias_data = blob->gpu_data();
  //  bias_cnt  = blob->count();
}

template<typename D> void adnn_init_weight(boost::shared_ptr< caffe::Blob<D> >& blob)
{
  std::cout << "AdnnInitLayerInfer : AdnnInitWeight  FIXME" << std::endl;
  // The second component in a blob_ is the weight
  // weight_data = blob->gpu_data();
  // weight_cnt  = blob->count();
}

template <typename D> int adnn_init_layer_infer(alib_obj lib_handle, const caffe::Blob<D> *src_blob, caffe::Blob<D> *des_blob)
{
  int status = 0;
  cl_command_queue adnn_queue;
  
  printf("***********************************************************************************************\n");
  printf("Stage4.1 : build forward propagation pipeline with a single stand-alone convolutional layer\n");
  printf("***********************************************************************************************\n");
  std::cout << "AdnnInitLayerInfer : INIT \n\t n_categories:"<<adnn_n_categories
	    << "\n\t batch_sz:"         <<adnn_batch_sz
	    << "\n\t input_channels:"   <<adnn_input_channels
	    << "\n\t input_w:"          <<adnn_input_w
	    << "\n\t input_h:"          <<adnn_input_h
	    << "\n\t n_output_features:"<<adnn_n_output_features
	    << "\n\t filter_sz:"        <<adnn_filter_sz
	    << "\n\t stride:"           <<adnn_stride
	    << std::endl;

  // >>>> Need items in adnn_setup_layer >>> Step1.2:InitParameters <<<<<

  // Step2: Prepare
  status = PrepareConvNode(lib_handle,
			   &adnn_layer_control,    // these are const
			   &adnn_filter_params,    // also
			   adnn_batch_sz,
			   adnn_input_channels,
			   adnn_input_h,
			   adnn_input_w,
			   adnn_n_output_features,  //maps,
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
			   (adnn_node_parameters*)NULL);     // * adnn_node_parameters

  // Step3 : Allocate  
  status = ADNNodeConstruct(node);

  // allocate data before the node build
  status = ADNNDataAllocate(node_src, 0);
  status = ADNNDataAllocate(node_weights, 0);
  status = ADNNDataAllocate(node_bias, 0);
  status = ADNNDataAllocate(node_sink, 0);
  status = ADNNodeBuild(node);

  if(!weight_cnt)
    {
      // initialization operator
      adnn_data_init_parameters init_weights;
      std::cout << "AdnnInitLayerInfer Using GAUSSIAN weights" << std::endl;      
      memset(&init_weights, 0, sizeof(init_weights));
      init_weights.init_distr = ADNN_WD_GAUSSIAN;
      init_weights.std = 0.01;
      // initilize (or upload) weights
      status = ADNNDataInit(node_weights, &init_weights);
    }
  else
    {
      std::cout << "AdnnInitLayerInfer Using External Weights" << std::endl;
      adnn::aDNNTensor *weight_tensor = static_cast<adnn::aDNNTensor*>(node_weights);
      D *weight_ptr = static_cast<D*>(weight_tensor->accessTensor(ADNN_MEM_ACCESS_WRITE,adnn_queue));
      memcpy(weight_ptr,weight_data,weight_cnt);
    }
  
  if(!bias_cnt)
    {
      adnn_data_init_parameters init_bias;
      std::cout << "AdnnInitLayerInfer Using CONSTANT Bias" << std::endl;
      memset(&init_bias, 0, sizeof(init_bias));
      init_bias.init_distr = ADNN_WD_CONSTANT;
      init_bias.mean = 0.01;
      status = ADNNDataInit(node_bias, &init_bias);
    }
  else
    {
      std::cout << "AdnnInitLayerInfer Using External Bias" << std::endl;
      adnn::aDNNTensor *bias_tensor = static_cast<adnn::aDNNTensor*>(node_bias);
      D *bias_ptr = static_cast<D*>(bias_tensor->accessTensor(ADNN_MEM_ACCESS_WRITE,adnn_queue));
      memcpy(bias_ptr,bias_data,bias_cnt);
    }

  // >>> Now we need to get the data out of the blob <<<< 
  const
  D *src_data; // from bottom blob
  D *des_data; // from top blob
  
  // extract/copy Blobs into node_src, and node_sink
  // ideally like to assign blob pointer to sys_mem of tensor 
  // see 'train' which will set these to randome values
  
  // I believe this is the data for node_src (src_data) and node_sink (des_data)
  src_data = src_blob->gpu_data();
  des_data = des_blob->mutable_gpu_data();

  copy(src_blob,static_cast<adnn::aDNNTensor*>(node_src));  // gpu_data()
  copy(des_blob,static_cast<adnn::aDNNTensor*>(node_sink));  // mutable_gpu_data
      
  return(status);
}

// Need this for EACH template type we plan to use - Caffe uses two data types: float and double
// 160311 NOW we have an issue - these are missing in caffe/src/caffe/layers/conv_layer.cpp - so adding in as externs???
// ACTUALLY the error was missing const
template int adnn_init_layer_infer(alib_obj lib_handle, const caffe::Blob<float>  *src, caffe::Blob<float>  *des);
template int adnn_init_layer_infer(alib_obj lib_handle, const caffe::Blob<double> *src, caffe::Blob<double> *des);
template void adnn_init_bias(boost::shared_ptr< caffe::Blob<float> >& blob);
template void adnn_init_weight(boost::shared_ptr< caffe::Blob<double> >& blob);
template void adnn_init_bias(boost::shared_ptr< caffe::Blob<double> >& blob);
template void adnn_init_weight(boost::shared_ptr< caffe::Blob<float> >& blob);

// To support tests in Stage3 (where caffe/src/caffe/layes/conv_layer.cpp, calls ...
template <typename T> int adnn_run_conv_layer(alib_obj lib_handle, const caffe::Blob<T> *src, caffe::Blob<T> *des)
{
  return(adnn_init_layer_infer(lib_handle,src,des));
}
template int adnn_run_conv_layer(alib_obj lib_handle, const caffe::Blob<float> *src, caffe::Blob<float> *des);
template int adnn_run_conv_layer(alib_obj lib_handle, const caffe::Blob<double> *src, caffe::Blob<double> *des);

// Step__F: Run Forward   (in both Infer and Training)
int adnn_run_forward(/*alib_obj lib_handle*/)
{
  printf("***********************************************************************************************\n");
  printf("STAGE4i : run forward propagation pipeline with a single stand-alone convolutional layer\n");
  printf("***********************************************************************************************\n");
  // run forward propagation
  return(ADNNodeRunInference(node, NULL));
}

int adnn_run_backward(/*alib_obj lib_handle*/)
{
  printf("***********************************************************************************************\n");
  printf("STAGE4i : run backward propagation pipeline with a single stand-alone convolutional layer\n");
  printf("***********************************************************************************************\n");
  // run backward propagation
  return(ADNNodeRunTraining(node, NULL));
}

/// WHO IS CALLING THIS ---- need in conv_layer.cpp

template <typename D> int adnn_term_layer(alib_obj lib_handle,caffe::Blob<D> *top_blob)
{
  int status;
  adnn_data_parameters data_params;  // holds metadata and pointers data (or actual buffer) 

  // One way
  // We just need to copy out the values from the node_des ito the Blob -  BUT WHERE IS THE BLOB 
  copy(static_cast<adnn::aDNNTensor*>(node_sink),top_blob);
  
  // Original
  // download output
  // 
  memset(&data_params, 0, sizeof(adnn_data_parameters));
  status = ADNNDataAccess(node_sink, 0, ADNN_MEM_ACCESS_READ, &data_params);

  // move the data out here
  status = ADNNDataCommit(node_sink);
}

template int adnn_term_layer(alib_obj lib_handle,caffe::Blob<float> *top_blob);
template int adnn_term_layer(alib_obj lib_handle,caffe::Blob<double> *top_blob);


// Step__ : Destroy or Term
// clean up
// template <typename D>
int adnn_cleanup_layer(alib_obj lib_handle)
{
  int status; 
  status = ADNNDataDestroy(&node_src);
  status = ADNNDataDestroy(&node_sink);
  status = ADNNDataDestroy(&node_weights);
  status = ADNNDataDestroy(&node_bias);
  
  status = ADNNodeDestroy(&node);

  return(status);
}
// END
