// File:TensorBlob.cpp
// Date:1603150204
// Added instances - cleaned out comments (see stage4.7.built) 

#include <CL/opencl.h>
#include "/c/AMD/MLopen/adnn/inc/AMDnn.h"
#include <vector>
#include <map>
#include "/c/AMD/MLopen/adnn/inc/AMDnn.hpp"   // one of these may be gone
#include "/c/AMD/MLopen/caffe/include/caffe/blob.hpp"
#include "/c/AMD/MLopen/caffe/include/caffe/common.hpp"

// try static
cl_command_queue queue=0;  // 160415 : need this for the accessTensor, but in the AMDnn.hpp it defaults to zero

///////////////////////////////////////////////  BIND //////////////////////////////////////////////////////////////////
// NOT sure if this is possible, need to examine Tensor and Blob - The ideas is to just set a pointer to where the other
// object keeps its data.  Tensor -> Blob, so change in Tensor* will change Blob data
// Fast (no copy), but not sure where the data is kepts, so for now using the copy template methods below

// Set Tensor pointer to Blob Data
template<typename D> adnn::aDNNTensor* bind(const caffe::Blob<D> *blob, adnn::aDNNTensor *tensor)
{
  static_cast<D*>(tensor->accessTensor(ADNN_MEM_ACCESS_WRITE,queue))=blob->gpu_data();  // NEED QUEUE
}

// Set Blob pointer to Tensor data 
template<typename D> caffe::Blob<D>  * bind(/*const*/ adnn::aDNNTensor *tensor, caffe::Blob<D> *blob)
{
  blob->mutable_gpu_data()= static_cast<D*>(tensor->accessTensor(ADNN_MEM_ACCESS_READ,queue));  // NEED QUEUE
}

/////////////////////////////////////////////////// COPY /////////////////////////////////////////////////////////////////////
//
// Get a pointer to the data and use a straight MemCopy
// Future versions could use some sort of iterations :   *desptr=*srcptr;

// Copy from Blob into Tensor - the data is const - used when Blob is a botton/src/input blob
template<typename D> adnn::aDNNTensor * copy(const caffe::Blob<D> *blob, adnn::aDNNTensor *tensor)
{
  memcpy((void*)tensor->accessTensor(ADNN_MEM_ACCESS_WRITE,queue),(const void*)blob->gpu_data(),blob->count());
  tensor->commitTensor();  
  return(tensor);
}

// Copy Blob to Tensor - this data is not const (ie mutable) - used when Blob is top/des/output
// I am using this when I initialize the des_node (node_sec) Tensor  (for example when network is going through another iteration)
template<typename D> adnn::aDNNTensor * copy(caffe::Blob<D> *blob, adnn::aDNNTensor *tensor)
{
  memcpy((void*)tensor->accessTensor(ADNN_MEM_ACCESS_WRITE,queue),(const void*)blob->mutable_gpu_data(),blob->count());
  tensor->commitTensor();
  return(tensor);
}

// Copy From Tensor to Blob
// This needs to be done when we are have finished running (forward/backward)
// This copies data back to blob so other Layers in Caffe network can use it.
// Used by Train (and maybe Infer) adnn_term_layer

template<typename D> caffe::Blob<D> * copy(adnn::aDNNTensor *tensor, caffe::Blob<D> *blob)
{
  // >>> NEED to get this queue - either from aLibDNN (pass as arg), or pass the queue
  //      dest,src,bytes 
  memcpy((void*)blob->mutable_gpu_data(),(const void*)tensor->accessTensor(ADNN_MEM_ACCESS_READ,queue),tensor->getSizeInBytes());
  return(blob);
}

  template adnn::aDNNTensor    * copy(const caffe::Blob<float> *blob, adnn::aDNNTensor *tensor);
  template adnn::aDNNTensor    * copy(caffe::Blob<float> *blob, adnn::aDNNTensor *tensor);
  template caffe::Blob<float>  * copy(adnn::aDNNTensor *tensor, caffe::Blob<float> *blob);
  template adnn::aDNNTensor    * copy(const caffe::Blob<double> *blob, adnn::aDNNTensor *tensor);
  template adnn::aDNNTensor    * copy(caffe::Blob<double> *blob, adnn::aDNNTensor *tensor);
  template caffe::Blob<double> * copy(adnn::aDNNTensor *tensor, caffe::Blob<double> *blob);



// /c/AMD/MLopen/hybrid/stage4/inc/TensorBlob.hpp:73:116: warning: there are no arguments to ‘memncpy’ that depend on a template parameter, so a declaration of ‘memncpy’ must be available [-fpermissive]
//  memncpy(tensor->accessTensor(ADNN_MEM_ACCESS_READ,queue),(void*)blob->mutable_gpu_data(),tensor->getSizeInBytes());


// Exmple use
// copy(src_blob[i],static_cast<adnn::aDNNTensor*>src_node);  from Blob into Tensor (need SIZE!!! yet)
// copy(static_cast<adnn::aDNNTensor*>src_node,des_blob[i]);  // NEED SIZ

// instantiations
// template return_type method(Blob<float>
// template double method(Blob<double>);
