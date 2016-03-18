// File:TensorBlob.cpp
// Date:1603150204

#include <vector>
#include <CL/opencl.h>
#include "/c/AMD/MLopen/adnn/inc/AMDnn.h"
//#include "/c/AMD/MLopen/adnn/inc/AMDnn.hpp" 
#include "/c/AMD/MLopen/caffe/include/caffe/blob.hpp"
#include "/c/AMD/MLopen/caffe/include/caffe/common.hpp"


/* >>> from aDNNTensor.hpp CDNN_Tensor::deepCopyBlob <<<<<
// AMDnn.hpp:334:    inline size_t     getSizeInBytes(void) const    {      return(size_bytes_);    }
cl_command_queue curr_queue = (queue != 0) ? queue : prefered_queue_;
_T *t_ptr = accessTensor(CL_MAP_WRITE_INVALIDATE_REGION, curr_queue);
_T *s_ptr = source.accessTensor(CL_MAP_READ);       <<<<<< REPLACE with src_data 
memcpy(t_ptr, s_ptr, getSizeInBytes());             <<<<<<<< won't work since we are coping IN from Caffe - need a caffe Blob method
commitTensor();
source.commitTensor();
*/

///////////////////////////////////////////////  BIND //////////////////////////////////////////////////////////////////
// NOT sure if this is possible, need to examine Tensor and Blob - The ideas is to just set a pointer to where the other
// object keeps its data.  Tensor -> Blob, so change in Tensor* will change Blob data
// Fast (no copy), but not sure where the data is kepts, so for now using the copy template methods below

// Set Tensor pointer to Blob Data
template<typename D> adnn::aDNNTensor* bind(const caffe::Blob<D> *blob, adnn::aDNNTensor tensor)
{
  static_cast<D*>(tensor->acessTensor(ADNN_MEM_ACCESS_WRITE,queue))=blob->gpu_data();  // NEED QUEUE
}

// Set Blob pointer to Tensor data 
template<typename D> caffe::Blob<D>  * bind(const adnn::aDNNTensor *tensor, caffe::Blob<D> *blob);
{
  blob->mutable_gpu_data()= static_cast<D*>(tensor->acessTensor(ADNN_MEM_ACCESS_READ,queue));  // NEED QUEUE
}

/////////////////////////////////////////////////// COPY /////////////////////////////////////////////////////////////////////
//
// Get a pointer to the data and use a straight MemCopy
// Future versions could use some sort of iterations :   *desptr=*srcptr;

// Copy from Blob into Tensor - the data is const - used when Blob is a botton/src/input blob
template<typename D> adnn::aDNNTensor* copy(const caffe::Blob<D> *blob, adnn::aDNNTensor *tensor)
{
  memncpy((void*)blob->gpu_data(),(void*)tensor->accessTensor(ADNN_MEM_ACCESS_WRITE,queue),blob->count());
  tensor->commitTensor() 
  return(tensor);
}

// Copy Blob to Tensor - this data is not const (ie mutable) - used when Blob is top/des/output
// I am using this when I initialize the des_node (node_sec) Tensor  (for example when network is going through another iteration)
template<typename D> adnn::aDNNTensor* copy(caffe::Blob<D> *blob, adnn::aDNNTensor *tensor)
{
  memncpy((void*)blob->mutable_gpu_data(),(void*)tensor->accessTensor(ADNN_MEM_ACCESS_WRITE,queue),blob->count());
  tensor->commitTensor() 
  return(tensor);
}

// Copy From Tensor to Blob
// This needs to be done when we are have finished running (forward/backward)
// This copies data back to blob so other Layers in Caffe network can use it.
// Used by Train (and maybe Infer) adnn_term_layer

template<typename D> caffe::Blob<D>  * copy(const adnn::aDNNTensor *tensor, caffe::Blob<D> *blob)
{
  // >>> NEED to get this queue - either from aLibDNN (pass as arg), or pass the queue 
  memncpy((void*)tensor->accessTensor(ADNN_MEM_ACCESS_READ,queue),(void*)blob->mutable_gpu_data(),tensor->getSizeInBytes());
  return(blob);
}


// Exmple use
// copy(src_blob[i],static_cast<adnn::aDNNTensor*>src_node);  from Blob into Tensor (need SIZE!!! yet)
// copy(static_cast<adnn::aDNNTensor*>src_node,des_blob[i]);  // NEED SIZ

// instantiations
// template return_type method(Blob<float>
// template double method(Blob<double>);

