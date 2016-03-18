// File:AMDNnImpl.cpp
// This is the C++ class that is called by the C interface
//----------------------------------------------------------------------------------------------------------------------------
/**********************************************************************
Copyright ?2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

?	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
?	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#include "aDNNInternal.hpp"

namespace adnn
{
  /************************************************************************************************************************
   **
   **			ADNNBase class
   **
   ************************************************************************************************************************/

  /**
   * Constructors
   */
  ADNNBase::ADNNBase()
  {
    internal_ = NULL;
    ref_count_ = 0;
    retain();
  }

  /**
   * Destructor
   */
  ADNNBase::~ADNNBase(void)
  {
    destroyInternal();
//		release();
  }

  int ADNNBase::release(void)
  {
    int ret = ref_count_--;
    if (ref_count_ == 0)
      {
	delete this;
      }
    return(ret);
  }

#if 0
  const std::string & ADNNBase::getName(void) const
  {
    if (getInternal())
      {
	return(((CDNN_Object*)getInternal())->getName());
      }
    return(empty_str);
  }
#endif

  int ADNNBase::removeInternal(void)
  {
    int ret = 0;
    if (getInternal())
      {
#if 0
	CDNN_Object * parent = &((CDNN_Object*)getInternal())->getParent();
	if (parent)
	  {
	    parent->removeObj(getInternal());
	  }
#endif
      }
    return(ret);
  }
  
  int ADNNBase::destroyInternal(void)
  {
    int ret = 0;
    removeInternal();
    if (getInternal())
      {
//	delete ((CDNN_Object*)getInternal());
	internal_ = 0;
      }
	return(ret);
  }
  void *ADNNBase::getWrapper(void)
  {
    void * ret = 0;
    if (getInternal())
      {
//	ret = ((CDNN_Object*)getInternal())->getWarpper();
      }
    return(ret);
  }

  /************************************************************************************************************************
   **
   **			aDNNEdge Class
   **
   ************************************************************************************************************************/

  aDNNEdge::aDNNEdge(void) : ADNNBase()
  {
    edge_type_ = ADNN_ED_INTERNAL;
    data_ = NULL;
    data_updated_ = false; // indicate new data
    weights_updated_ = false;
    pooling_method_ = ADNN_POOLING_AVE;
    memset(&lrn_parameters_, 0, sizeof(adnn_lrn_parameters));
  }

  aDNNEdge::aDNNEdge(const ADNNBase & lib, const adnn_net_edge_parameters & c_descr) : ADNNBase()
  {
    setParent((void*)&lib);     // ?? why not in ADNNBase(lib) 
    edge_type_ = c_descr.edge_type;
    name_ = (c_descr.name) ? c_descr.name : "";
    data_ = (aDNNTensor *)c_descr.data;
    weights_data_ = (aDNNTensor *)c_descr.weights;
    bias_data_ = (aDNNTensor *)c_descr.bias;
    filter_params_ = c_descr.filter_params;
    pooling_method_ = c_descr.pooling_method;
    lrn_parameters_ = c_descr.lrn_parameters;
    
    data_updated_ = false; // indicate new data
    weights_updated_ = false;
  }

  aDNNEdge::aDNNEdge(const aDNNEdge & rh)
  {
    *this = rh;
  }

  aDNNEdge::~aDNNEdge(void)
  {
  }

  const aDNNEdge & aDNNEdge::operator = (const aDNNEdge & rh)
  {
    *(ADNNBase*)this = *(ADNNBase*)&rh;
    edge_type_ = rh.edge_type_;
    name_ = rh.name_;
    data_ = rh.data_;
    data_ = rh.data_;
    weights_data_ = rh.weights_data_;
    bias_data_ = rh.bias_data_;
    filter_params_ = rh.filter_params_;
    pooling_method_ = rh.pooling_method_;
    lrn_parameters_ = rh.lrn_parameters_;
    data_updated_ = rh.data_updated_; // indicate new data
    weights_updated_ = rh.weights_updated_;
    return *this;
  }
}; // adnn

/************************************************************************************************************************
**
**			ADNN Library
**
************************************************************************************************************************/

alib_obj ADNNLibCreate(const adnn_lib_parameters * lib_params)
{
  cl_context l_context = lib_params->context;
  cl_device_type l_accel_type = (lib_params->accel_type) ? lib_params->accel_type : CL_DEVICE_TYPE_GPU;
  std::string l_accel_p = (lib_params->accel_platform) ? lib_params->accel_platform : "Advanced Micro Devices, Inc.";
  std::string l_kern_p = (lib_params->ocl_kernels_path) ? lib_params->ocl_kernels_path : "";
  
  adnn::ADNNLib * newLib = adnn::ADNNLibCreate(l_context, l_accel_type, l_accel_p, l_kern_p);
  assert(newLib);
  
  return((alib_obj)newLib);
}

int ADNNLibDestroy(alib_obj * alib)
{
  int ret = ADNN_SUCCESS;
  assert(alib);
  adnn::ADNNLibDestroy((adnn::ADNNLib *)*alib);
  return(ret);
}

const char *ADNNLibGetName(alib_obj lib)
{
  return (((adnn::ADNNLib *)lib)->getName().c_str());
}

int ADNNLibInspect(alib_obj alib, ADNNLIB_INSPECT cause, size_t * size, void * data)
{
  int ret = ADNN_SUCCESS;
  adnn::ADNNLib * Lib = (adnn::ADNNLib *)alib;
  if (Lib)
    {
      switch (cause)
	{
	case ADNNLIB_INSPECT_CONTEXT:
	  if (data)
	    {
	      *(cl_context *)data = Lib->getContext();
	    }
	  else
	    {
	      *size = sizeof(cl_context);
	    }
	  break;

	case ADNNLIB_INSPECT_DEVICES:
	  printf("Not implemented\n");
	  break;

	default:
	  printf("ERROT: unknown cause: %d\n", cause);
	  ret = ADNN_GENERAL_FAILURE;
	  break;
	}
    }
  else
    {
      ret = ADNN_GENERAL_FAILURE;
    }
  return(ret);
}

int ADNNLibCreateDeviceQueue(alib_obj alib,
			     cl_device_id deviceId,
			     const cl_command_queue_properties *prop,
			           cl_command_queue * new_queue)
{
  int ret = ADNN_SUCCESS;
  adnn::ADNNLib * Lib = (adnn::ADNNLib *)alib;
  if (Lib && new_queue)
    {
      *new_queue = Lib->createQueue(deviceId, prop);
    }
//	printf("ADNNLibCreateDeviceQueue : not implemented\n");
  return(ret);
}


/************************************************************************************************************************
**
**			aDNN
**
************************************************************************************************************************/


anet_obj ADNNCreate(alib_obj library, const adnn_net_parameters * net_params)
{
  adnn::ADNNLib * lib = (adnn::ADNNLib *)library;
  adnn_net_parameters fixed_net_params = *net_params;
  fixed_net_params.type = ADNN_NODE_NET;
  adnn::aDNNode* new_net = lib->ADNNCreate(fixed_net_params);
  return((anet_obj)new_net);
}


int ADNNDestroy(anet_obj * anet)
{
  int ret = ADNN_SUCCESS;
  
  adnn::ADNN* net = (adnn::ADNN*)(*anet);
  adnn::ADNNLib * lib = (adnn::ADNNLib *)net->getParent();
  
  assert(net);
  
  ret = lib->ADNNDestroy(net);
  
  return(ret);
}

const char *ADNNGetName(anet_obj net)
{
  return (((adnn::aDNNode *)net)->getName().c_str());
}

int ADNNConnect(anet_obj anet)
{
  int ret = 0;
  adnn::ADNN* net = (adnn::ADNN*)(anet);
  ret = net->Connect();
  return(ret);
}

/*-------------------------------------------------------------------------
Construct inference only net.
defines build and run options for each Node,
creates execution plans,
calculates memory reqierment for each Node and total per a Net run.
--------------------------------------------------------------------------*/

int ADNNConstruct(anet_obj anet)
{
  int ret = 0;
  adnn::aDNNode* net = (adnn::aDNNode*)(anet);
  ret = net->Construct();
  return(ret);
}

/*-------------------------------------------------------------------------
Construct training net.
defines build and run options for each Node,
creates execution plans,
calculates memory reqierment for each Node and total per a Net run.
--------------------------------------------------------------------------*/
int ADNNConstructTraining(anet_obj anet)		// Net object.
{
  int ret = 0;
  adnn::aDNNode* net = (adnn::aDNNode*)(anet);
  ret = net->Construct();
  ret = net->ConstructBwd();
  return(ret);
}

/*-------------------------------------------------------------------------
Build execution passes for inference only net
allocates memory for internal buffers,
compiles kernels,
finalizes execution plans per a Node,
considers global Net optimizations.
--------------------------------------------------------------------------*/

int ADNNBuild(anet_obj anet)
{
  int ret = 0;
  adnn::aDNNode* net = (adnn::aDNNode*)(anet);
  ret = net->Build();
  return(ret);
}

/*-------------------------------------------------------------------------
Build execution passes for training net
allocates memory for internal buffers,
compiles kernels,
finalizes execution plans per a Node,
considers global Net optimizations.
--------------------------------------------------------------------------*/
int ADNNBuildTraining(anet_obj anet)			// Net object.
{
  int ret = 0;
  adnn::aDNNode* net = (adnn::aDNNode*)(anet);
  ret = net->Build();
  ret = net->BuildBwd();
  return(ret);
}

/*-------------------------------------------------------------------------
runs one inference iteration,
accepting udates to a running parameters per each Node.
--------------------------------------------------------------------------*/
int ADNNRunInference(anet_obj anet, int n_running_params, const adnn_node_parameters * running_params)
{
  int ret = 0;
  adnn::ADNN* net = (adnn::ADNN*)(anet);
  ret = net->RunFwd(n_running_params, running_params);
  return(ret);
}

/*-------------------------------------------------------------------------
runs one training iteration,
accepting udates to a running parameters per each Node.
--------------------------------------------------------------------------*/
int ADNNRunTraining(anet_obj anet,				// Net object,
	int n_running_params,					// number updating parameter structures,
	const adnn_node_parameters * running_params)		// array of updating parameter structures.
{
  int ret = 0;
  adnn::ADNN* net = (adnn::ADNN*)(anet);
  ret = net->RunFwd(n_running_params, running_params);
  ret = net->RunBwd();
  ret = net->UpdateWeights();
  return(ret);
}


/************************************************************************************************************************
**
**			aDNNode
**
************************************************************************************************************************/

anode_obj ADNNodeCreate(alib_obj library, const adnn_node_parameters * layer_params)
{
  anode_obj ret = 0;
  adnn::ADNNLib * lib = (adnn::ADNNLib *)library;

  ret = (anode_obj)lib->createNode(*layer_params);
  
  return(ret);
}

/*
add set of layers to the net.
each layer input edge's name has to have a pair node name, evccept for the input node
*/

// TT: this seems odd that we have to delete the Nodes[]
// Basically need to convert the anode_objs into real adnn::aDNNodes before passing to ADNN
int ADNNodesAdd(anet_obj net, int n_nodes, const anode_obj * nodes)
{
  int ret = 0;
  adnn::ADNN * net_obj = (adnn::ADNN *)net;

  adnn::aDNNode ** Nodes = new adnn::aDNNode *[n_nodes];
  for (int i = 0; i < n_nodes; ++i)
    {
      Nodes[i] = (adnn::aDNNode *)nodes[i];
    }

  ret = net_obj->AddNodes(n_nodes, Nodes);

  delete[] Nodes;
  return(ret);
}

// ?? since we've passed all the data and pointers into ADNN - we tell it to delete
int ADNNodeDestroy(anode_obj * alayer)
{
  int ret = ADNN_SUCCESS; // really is no other return (might just have return(ADNN_SUCCESS); below
  adnn::aDNNode * Node = (adnn::aDNNode *)*alayer;
  adnn::ADNNLib & lib = *(adnn::ADNNLib *)Node->getParent();
  lib.removeNode(Node);  // is this recurse??
  return(ret);
}

/*-------------------------------------------------------------------------
updates Node's parmeters
--------------------------------------------------------------------------*/
// TODO ??
int ADNNodeUpdate(anode_obj  alayer,			// ADNNode object
	const adnn_node_parameters * layer_params)	// updated set of parameters.
{
  int ret = 0;
  printf("ADNNodeUpdate is not inplemented\n");
  return(ret);
}


const char *ADNNodeGetName(anode_obj alayer)
{
  const char * ret = 0;
  return(ret);
}


// defines build and run options
// calculates memory reqierment
int ADNNodeConstruct(anode_obj node)
{
  int ret = 0;
  adnn::aDNNode * Node = (adnn::aDNNode *)node;
  ret = Node->Construct();
  return(ret);
}

/*-------------------------------------------------------------------------
defines build and run options for inference (forward propagation pass) nad training (backward pass).
calculates memory reqierment
--------------------------------------------------------------------------*/

int ADNNodeConstructTraining(anode_obj node)					// ADNNode object
{
  int ret = 0;
  adnn::aDNNode * Node = (adnn::aDNNode *)node;
  ret = Node->Construct();
  ret |= Node->ConstructBwd();
  return(ret);
}

// compiles execution kernels
// memory has to be allocated at this point
int ADNNodeBuild(anode_obj node)
{
  int ret = 0;
  adnn::aDNNode * Node = (adnn::aDNNode *)node;
  ret = Node->Build();
  return(ret);
}

/*-------------------------------------------------------------------------
inference (forward pass) + training (backward pass)
compiles execution kernels,
caches (passes) OCL kernel arguments,
creates execution plan.

Notes:
it's preferable but not mandotary to have memory to be allocated at this point.
the actual OCL memory buffers can be passed at run-time.
--------------------------------------------------------------------------*/

int ADNNodeBuildTraining(anode_obj node)					// ADNNode object
{
  int ret = 0;
  adnn::aDNNode * Node = (adnn::aDNNode *)node;
  ret = Node->Build();
  ret |= Node->BuildBwd();
  return(ret);
}

// runs one iteration
// source data has to be uploaded
// weights have to be initilized or uploaded at this point

int ADNNodeRunInference(anode_obj node, const adnn_node_parameters * running_params)
{
  int ret = 0;
  adnn::aDNNode * Node = (adnn::aDNNode *)node;
  Node->RunFwd(running_params);
  return(ret);
}

/*-------------------------------------------------------------------------
runs a single backward iteration.

Notes:
at this point:
source data has to be uploaded,
weights have to be initilized or uploaded.
--------------------------------------------------------------------------*/

int ADNNodeRunTraining(anode_obj node,			// ADNNode object
	const adnn_node_parameters * running_params)	// node parameters structure passing run-time arguments if needed.
{
  int ret = 0;
  adnn::aDNNode * Node = (adnn::aDNNode *)node;
  Node->RunBwd(running_params);
  Node->UpdateWeights();
  return(ret);
}


int ADNNodeInspect(anode_obj node, ADNNODE_INSPECT cause, size_t * size, void * info)
{
  int ret = 0;
  assert(size);
  adnn::aDNNode * Node = (adnn::aDNNode *)node;

  switch (cause)
    {
    case ADNNODE_INSPECT_INPUT_EDGE :			// input edge
    case ADNNODE_INSPECT_OUTPUT_EDGE :			// output edge
      printf("Not implemented cause: %d\n", cause);
      break;

    case ADNNODE_INSPECT_EXECUTION_PLAN_FWD:
      {
	int n = (int)Node->getForwardExeStages().size();
	if (!info)
	  {
	    *size = sizeof(adnn_node_exe_parameters) * n;
	  }
	else if (*size == sizeof(adnn_node_exe_parameters) * n)
	  {
	    
	    adnn_node_exe_parameters * p_info = (adnn_node_exe_parameters *)info;

	    for (int i = 0; i < n; ++i)
	      {
		Node->getForwardExeStages()[i].retrieveExeParams(p_info[i]);
	      }
	  }
	else
	  {
	    printf("ERROR: incorrect size: %zd\n", *size);
	  }
      }
      break;

    case ADNNODE_INSPECT_EXECUTION_PLAN_BWD:
      printf("Not implemented cause: %d\n", cause);
      break;

    case ADNNODE_INSPECT_N_SLOTS:		// number of internal data slots
      if (!info)
	{
	  *size = sizeof(size_t);
	}
      else if (*size == sizeof(size_t))
	{
	  *(size_t*)info = Node->getNSlots();
	}
      break;

    case ADNNODE_INSPECT_SLOT_LIST:		// list of internal slot names
      if (!info)
	{
	  *size = sizeof(char*) * Node->getNSlots();
	}
      else if (*size == sizeof(size_t) * Node->getNSlots())
	{
	  Node->getSlotsNames((const char**)info);
	}
      break;

    case ADNNODE_INSPECT_SLOT:			// internal data object assosiated with the slot
      if (!info)
	{
	  *size = sizeof(void*);
	}
      else if (*size == sizeof(void*))
	{
	  std::string slot_nm = (char*)info;
	  *(anode_obj*)info = *(anode_obj*)&Node->getSlot(slot_nm);
	}
      break;

    default:
      printf("ERROR: unknown inpection cause: %d\n", cause);
      break;
    }
  return(ret);
}

/************************************************************************************************************************
**
**			ADNN MD
**
************************************************************************************************************************/

adata_obj ADNNDataCreate(alib_obj library, const adnn_data_parameters * data_params)
{
  adata_obj ret = 0;
  adnn::ADNNLib * lib = (adnn::ADNNLib *)library;
  
  ret = lib->createTensor(*data_params);

  return(ret);
}

/*-------------------------------------------------------------------------
create a new data object with the same parameters as the original.
has to be allocated and destroyed independently from teh original.
--------------------------------------------------------------------------*/

adata_obj  ADNNDataClone(adata_obj original,	// original data object
			 bool no_strides)	// the same data format, layout and dimensions, but not strides
{

  adata_obj ret = 0;
  int status = 0;

  adnn_data_parameters data;

  status = ADNNDataInspect(original, &data);
  if (status == ADNN_SUCCESS)
    {
      adnn::aDNNTensor * tens = (adnn::aDNNTensor*)original;
      
      adnn::ADNNLib * lib = (adnn::ADNNLib *)tens->getParent();
      
      for (int i = 0; no_strides && i < ADNN_MAX_TENSOR_DIM; ++i)
	{
	  data.strides[i] = 0;
	}
      
      ret = lib->createTensor(data);
    }
  return(ret);
  }


int ADNNDataDestroy(adata_obj * data)
{
  int ret = 0;

  adnn::aDNNTensor * tens = (adnn::aDNNTensor*)*data;
  
  ret = tens->release();

  return(ret);
}

int ADNNDataAllocate(adata_obj data, int alloc_flags)
{
  int ret = 0;
  adnn::aDNNTensor * tens = (adnn::aDNNTensor*)data;

  ret = tens->allocTensor(alloc_flags);
  return(ret);
}

int ADNNDataRetain(adata_obj data)
{
  int ret = 0;
  return(ret);
}

int ADNNDataRelease(adata_obj data)
{
  int ret = 0;
  return(ret);
}

int ADNNDataAccess(adata_obj data,
		   cl_command_queue non_default_queue,	// if set the queu has been used to transfer data from/to GPU memory, ...
		   int access_flags,			// ... if 0, the defualt queue is going to be used
		   adnn_data_parameters * data_params)
{
  int ret = 0;

  adnn::aDNNTensor * tens = (adnn::aDNNTensor*)data;
  void * mapped = tens->accessTensor(access_flags, non_default_queue);
  assert(mapped);
  ret = ADNNDataInspect(data, data_params);
  return(ret);
}

int ADNNDataInspect(adata_obj data, adnn_data_parameters * data_params)
{
  int ret = 0;
  adnn::aDNNTensor * tens = (adnn::aDNNTensor*)data;
  assert(data_params);
  memset(data_params, 0, sizeof(adnn_data_parameters));
  ret = tens->getParams(*data_params);
  return(ret);
}

int ADNNDataCommit(adata_obj data)
{
  int ret = 0;
  adnn::aDNNTensor * tens = (adnn::aDNNTensor*)data;
  tens->commitTensor();
  return(ret);
}

int ADNNDataInit(adata_obj data, const adnn_data_init_parameters * data_init)
{
  int ret = 0;
  adnn::aDNNTensor * tens = (adnn::aDNNTensor*)data;
  
  ret = tens->initTensor(*data_init);
  return(ret);
}
//END



