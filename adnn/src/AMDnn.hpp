// File:AMDNN.hpp
// Path:repo/stt/apcLibs/aDNN/aLibDNN/
// 160311: Kernel Path fiex 
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


#ifndef ALIB_DNN_HPP_
#define ALIB_DNN_HPP_

typedef float aDType;

namespace adnn
{

  double CalculateErr(aDType c_val, aDType g_val);

  class ADNN;
  class aDNNode;
  //class ADNN;
  class aDNNodeConv;
  class aDNNTensor;

 class adnn_filter1D
   {
    public:

     adnn_filter1D()
     {
       size = 0;
       pad = 0;
       stride = 0;
     }

     adnn_filter1D(const adnn_filter1D &rh)
     {
       *this = rh;   // ?? We want a pointer to 'other' ??  (also below) -- If we are doing this then maybe should have a reference count in rh that gets incremented  so we don't delete it
     }

     adnn_filter1D(const adnn_filter1D_parameters& c_descr)
     {
       size = c_descr.size;
       pad = c_descr.pad;
       stride = c_descr.stride;
     }

     const adnn_filter1D &operator= (const adnn_filter1D &rh)
     {
       size = rh.size;
       pad = rh.pad;
       stride = rh.stride;
       return *this;
     }

     // DO WE WANT TO protected: OR private: these 
     int size;
     short pad;
     short stride;
 };

 // filyter descriptor
 class adnn_filter
 {
 public:
   adnn_filter()
   {
     non_sharedBiases = false; // shared by default
   }

   adnn_filter(const adnn_filter_parameters & c_descr)
   {
     non_sharedBiases = c_descr.non_sharedBiases; // shared by default
     correlation = c_descr.correlation;
     for (int i = 0; i < c_descr.n_dims; i++)
       {
	 filter.push_back(adnn_filter1D(c_descr.filter[i]));
       }
   }
   
   adnn_filter(const adnn_filter & rh)
   {
     *this = rh;
   }

   const adnn_filter & operator = (const adnn_filter & rh)
   {
     non_sharedBiases = rh.non_sharedBiases; // shared by default
     correlation = rh.correlation;
     for (int i = 0; i < rh.filter.size(); i++)
       {
	 filter.push_back(rh.filter[i]);
       }
     return *this;
   }

   // protected OR private ???
   
   std::vector<adnn_filter1D> filter;
   bool non_sharedBiases;	// if false biases are shared
   bool correlation;            // if flase - convolution,  if true - correlation
 };

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  class adnn_data_init
  {
  public:

    adnn_data_init()  // : mean(1.0)  <<<--- is this what you want for 'const' 
    {
      init_distr = ADNN_WD_CONSTANT;
      mean = 1.; // const
      std = 0;
    }

    adnn_data_init(const adnn_data_init_parameters & c_descr)
    {
      init_distr = c_descr.init_distr;
      mean = c_descr.mean;
      std = c_descr.std;
    }

    // protect ??
    ADNN_DATA_INIT_DISTR init_distr;
    double mean;
    double std;
  };



  /************************************************************************************************************************
   **
   **			ADNNBase Class
   **
   ************************************************************************************************************************/

  class ADNNBase
  {
  public:

    int retain(void)
    {
      int ret = ref_count_++;
      return(ret);
    }

    int release(void);

    inline int getRefCount(void) const
    {
      return(ref_count_);
    }

    inline void * getInternal(void) const
    {
      return(internal_);
    }

    inline void setInternal(void* internal)
    {
      destroyInternal();
      internal_ = internal;
    }

    inline void setParent(void * parent)
    {
      parent_ = parent;
    }

    inline const void * getParent(void) const
    {
      return(parent_);
    }

    inline const std::string & getName(void) const
    {
      return(name_);
    }

    inline void setName(const std::string & name)
    {
      name_ = name;
    }

    inline ADNN_NODE_TYPE getType(void) const
    {
      return(type_);
    }

  protected:

    /**
     * Constructors
     */
    ADNNBase();

    /**
     * Destructor
     */
    virtual ~ADNNBase(void);

    // internal implementation        // ?? protected 
    void * parent_;
    void * internal_;
    ADNN_NODE_TYPE type_;
    std::string name_;
    std::string empty_str;
    int destroyInternal(void);
    int removeInternal(void);
    void *getWrapper(void);
    
    int ref_count_;

  };


  /************************************************************************************************************************
   **
   **			aDNNLib Class
   **
   ************************************************************************************************************************/

  //   >>>> WHY THIS - is this your "C" interface <<<<<
  
  class ADNNLib;

  adnn::ADNNLib * ADNNLibCreate(cl_context context = 0,
				cl_device_type accel_type = CL_DEVICE_TYPE_GPU,
				std::string accel_platform = "Advanced Micro Devices, Inc.",
				std::string ocl_kernels_path = "");
  
  int ADNNLibDestroy(adnn::ADNNLib * alib);

  class ADNNLib : public ADNNBase
  {
  public:

    friend adnn::ADNNLib * ADNNLibCreate(cl_context context,
					 cl_device_type accel_type,
					 std::string accel_platform,
					 std::string ocl_kernels_path);
    
    friend int ADNNLibDestroy(adnn::ADNNLib * alib);

    aDNNode* ADNNCreate(const adnn_net_parameters & net_param);
    int ADNNDestroy(ADNN* net);

    inline const void * getOclBackEnd(void) const     {      return(ocl_);    }

    cl_context getContext(void) const;
    
    cl_command_queue getQueue(int device_indx = 0, int queue_indx = 0) const;

    cl_command_queue createQueue(cl_device_id deviceId, const cl_command_queue_properties *prop = 0);

    aDNNTensor * createTensor(const adnn_data_parameters & data_params);

    aDNNode * createNode(const adnn_node_parameters & node_descr);

    int removeNode(aDNNode *node);

    const std::string & getKernelPath()    {  return(kernel_path_);  }

  protected:

    ADNNLib(cl_context context = 0, cl_device_type accel_type = CL_DEVICE_TYPE_GPU, std::string accel_platform = "Advanced Micro Devices, Inc.", std::string ocl_kernels_path = "");

    ~ADNNLib(void);

    void * ocl_;

    std::string kernel_path_;
  };

  /************************************************************************************************************************
   **
   **			aDNNTensor Class
   **
   ************************************************************************************************************************/

//  #define ADNN_DATA_CTL_OWNED				(1<<16)    // data is managed by library

  typedef enum {aDNN_TENSOR_0DIM,
		aDNN_TENSOR_WIDTH,
		aDNN_TENSOR_HEIGHT,
		aDNN_TENSOR_DEPTH,
		aDNN_TENSOR_BATCH,
		aDNN_TENSOR_4THDIM,
		aDNN_TENSOR_5THDIM,
		aDNN_TENSOR_6THDIM,
		aDNN_TENSOR_7THDIM,
		aDNN_TENSOR_8THDIM,
		aDNN_TENSOR_TOTAL,
	       }aDNN_TENSOR_DIM;

  // tensor descriptor
  class aDNNTensor : public ADNNBase
  {
  public:
    aDNNTensor();
    aDNNTensor(const ADNNBase & lib, const adnn_data_parameters & c_descr);
    aDNNTensor(const aDNNTensor & rh);

    ~aDNNTensor();

    const aDNNTensor & operator=(const aDNNTensor & rh);

    inline int getNDim(void) const
    {
      return((int)n_dims_);
    }

    size_t getDim(int dim) const;

    size_t getStride(int dim) const;

    inline size_t getDimOrig(int dim) const    {      assert(dim < dims_.size());      return(dims_[dim]);    }
    inline size_t getStrideOrig(int dim) const    {      assert(dim < strides_.size());      return(strides_[dim]);    }
    inline ADNN_DATA_FORMAT getDataFormat(void) const    {      return(data_format_);    }    
    inline ADNN_DATA_BATCH_FORMAT getBatchFormat(void) const    {      return(batch_format_);    }    
    inline size_t     getSize(void) const    {      return(size_);    }
    inline size_t     getSizeInBytes(void) const    {      return(size_bytes_);    }
    inline cl_context getContext(void) const    {      return(context_);    }
    inline void       setContext(cl_context context)    {      context_ = context;    }
    inline cl_mem     getOCLBuffer(void) const    {      return(ocl_mem_);    }

    int getParams(adnn_data_parameters & c_descr) const;

    virtual int initTensor(const adnn_data_init & data_init, cl_command_queue queue = 0);
    virtual int allocTensor(unsigned int flags = 0, void * data_ptr = 0, cl_context context = 0);
    virtual void * accessTensor(unsigned int flags = 0, cl_command_queue queue = 0);
    virtual void commitTensor(void);
    
    virtual const aDNNTensor & mul2(size_t c_cols, size_t c_rows,
				    aDNNTensor & tA, size_t a_cols, size_t a_rows,
				    aDNNTensor & tB, size_t b_cols, size_t b_rows,
				    int traspose_A = 0, int transpose_B = 0, double alpha = 1, double beta = 0,
				    cl_command_queue queue = 0);
  protected:

    ADNN_DATA_FORMAT data_format_;
    ADNN_DATA_BATCH_FORMAT batch_format_;
    size_t size_;
    size_t size_bytes_;
    void * sys_mem_;
    cl_mem ocl_mem_;
    unsigned int control_bits_;

    std::vector<int> remap_;
    std::vector<size_t> dims_;
    std::vector<size_t> strides_;
    size_t n_dims_;

    int inited_;
    int allocated_;

    void calculate(void);

    cl_context context_;

  };

  /************************************************************************************************************************
   **
   **			aDNNEdge Class
   **
   ************************************************************************************************************************/

  // edge descriptor
  class aDNNEdge : public ADNNBase
  {
  public:
    aDNNEdge(void);
    aDNNEdge(const ADNNBase & lib, const adnn_net_edge_parameters & c_descr);
    aDNNEdge(const aDNNEdge & rh);
   ~aDNNEdge(void);

    const aDNNEdge & operator = (const aDNNEdge & rh);

    inline ADNN_EDGE_DIR_TYPE getEdgeType(void) const
    {
      return(edge_type_);
    }

    inline void setEdgeType(ADNN_EDGE_DIR_TYPE edge_type)
    {
      edge_type_ = edge_type;
    }

    inline std::vector<aDNNode*> ::iterator getConnectedNode(void) const
    {
      return(node_pos_);
    }

    inline  void setConnectedNode(std::vector<aDNNode*> ::iterator node_pos)
    {
      node_pos_ = node_pos;
    }

    inline const aDNNTensor & getData(void) const
    {
      return(*data_);
    }

    inline void  setData(const aDNNTensor * data)
    {
      data_ = (aDNNTensor * )data;
    }

    inline void  setWeightData(const aDNNTensor * data)
    {
      weights_data_ = (aDNNTensor *)data;
    }
    
    inline const aDNNTensor & getWeightsData(void) const
    {
      return(*weights_data_);
    }

    inline void  setBiasData(const aDNNTensor * data)
    {
      bias_data_ = (aDNNTensor *)data;
    }

    inline const aDNNTensor & getBiasData(void) const
    {
      return(*bias_data_);
    }

    inline bool isData(void) const
    {
      return(data_ != NULL);
    }

    inline bool isWeights(void) const
    {
      return(weights_data_ != NULL);
    }

    inline bool isBias(void) const
    {
      return(bias_data_ != NULL);
    }

    inline int getPad(int f_dim = 0) const
    {
      return(filter_params_.filter[f_dim].pad);
    }

    inline int getKernelStride(int f_dim = 0) const
    {
      return(filter_params_.filter[f_dim].stride);
    }

    inline int getKernelSz(int f_dim = 0) const
    {
      return(filter_params_.filter[f_dim].size);
    }

    inline  ADNN_POOLING_METHOD getPoolingMethod(void) const
    {
      return(pooling_method_);
    }

    inline ADNN_LRN_REGION getNormRegion(void) const      // TT: ?? What is a NormRegion 
    {
      return(lrn_parameters_.region);
    }

    inline double getAlpha(void) const   // TT: ??
    {
      return(lrn_parameters_.alpha);
    }

    inline double getBeta(void) const   // TT: ??
    {
      return(lrn_parameters_.beta);
    }

    inline int getAreaSz(void) const // TT: ?? why not getKernelSize() 
    {
      return(lrn_parameters_.kernel_sz);
    }

    inline bool isDataUpdated(void) const
    {
      return(data_updated_);
    }

    inline void setDataUpdated(bool updated)
    {
      data_updated_ = updated;
    }

    inline bool isWeightsUpdated(void) const
    {
      return(weights_updated_); // indicate new data
    }
    
    inline void setWeightsUpdated(bool updated)
    {
      weights_updated_ = updated; // indicate new data
    }

    inline bool isBiasUpdated(void) const
    {
      return(bias_updated_); // indicate new data
    }

    inline void setBiasUpdated(bool updated)
    {
      bias_updated_ = updated; // indicate new data
    }

  protected:

    ADNN_EDGE_DIR_TYPE edge_type_;
    aDNNTensor *data_;
    bool data_updated_; // indicate new data

    std::vector<aDNNode*> ::iterator node_pos_;

    aDNNTensor *weights_data_;
    aDNNTensor *bias_data_;
//  aDNNTensor *weights_diff_;
//  aDNNTensor *bias_diff_;
    bool weights_updated_; // indicate new data
    bool bias_updated_; // indicate new data
    adnn_filter filter_params_;
    ADNN_POOLING_METHOD pooling_method_;
    adnn_lrn_parameters lrn_parameters_;

//  ADNN_LRN_REGION LRNregion_;
//  double alpha_;			// alpha, shift
//  double beta_;			// beta, scale
//  double power_;
   
};


  ///////////////////////////////////////////////////////
  //
  // DNN_OCL_kern_exe
  //
  //////////////////////////////////////////////////////
  class CDNN_OCL_kern_exe;
  typedef std::pair<size_t, void*>       ocl_arg;
  typedef std::map<int, ocl_arg>         ocl_args;
  typedef std::vector<cl_event>          ocl_wait_events;
  typedef std::vector<CDNN_OCL_kern_exe> layer_ocl_exe;


  class CDNN_OCL_kern_exe : public ADNNBase
  {
  public:
    CDNN_OCL_kern_exe();
    CDNN_OCL_kern_exe(ADNNBase * parent,
		      std::string name,
		      std::string file_nm = "",
		      std::string build_options = "",
		      cl_kernel ocl_kern = 0,
		      std::vector<size_t> * glb_sz = NULL,
		      std::vector<size_t> * lcl_sz = NULL,
		      cl_command_queue queue = NULL);
    
    CDNN_OCL_kern_exe(const CDNN_OCL_kern_exe & copy);
    ~CDNN_OCL_kern_exe();

    // TT what makes this 'nowait' - no call to clFinish or clFlush ??
    int ExecuteNoWait(ocl_args * args = NULL,
		      cl_command_queue queue = 0);

    int Construct(const std::string & options = "");

    // additional options has to be exactly the same as in Construct

    int Build(const ocl_args & args, int q_indx = 0, const std::string & additional_options = "");

    inline const std::string & getKernFileNm(void) const
    {
      return(kern_src_file_);
    }

    inline const std::string & getKernNm(void) const
    {
      return(kern_nm_);
    }

    inline const std::string & getKernSrcSring(void) const
    {
      return(kern_src_string_);
    }

    inline const std::string & getKernBuildOptions(void) const
    {
      return(kern_build_options_);
    }

    inline void setKernBuildOptions(const std::string & build_optins)
    {
      kern_build_options_ = build_optins;
    }

    inline cl_kernel getOclKern(void) const
    {
      return(kernel_);
    }

    inline void setOclKern(cl_kernel kernel)
    {
      kernel_ = kernel;
    }


		inline const std::vector<size_t> & getLclSize(void) const
		{
			return(lcl_sz_);
		}

		inline void setLclSize(const std::vector<size_t> & lcl_sz)
		{
			lcl_sz_ = lcl_sz;
		}


		inline const std::vector<size_t> & getGblSize(void) const
		{
			return(glb_sz_);
		}

		inline void setGblSize(const std::vector<size_t> & gbl_sz)
		{
			glb_sz_ = gbl_sz;
		}


		inline cl_command_queue getOclQueue(void) const
		{
			return(queue_);
		}

		inline void setOclQueue(cl_command_queue queue)
		{
			queue_ = queue;
		}


		inline const ocl_wait_events & getWaitEvents(void) const
		{
			return(wait_events_);
		}

		inline cl_event getComplEvent(void) const
		{
			return(completion_event_);
		}

		void retrieveExeParams(adnn_node_exe_parameters & params);

  protected:
    std::string kern_src_file_;
    std::string kern_nm_;
    std::string kern_src_string_;
    std::string kern_build_options_;
    cl_kernel kernel_;
    std::vector<size_t> lcl_sz_;       // TT: ?? what is lcl
    std::vector<size_t> glb_sz_;       // TT: ?? global what? queue, memory
    cl_command_queue queue_;
    cl_event completion_event_;
    ocl_wait_events wait_events_;
  };



  /************************************************************************************************************************
   **
   **			aDNNode Class
   **
   ************************************************************************************************************************/

#define ADNN_INPUT_NM		".input"
#define ADNN_OUTPUT_NM		".output"
#define ADNN_WEIGHTS_NM		".weights"
#define ADNN_BIAS_NM		".bias"
#define ADNN_DIFF_NM		".df"
#define ADNN_HISTORY_NM		".hist"
#define ADNN_VERIFY_NM		".vf"
#define ADNN_MAXINDX_NM		".max_indxs"
#define ADNN_SCALE_NM		".scale"
#define ADNN_TRANSPOSE_NM	".transpose"
#define ADNN_SUM_NM		".sum"
#define ADNN_WIN_TRANSFORM_NM	".wintrasnform"   // transformed input
#define ADNN_WIN_COLS_FWD_NM	".cols_fwd"       // transformed input
#define ADNN_WIN_TWEIGHTS_NM	ADNN_WIN_TRANSFORM_NM // transformed weights
#define ADNN_WIN_INPUT_NM	ADNN_WIN_TRANSFORM_NM   // transformed input
#define ADNN_WIN_EWM_NM		".ewm"      // element-wise multiplication
#define ADNN_WIN_CTLMAP_NM	".ctlmap"      // control map name

  class aDNNode : public ADNNBase
  {
  public:
    friend class ADNNLib;
    friend class ADNN;
    const aDNNode & operator = (const aDNNode & rh);

    virtual int Connect(void);
    virtual int Construct(void);
    virtual int ConstructBwd(void);
    virtual int Build(void);
    virtual int BuildBwd(void);
    virtual int Run(void);
    virtual int RunFwd(const adnn_node_parameters * running_params = NULL);
    virtual int RunBwd(const adnn_node_parameters * running_params = NULL);
    virtual int RunHostFwd(void);
    virtual int RunHostBwd(void);
    virtual int UpdateWeights(void);
    virtual int UpdateWeightsHost(void);

    const ADNN & getNet(void) const
    {
      return(*my_net_);
    }

    const void setNet(ADNN * net)
    {
      my_net_ = net;
    }

    const ADNN * isNet(void) const
    {
      return(my_net_);
    }

// get bottom edge / forward data
    inline const aDNNTensor & getBotFwd(int edge_indx = 0) const
    {
      return(inputs_[edge_indx].getData());
    }

    inline const aDNNTensor & getTopFwd(int edge_indx = 0) const
    {
      return(outputs_[edge_indx].getData());
    }

    inline const aDNNTensor & getBotWeightsFwd(int edge_indx = 0) const
    {
      return(inputs_[edge_indx].getWeightsData());
    }

    inline const aDNNTensor & getBotBiasFwd(int edge_indx = 0) const
    {
      return(inputs_[edge_indx].getBiasData());
    }

    inline bool isTopData(int edge_indx = 0) const
    {
      return(!outputs_.empty() && outputs_[edge_indx].isData());
    }

    inline bool isBotData(int edge_indx = 0) const
    {
      return(!inputs_.empty() && inputs_[edge_indx].isData());
    }

    inline bool isWeights(int edge_indx = 0) const
    {
      return(!inputs_.empty() && inputs_[edge_indx].isWeights());
    }

    inline bool isBias(int edge_indx = 0) const
    {
      return(!inputs_.empty() && inputs_[edge_indx].isBias());
    }

    inline const std::string getBotNm(int edge_indx = 0)
    {
      std::string slot_nm = name_ + getInputEdgeName(edge_indx);
      return(slot_nm);
    }

    inline const std::string getTopNm(int edge_indx = 0)
    {
      std::string slot_nm = name_ + getOutputEdgeName(edge_indx);
      return(slot_nm);
    }

    inline const std::string getBotDiffNm(int edge_indx = 0)
    {
      std::string slot_nm = getBotNm(edge_indx) + ADNN_DIFF_NM;
      return(slot_nm);
    }

    inline const std::string getTopDiffNm(int edge_indx = 0)
    {
      std::string slot_nm = getTopNm(edge_indx) + ADNN_DIFF_NM;
      return(slot_nm);
    }

    inline const std::string getWeightsNm(int edge_indx = 0)
    {
      std::string slot_nm = name_ + getInputEdgeName(edge_indx)  + ADNN_WEIGHTS_NM;
      return(slot_nm);
    }

    inline const std::string getBiasNm(int edge_indx = 0)
    {
      std::string slot_nm = name_ + getInputEdgeName(edge_indx) + ADNN_BIAS_NM;
      return(slot_nm);
    }

    inline const std::string getWeightsDiffNm(int edge_indx = 0)
    {
      std::string slot_nm = getWeightsNm(edge_indx) + ADNN_DIFF_NM;
      return(slot_nm);
    }

    inline const std:: string getBiasDiffNm(int edge_indx = 0)
    {
      std::string slot_nm = getBiasNm(edge_indx) + ADNN_DIFF_NM;
      return(slot_nm);
    }

    inline const aDNNTensor & getBotDiff(int edge_indx = 0)
    {
      std::string slot_nm = getBotDiffNm(edge_indx);
      return((const aDNNTensor & )getSlot(slot_nm));
    }

    inline const aDNNTensor & getTopDiff(int edge_indx = 0)
    {
      std::string slot_nm = getTopDiffNm(edge_indx);
      return((const aDNNTensor &)getSlot(slot_nm));
    }

    inline const aDNNTensor & getWeightsDiff(int edge_indx = 0)
    {
      std::string slot_nm = getWeightsDiffNm(edge_indx);
      return((const aDNNTensor &)getSlot(slot_nm));
    }

    inline const aDNNTensor & getBiasDiff(int edge_indx = 0)
    {
      std::string slot_nm = getBiasDiffNm(edge_indx);
      return((const aDNNTensor &)getSlot(slot_nm));
    }


// edges
    int getNInternalInputEdges(void) const;
    int getNInternalOutputEdges(void) const;

    inline bool isSource(void) const
    {
      return(getNInternalInputEdges() == 0);
    }

    inline bool isSink(void) const
    {
      return(getNInternalOutputEdges() == 0);
    }

    std::vector<aDNNEdge>::iterator getInputByName(const std::string & name);

    void setInputNode(std::vector<aDNNode*> ::iterator input_node, int edge_indx = 0);

    void setOutputNode(std::vector<aDNNode*> ::iterator output_node, int edge_indx = 0);

    inline void setOutputEdgeType(ADNN_EDGE_DIR_TYPE edge_type, int edge_indx = 0)
    {
      if (outputs_.size() <= edge_indx)
	{
	  outputs_.resize(edge_indx + 1);
	}
      outputs_[edge_indx].setEdgeType(edge_type);
    }

    inline ADNN_EDGE_DIR_TYPE getOutputEdgeType(int edge_indx = 0) const
    {
      ADNN_EDGE_DIR_TYPE ret = ADNN_ED_INTERNAL;
      if (!outputs_.empty() && edge_indx < outputs_.size())
	{
	  ret = outputs_[edge_indx].getEdgeType();
	}
      return(ret);
    }

    inline void setOutputEdgeName(std::string name, int edge_indx = 0)
    {
      if (outputs_.size() <= edge_indx)
	{
	  outputs_.resize(edge_indx + 1);
	}
      outputs_[edge_indx].setName(name);
    }

    inline const std::string getInputEdgeName(int edge_indx = 0) const
    {
//ORG return((!inputs_.empty() && edge_indx < inputs_.size() && !inputs_[edge_indx].getName().empty()) ? null_inp_str_ + std::to_string((long long)edge_indx) + "." + inputs_[edge_indx].getName() : null_inp_str_ + std::to_string((long long)edge_indx));
      return((!inputs_.empty() && edge_indx < inputs_.size() && !inputs_[edge_indx].getName().empty()) ? null_inp_str_                                        + "." + inputs_[edge_indx].getName() : null_inp_str_                                       );
    }

    inline const std::string getOutputEdgeName(int edge_indx = 0) const
    {
//ORG return((!outputs_.empty() && edge_indx < outputs_.size() && !outputs_[edge_indx].getName().empty()) ? null_out_str_ + std::to_string((long long)edge_indx) + "." + outputs_[edge_indx].getName() : null_out_str_ + std::to_string((long long)edge_indx));
      return((!outputs_.empty() && edge_indx < outputs_.size() && !outputs_[edge_indx].getName().empty()) ? null_out_str_                                        + "." + outputs_[edge_indx].getName() : null_out_str_                                       );
    }

    inline std::vector<aDNNode*> ::iterator getInputNode(int edge_indx = 0) const
    {
      return(inputs_[edge_indx].getConnectedNode());
    }

    inline std::vector<aDNNode*> ::iterator getOutputNode(int edge_indx = 0) const
    {
      return(outputs_[edge_indx].getConnectedNode());
    }

    inline aDNNEdge & getInputEdge(int edge_indx = 0)
    {
      return(inputs_[edge_indx]);
    }

    inline aDNNEdge & getOutputEdge(int edge_indx = 0)
    {
      return(outputs_[edge_indx]);
    }

    inline int getNOutputFeatureMaps(void)
    {
      int n_feature_maps = 0;
      n_feature_maps = (int)getOutputEdge().getData().getDim(aDNN_TENSOR_DEPTH);
      return(n_feature_maps);
    }

    inline std::vector<aDNNEdge> & getInputEdges(void)
    {
      return(inputs_);
    }

    inline std::vector<aDNNEdge> & getOutputEdges(void)
    {
      return(outputs_);
    }

    // list of tesnors allocated by this node
    inline std::map<aDNNTensor *, std::string> & getOwnedTensors(void)
    {
      return(owned_tensors_);
    }

    // list of tensors referred by this node
    inline std::map<std::string, aDNNTensor *> & getUsedTensors(void)
    {
      return(used_tensors_);
    }

// slot data base oprations
    aDNNTensor & createSlot(const std::string & name, const adnn_data_parameters & c_descr, bool reference = false);
    aDNNTensor & addSlot(const std::string & name, const aDNNTensor & data, bool reference = false);
    aDNNTensor & cloneSlot(const std::string & name, const aDNNTensor & data, bool reference = false);
    aDNNTensor & getSlot(const std::string & name);
    bool isSlotEmpty(const std::string & name);
    inline size_t getNSlots(void) const
    {
      return(used_tensors_.size());
    }

    int getSlotsNames(const char** nm_list) const;

// build
    inline const std::string & getGenericCompOptions(void) const
    {
      return(generic_comp_otions_);
    }

// control
    bool isPerLayerTiming(void) const;

    //  per layer iterations for timing
    inline int getNTimingIter(void) const
    {
      int iter = (isPerLayerTiming()) ? per_layer_iter_ : 1;
      return(iter);
    }

    inline bool isPerLayerMessaging(void) const
    {
      return(per_layer_messages_);
    }
    
    inline int getDebugLevel(void) const
    {
      return(debug_level_);
    }

    inline layer_ocl_exe & getForwardExeStages(void)
    {
      return(ocl_fwd_execs_);
    }

    inline layer_ocl_exe & getBackwarExeStages(void)
    {
      return(ocl_bwd_execs_);
    }

    const void * getBaseOcl(void) const;

// parameters
    inline void getNeuronArgs(aDType &power, aDType &scale, aDType &shift) const
    {
      power = (aDType)neuron_params_.power;
      shift = (aDType)neuron_params_.alpha;
      scale = (aDType)neuron_params_.beta;
    }

    inline ADNN_NEURON_TYPE getNeuronType(void) const
    {
      return(neuron_params_.type);
    }

    inline int getPad(int edge_indx = 0, int dim = 0)
    {
      return(getInputEdge(edge_indx).getPad(dim));
    }

    inline int getKernelStride(int edge_indx = 0, int dim = 0)
    {
      return(getInputEdge(edge_indx).getKernelStride(dim));
    }

    inline int getKernelSz(int edge_indx = 0, int dim = 0)
    {
      return(getInputEdge(edge_indx).getKernelSz(dim));
    }

    inline ADNN_POOLING_METHOD getPoolingMethod(int edge_indx = 0)
    {
      return(getInputEdge(edge_indx).getPoolingMethod());
    }

    inline 	ADNN_LRN_REGION getNormRegion(int edge_indx = 0)
    {
      return(getInputEdge(edge_indx).getNormRegion());
    }

    inline int getLocalArea(int edge_indx = 0)
    {
      return(getInputEdge(edge_indx).getAreaSz());
    }

    inline double getAlpha(int edge_indx = 0)
    {
      return(getInputEdge(edge_indx).getAlpha());
    }

    inline double getBeta(int edge_indx = 0)
    {
      return(getInputEdge(edge_indx).getBeta());
    }
    
// update

    inline const adnn_lr_policy_params & getLearningParams(bool weights) const
    {
      return((weights) ? update_params_.weights_lr : update_params_.bias_lr);
    }

    inline double getMomentum(bool weights) const
    {
      return((weights) ? update_params_.weights_momentum : update_params_.bias_momentum);
    }

    inline 	double getDecay(bool weights) const
    {
      return((weights) ? update_params_.weights_decay : update_params_.bias_decay);
    }

    inline  ADNN_LEARNINGPOLICY get_lr_policy(bool weights) const
    {
      return((weights) ? update_params_.weights_lr.policy : update_params_.bias_lr.policy);
    }

    inline double get_stepsize(bool weights) const
    {
      return((weights) ? update_params_.weights_lr.step : update_params_.bias_lr.step);
    }

    inline double get_gamma(bool weights) const
    {
      return((weights) ? update_params_.weights_lr.gamma : update_params_.bias_lr.gamma);
    }

    inline double get_power(bool weights) const
    {
      return((weights) ? update_params_.weights_lr.power : update_params_.bias_lr.power);
    }

    inline size_t getInternalCounter(void) const
    {
      return(iter_counter_);
    }

    inline size_t updateInternalCounter(size_t step)
    {
      size_t ret = iter_counter_;
      iter_counter_ += step;
      return(ret);
    }

    inline size_t setInternalCounter(size_t val)
    {
      size_t ret = iter_counter_;
      iter_counter_ = val;
      return(ret);
    }

  protected:

    std::vector<aDNNEdge> inputs_;      // inputs (on-direction) edges (data set); its name is the name the up-stream node
    std::vector<aDNNEdge> outputs_;      // node output edge, the name is nodes name if a single output
    adnn_neuron_parameters neuron_params_;  // ?? how neurons are made, type .... ??
    adnn_update_params update_params_; //  ?? how updates are done 

    // string id has the form
    // "name.weights"
    // "name.df"
    // "name.weights.df"
    // "name.weights.history"
    // and can have postfix ".vr" for verification

    int ConstructInput(cl_command_queue queue = 0);
    int ConstructOutput(cl_command_queue queue = 0);
    int InitializeInputWeights(void);
    int ConstructInputWeights(cl_command_queue queue = 0);
    int ConstructWeightsBwd(void);
    int BuildWeightsBwd(void);
    int UpdateWeightsInternal(void);
    virtual int ConstructOptions(void);
    virtual int VerifyFwd(void);
    virtual int VerifyBwd(void);
    virtual int VerifyUpdateWeights(void);


    // list of tesnors allocated by this node
    std::map<aDNNTensor *, std::string> owned_tensors_;

    // list of tensors referred by this node
    std::map<std::string, aDNNTensor *> used_tensors_;

    ADNN * my_net_;


    // build & execute
    // gereric compilation options
    std::string generic_comp_otions_;

    // exec ocl kernels
    layer_ocl_exe ocl_fwd_execs_;
    layer_ocl_exe ocl_bwd_execs_;
    layer_ocl_exe ocl_update_execs_;

    // control and  timing
    // timing per layer
    bool per_layer_timing_;

    //  per layer iterations for timing
    int  per_layer_iter_;
    bool per_layer_messages_;
    int  debug_level_;
    void * monitor_;
    double processing_time_;
    size_t iter_counter_;

    std:: string null_inp_str_;
    std:: string null_out_str_;

  protected:

    aDNNode(const ADNNBase & lib, const adnn_node_parameters & node_params);
//  aDNNode(const ADNNBase & lib, const adnn_node_parameters & node_params);
    ~aDNNode(void);
    aDNNode();
    aDNNode(const aDNNode & rh);

    int init(const adnn_node_parameters & c_descr);
    int update(const adnn_node_parameters & c_descr);
    int calculateUpdateRates(aDType & local_rate, aDType & local_decay, bool weights = true);
  };

  /************************************************************************************************************************
   **
   **			ADNN network Class
   **
   ************************************************************************************************************************/

 class ADNN : public aDNNode
 {
 public:

   friend class ADNNLib;

   const aDNNode & operator = (const ADNN & rh);

   aDNNode * AddNode(const adnn_node_parameters & node_descr);
   int AddNodes(int n_nodes, aDNNode ** nodes);
   int RemoveNode(aDNNode * node);

   int Connect(void);
   int Construct(void);
   int ConstructBwd(void);
   int Build(void);
   int BuildBwd(void);
   int Run(bool forward = true, int n_running_params = 0, const adnn_node_parameters * running_params = NULL);
   int RunFwd(int n_running_params = 0, const adnn_node_parameters * running_params = NULL);
   int RunBwd(int n_running_params = 0, const adnn_node_parameters * running_params = NULL);
   int UpdateWeights(void);

   int getNInternalInputEdges(void) const;
   
 protected:

   std::vector<aDNNode*> ::iterator findNode(aDNNode * node);

   ADNN(const aDNNode & net, const adnn_node_parameters & node_params);
   // root - net - creation
   ADNN(const ADNNLib & lib, const adnn_node_parameters & node_params);
   ADNN();
   ADNN(const ADNN & rh);

   ~ADNN(void);

 protected:
   std::vector<aDNNode *> net_;
   std::vector<aDNNode *> net_owned_;
 };


  /************************************************************************************************************************
   **
   **			ADNNode Convulution Class
   **
   ************************************************************************************************************************/

 class aDNNodeConv : public aDNNode
 {

 public:

   friend class ADNN;       // TT: do we want to break inecapsulation here? 
   friend class ADNNLib;

   const aDNNode & operator = (const aDNNodeConv & rh);
   
   int Connect(void);
   int Construct(void);
   int ConstructBwd(void);
   int Build(void);
   int BuildBwd(void);
   int Run(void);
   int RunFwd(const adnn_node_parameters * running_params = NULL);
   int RunBwd(const adnn_node_parameters * running_params = NULL);
   int RunHostFwd(void);
   int RunHostBwd(void);
   int UpdateWeights(void);

 protected:

   aDNNodeConv(const ADNNBase & lib, const adnn_node_parameters & node_params);
   ~aDNNodeConv(void);
   aDNNodeConv(void);
   aDNNodeConv(const aDNNodeConv & rh);

   //	int ConstructOptions(void);

   int VerifyFwd(void);
   int VerifyBwd(void);

 private:

   bool old_;

   // generic alg
   int ConstructGen_NCHW(void);
   int BuildGen(void);
   int RunHostGenFwd(void);

   // direct algorithm
   int in_main_loop_;

   int ConstructGT32_NCHW(void);
   int ConstructLE32_NCHW(void);
   int Construct_NCHW_N3(void);

   // winograd algorithm
   bool win_alg_;
   int RunHostFwdWin(void);
   int ConstructFwdWin_NCHW(void);
   int BuildFwdWin_NCHW(void);
   int RunFwdWin_NCHW(const adnn_node_parameters * running_params);
   int VerifyFwdWin(void);
   int tile_sz0_;  // conv local tile size in win alg
   int tile_sz1_;
   void * input_transform_ctlmap_; // control map
   void * inverse_transform_ctlmap_; // control map
   bool win2x2_;
 };


  class aDNNodeNeuron : public aDNNode
  {

  public:
    friend class ADNN;
    friend class ADNNLib;

    const aDNNode & operator = (const aDNNodeNeuron & rh);

    int Connect(void);
    int Construct(void);
    int ConstructBwd(void);
    int Build(void);
    int BuildBwd(void);
    int Run(void);
    int RunFwd(const adnn_node_parameters * running_params = NULL);
    int RunBwd(const adnn_node_parameters * running_params = NULL);
    int RunHostFwd(void);
    int RunHostBwd(void);

  protected:

    aDNNodeNeuron(const ADNNBase & lib, const adnn_node_parameters & node_params);
    ~aDNNodeNeuron(void);
    aDNNodeNeuron(void);
    aDNNodeNeuron(const aDNNodeNeuron & rh);
   //   ~aDNNodeConv(void);

   int ConstructOptions(void);

   int VerifyFwd(void);
   int VerifyBwd(void);

 private:

   bool old_;

   // generic alg
   int ConstructGen_NCHW(void);
   int BuildGen(void);
   int RunHostGenFwd(void);

   // direct algorithm
   int in_main_loop_;

   int ConstructGT32_NCHW(void);
   int ConstructLE32_NCHW(void);
   int Construct_NCHW_N3(void);

   // winograd algorithm
   bool win_alg_;
   int RunHostFwdWin(void);
   int ConstructFwdWin_NCHW(void);
   int BuildFwdWin_NCHW(void);
   int RunFwdWin_NCHW(const adnn_node_parameters * running_params);
   int VerifyFwdWin(void);
   int tile_sz0_;  // conv local tile size in win alg
   int tile_sz1_;
   void * input_transform_ctlmap_; // control map
   void * inverse_transform_ctlmap_; // control map
   bool win2x2_;

 };


  class aDNNodePooling : public aDNNode
  {

  public:
    friend class ADNN;
    friend class ADNNLib;

    const aDNNode & operator = (const aDNNodePooling & rh);
    int Connect(void);
    int Construct(void);
    int ConstructBwd(void);
    int Build(void);
    int BuildBwd(void);
    int Run(void);
    int RunFwd(const adnn_node_parameters * running_params = NULL);
    int RunBwd(const adnn_node_parameters * running_params = NULL);
    int RunHostFwd(void);
    int RunHostBwd(void);

  protected:

    aDNNodePooling(const ADNNBase & lib, const adnn_node_parameters & node_params);
    aDNNodePooling(void);
    aDNNodePooling(const aDNNodePooling & rh);
    ~aDNNodePooling(void);

    int ConstructOptions(void);

    int VerifyFwd(void);
    int VerifyBwd(void);

  };


  class aDNNodeLRN : public aDNNode
  {

  public:
    friend class ADNN;
    friend class ADNNLib;

    const aDNNode & operator = (const aDNNodeLRN & rh);
    int Connect(void);
    int Construct(void);
    int ConstructBwd(void);
    int Build(void);
    int BuildBwd(void);
    int Run(void);
    int RunFwd(const adnn_node_parameters * running_params = NULL);
    int RunBwd(const adnn_node_parameters * running_params = NULL);
    int RunHostFwd(void);
    int RunHostBwd(void);

  protected:

    aDNNodeLRN(const ADNNBase & lib, const adnn_node_parameters & node_params);
    ~aDNNodeLRN(void);
    aDNNodeLRN(void);
    aDNNodeLRN(const aDNNodeLRN & rh);

    int ConstructOptions(void);

    int VerifyFwd(void);
    int VerifyBwd(void);
  };


  class aDNNodeFullyConnect : public aDNNode
  {
  public:
    friend class ADNN;
    friend class ADNNLib;

    const aDNNode & operator = (const aDNNodeFullyConnect & rh);
    int Connect(void);
    int Construct(void);
    int ConstructBwd(void);
    int Build(void);
    int BuildBwd(void);
    int Run(void);
    int RunFwd(const adnn_node_parameters * running_params = NULL);
    int RunBwd(const adnn_node_parameters * running_params = NULL);
    int RunHostFwd(void);
    int RunHostBwd(void);
    int UpdateWeights(void);

  protected:

    aDNNodeFullyConnect(const ADNNBase & lib, const adnn_node_parameters & node_params);
    ~aDNNodeFullyConnect(void);
    aDNNodeFullyConnect(void);
    aDNNodeFullyConnect(const aDNNodeFullyConnect & rh);

    int ConstructOptions(void);

    int VerifyFwd(void);
    int VerifyBwd(void);
  };


  class aDNNodeSoftMax : public aDNNode
  {
  public:
    friend class ADNN;
    friend class ADNNLib;
    
    const aDNNode & operator = (const aDNNodeSoftMax & rh);
    int Connect(void);
    int Construct(void);
    int ConstructBwd(void);
    int Build(void);
    int Run(void);
    int BuildBwd(void);
    int RunFwd(const adnn_node_parameters * running_params = NULL);
    int RunBwd(const adnn_node_parameters * running_params = NULL);
    int RunHostFwd(void);

    inline bool isCrossEntrypyLoss(void) const
    {
      return(with_crossentropy_loos_);
    }

    inline void setCrossEntrypyLoss(bool crossentropy_loss)
    {
      with_crossentropy_loos_ = crossentropy_loss;
    }

#if 0
// if the loss layer make the result a bottom differencial to move it up the stream
    const aDNNTensor & getTopFwd(int edge_indx = 0)
    {
      if (isCrossEntrypyLoss())
	{
	  return(getBotDiff(edge_indx));
	}
      else
	{
	  return(aDNNode::getTopFwd(edge_indx));
	}
    }

#endif
  protected:
    
    aDNNodeSoftMax(const ADNNBase & lib, const adnn_node_parameters & node_params);
    ~aDNNodeSoftMax(void);

    aDNNodeSoftMax(void);
    aDNNodeSoftMax(const aDNNodeSoftMax & rh);

    int ConstructOptions(void);
    int VerifyFwd(void);

    bool with_crossentropy_loos_;
    int BuildIntnl(void);	// ??? 
  };
}


//adnn::ADNN * CreateADNNObj(cl_context context = 0, cl_device_type accel_type = CL_DEVICE_TYPE_GPU, std::string accel_platform = "Advanced Micro Devices, Inc.", std::string ocl_kernels_path = "");
//int DestroyADNNObj(adnn::ADNN * alib);


#endif
