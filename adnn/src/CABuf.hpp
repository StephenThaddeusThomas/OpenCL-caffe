/*---------------------------------------------------------
CABuf
----------------------------------------------------------*/

namespace adnn
{

#define _CBUF_MEM_SYS_ONLY  ADNN_MEM_ALLOCSYS_ONLY
#define _CBUF_MEM_OCL_ONLY  ADNN_MEM_ALLOCOCL_ONLY
  
  template<typename T>
  class CABuf
  {
  public:

    CABuf(cl_context _context);
    virtual ~CABuf(void);

    /// TODO : correct copy costractor, operator =, clone()
    CABuf(const CABuf & _src);

    /** TODO CABuf clone(void) **/
  
    int create(size_t _sz, uint _flags = 0, const T *_buf = 0);
    int create(size_t _sz, uint _flags = 0);
  
    int attach(const T *_buf, size_t _sz); // TODO : CORRECT 
    int attach(cl_mem _buf, size_t _sz);	 // TODO : CORRECT 
  
    T*  map(uint _flags, cl_command_queue _mappingQ = NULL);
    T*  mapA(cl_command_queue _mappingQ, uint _flags, cl_event *_wait_event = NULL, cl_event *_set_event = NULL );
    int unmap(cl_event *_wait_event = NULL, cl_event *_set_event = NULL);
  
    int copyToDevice(uint _flags = 0, const T* _data = NULL, size_t _len = 0, cl_command_queue _commandQueue= 0);
    int copyToDeviceA(uint _flags, const T* _data = NULL, size_t _len = -1, cl_command_queue _commandQueue = 0);
    int copyToHost(cl_command_queue _commandQueue = 0);
    int copy(CABuf<T> & _src, size_t _src_offset = 0, size_t _dst_offset = 0, cl_command_queue _commandQueue = 0);

    int setValue(T _val, cl_command_queue _commandQueue = 0);
    // ?? What is difference - where are the comments
    int setValue2(T _val, cl_command_queue _commandQueue = 0);

    int release(void);
    int set_value(T val);

    inline void setContext(cl_context _context)   { context_ = _context;  }
    inline cl_context getContext(void)            { return(context_);     }
    inline const cl_mem & getCLMem(void) 	        { return(buf_);	        }
    inline T * & getSysMem(void)	                { return(sys_ptr_);	}
    inline T * & getMappedMem(void)               { return(map_ptr_);	}
    inline void setSysOwnership(bool own )	{ sys_own_ = own;	}
    inline bool getSysOwnership(void)             { return(sys_own_);	}
    
    // DANGEROUS  ?? WHY
    inline void setLen(size_t _len)	        { len_ = _len;	        }

    inline size_t getLen(void)	                { return(len_);	        }
    inline uint getFlags(void)	                { return(flags_);	}
    
  protected:
  
    cl_context context_;
    cl_command_queue mappingQ_;
    T * sys_ptr_;
    T * map_ptr_;
    cl_mem buf_;
    size_t len_;
    uint flags_;
    bool sys_own_;
    bool cl_own_;
    size_t offset_;
  };


  template<typename T>
  class CASubBuf : public CABuf<T>
  {
  public:

    CASubBuf(CABuf<T> & _base);

    int create(size_t _offset, size_t _sz, uint _flags);
    int create(size_t _sz, uint _flags);
  
  protected:
  
    cl_mem base_buf_;
  };

}; // namespace adnn

