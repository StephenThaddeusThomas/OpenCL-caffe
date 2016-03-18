// File:aDNNOCL.hpp
// Date:160221
// TASK:move all non-trivial method bodies to the .cpp file
// then break up the files so that we have ONE class per-file 

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

#ifndef ADNN_OCL_H_
#define ADDN_OCL_H_

#include "AMDnn.h"
#include "aLibsCLUtil.hpp"

namespace adnn
{
  class CaLibsOCL;

  // TT: "C" interface, <Singleton> Pattern
  CaLibsOCL & getaDNNOCL(void);
  CaLibsOCL * createaDNNOCL(void);

  class CaLibsOCL
  {
//    streamsdk::SDKDeviceInfo deviceInfo;            /**< Structure to store device information*/
//    streamsdk::KernelWorkGroupInfo kernelInfo;      /**< Structure to store kernel related info */

  public:

    CaLibsOCL(void);
    ~CaLibsOCL(void);
 
    /**
     * OpenCL related initialisations. 
     * @return ALIBS_SUCCESS on success and ALIBS_FAILURE on failure
     */
    int setupCL(cl_context context = 0,
		cl_device_type device = CL_DEVICE_TYPE_GPU,
		std::string accel_platform = "Advanced Micro Devices, Inc.",
		std::string ocl_kernels_path = "");

    /**
     * Cleanup memory allocations
     * @return ALIBS_SUCCESS on success and ALIBS_FAILURE on failure 
     */
    int cleanup();

    /**
     * Cleanup memory allocations     <<<<< ??
     *  @param prop array of CL queue properties <<<< ??
     *  @param deviceId id device of the queue   <<<< ?? 
     */
    inline cl_device_id getDeviceId(int device_indx = 0) const
    {
      cl_device_id devId = 0;
      if (device_indx < devices_.size())
	{
	  devId = devices_[device_indx];
	}
      return(devId);
    }

    // ?? are these making a COPY ?? of the command_queue object
    // ?? what is this (where is it defined)
    cl_command_queue getClQueue(cl_device_id deviceId, int queue_indx = 0);
    cl_command_queue getClQueue(int device_indx = 0, int queue_indx = 0);
    cl_command_queue createClQueue(int device_indx = 0, const cl_command_queue_properties *prop = 0);
    cl_command_queue createClQueue(cl_device_id deviceId, const cl_command_queue_properties *prop = 0);

    inline cl_context getClContext(void) const
    {
      return(context_);
    }

    int loadProgram(std::string & source, const std::string & _kernel_file, const std::string & _comp_options);

    cl_kernel getKernel(const std::string & kernel_file, const std::string & kernel_name, const std::string & comp_options);

    inline const std::string & getKernelPath(void) const
    {
      return(kernels_path_);
    }
    
    bool isDeviceCL20(int device_indx) const;
    int findDeviceIndxById(cl_device_id deviceId) const;

protected:
    bool own_context_;   // TT: not explicitly initialized on constructor
    int init_counter_;
    cl_device_type dType_;
    cl_platform_id platform_;
    cl_context context_;
    std:: vector<cl_device_id> devices_;
    std:: vector<alibs::aLibsDeviceInfo*> device_infors_;
    std::map<std::string,alibs::buildProgramData*> build_prog_map_;
    std::string kernels_path_;
    std::map<cl_device_id, std::vector<cl_command_queue>> queues_;  // TT: !init
};

/*---------------------------------------------------------
CABuf
----------------------------------------------------------*/
#define _CBUF_MEM_SYS_ONLY  ADNN_MEM_ALLOCSYS_ONLY
#define _CBUF_MEM_OCL_ONLY  ADNN_MEM_ALLOCOCL_ONLY

#ifdef TURNED_OFF
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
#endif

#endif
