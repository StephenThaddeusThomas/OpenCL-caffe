// File:aDNNOCL.cpp
// Date:160221
// TASK:move all non-trivial method bodies from .hpp to the .cpp file
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

#include "aDNNInternal.hpp"

namespace adnn
{
  static CaLibsOCL * spaLibsOCL = NULL;

  CaLibsOCL & getaDNNOCL(void)
  {
    if (!spaLibsOCL)
      {
	spaLibsOCL = new CaLibsOCL;
      }
    return *spaLibsOCL;
  }

  CaLibsOCL * createaDNNOCL(void)
  {
    CaLibsOCL * ret = 0;
    ret = new CaLibsOCL;
    return ret;
  }

  CaLibsOCL::CaLibsOCL(void)
  {
    init_counter_ = 0;
    dType_ = CL_DEVICE_TYPE_GPU;
    platform_ = 0;
    context_ = 0;
    devices_.clear();
    device_infors_.clear();
    build_prog_map_.clear();
    kernels_path_ = "";
  }

  CaLibsOCL:: ~CaLibsOCL(void)
  {
    cleanup();
  }

  int CaLibsOCL::setupCL(cl_context _context,
			 cl_device_type deviceType,
			 std::string accel_platform,
			 std::string ocl_kernels_path)
  {
    cl_int status = ADNN_SUCCESS;

    dType_ = deviceType;
    if (++init_counter_ == 1)
      {
	if (_context != 0)
	  {
	    context_ = _context;
	  }
	else
	  {
	    own_context_ = true;

	    // Have a look at the available platforms and pick either
	    // the AMD one if available or a reasonable default.
	    cl_platform_id platform_ = NULL;
	    status = alibs::getPlatformL(platform_, 0, false, dType_, accel_platform);
	    CHECK_ERROR(status, ALIBS_SUCCESS, "graal::getPlatform() failed");

	    // Display available devices.
	    status = alibs::displayDevices(platform_, dType_);
	    CHECK_ERROR(status, ALIBS_SUCCESS, "graal::displayDevices() failed");

	    // If we could find our platform, use it. Otherwise use just available platform.
	    cl_context_properties cps[3] =
	      {
		CL_CONTEXT_PLATFORM, (cl_context_properties)platform_, 0
	      };
	    context_ = clCreateContextFromType(cps,
					       dType_,
					       NULL,
					       NULL,
					       &status);
	    CHECK_OPENCL_ERROR(status, "clCreateContextFromType failed.");
	  }

	// getting device on which to run the sample
	status = alibs::getDevices(context_, devices_, 0, false);
	CHECK_ERROR(status, 0, "sampleCommon::getDevices() failed");

	for (int i = 0; i < devices_.size(); i++)
	  {
	    alibs::aLibsDeviceInfo * deviceInfo = new alibs::aLibsDeviceInfo;
	    status = deviceInfo->setDeviceInfo(devices_[i]);
	    CHECK_OPENCL_ERROR(status, "deviceInfo.setDeviceInfo failed");
	    device_infors_.push_back(deviceInfo);
	  }

	// create generic default queue
	cl_command_queue_properties prop;
	memset(&prop, 0, sizeof(prop));
	createClQueue(0, &prop);
      }

    kernels_path_ = ocl_kernels_path;
    return status;
  }

  int CaLibsOCL::loadProgram(std::string & source, const std::string & _kernel_file, const std::string & _comp_options)
  {
    cl_int status = 0;
    std::string key = _kernel_file + "." + _comp_options;
    std::map<std::string, alibs::buildProgramData*>::iterator b;
    b = build_prog_map_.find(key);
    if (b == build_prog_map_.end())
      {
	// did not find the program
	// build it
	// create a CL program using the kernel source 
	alibs::buildProgramData *buildData = new alibs::buildProgramData;
	buildData->kernelName = _kernel_file;
	buildData->devices = &devices_[0];
	buildData->deviceId = 0;
	buildData->flagsStr = _comp_options;
	buildData->kernelPath = kernels_path_;
	status = loadOpenCLProgram(context_, *buildData);
	CHECK_OPENCL_ERROR_MSG(status, "load program failed.");

	build_prog_map_[key] = buildData;
      }
    b = build_prog_map_.find(key);
    source = (*b).second->source;
    return status;
  }

  cl_kernel CaLibsOCL::getKernel(const std::string & kernel_file, const std::string & kernel_name, const std::string & comp_options)
  {
    cl_kernel ret = 0;
    cl_int status = 0;
    std::string key = kernel_file + "." + comp_options;
    std::map<std::string, alibs::buildProgramData*>::iterator b;
    b = build_prog_map_.find(key);
    if (b == build_prog_map_.end())
      {
	printf("ERROR: cannot find build options in the cache\n");
	return(ret);
      }
    bool built = (*b).second->built;

// TODO: ATOMIC
    if (!built)
      {
	status = buildOpenCLProgram(context_, *b->second);
	CHECK_OPENCL_ERROR_MSG(status, "build failed.");
	(*b).second->built = true;
      }

    cl_program prog = (*b).second->program;
    // get a kernel object handle for a kernel with the given name 
    ret = clCreateKernel(prog, kernel_name.c_str(), &status);
    printf("Kernel is %s : 0x%x\n", kernel_name.c_str(), (long long)ret);
    CHECK_OPENCL_ERROR_MSG(status, "clCreateKernel failed.");

    return ret;
  }

  int CaLibsOCL::cleanup()
  {
    int status = CL_SUCCESS;
    if (spaLibsOCL && --init_counter_ == 0)
      {
	std::map<cl_device_id, std::vector<cl_command_queue>> ::iterator id;
	for (id = queues_.begin(); id != queues_.end(); ++id)
	  {
	    std::vector<cl_command_queue> & qs = (*id).second;
	    std::vector<cl_command_queue>::iterator iq;
	    for (iq = qs.begin(); iq != qs.end(); ++iq)
	      {
		clReleaseCommandQueue((*iq));
	      }
	  }
	devices_.clear();

	for (std::vector<alibs::aLibsDeviceInfo*>::iterator i = device_infors_.begin(); i != device_infors_.end(); i++)
	  {
	    delete *i;
	  }
	device_infors_.clear();

	for (std::map<std::string, alibs::buildProgramData*>::iterator b = build_prog_map_.begin(); b != build_prog_map_.end(); b++)
	  {
	    delete b->second;
	    b->second = 0;
	  }

	build_prog_map_.clear();

	if (own_context_ && context_)
	  {
	    status = clReleaseContext(context_);
	    CHECK_OPENCL_ERROR(status, "clReleaseContext failed.");
	    context_ = 0;
	  }
	own_context_ = false;
	spaLibsOCL = NULL;
	delete this;
      }
    return status;
  }

  cl_command_queue CaLibsOCL::createClQueue(int device_indx, const cl_command_queue_properties * prop)
  {
    cl_int status = CL_SUCCESS;
    cl_command_queue cQueue = 0;
		if (device_infors_.size() > device_indx)
		{
			if (isDeviceCL20(device_indx))
			{
				cQueue = clCreateCommandQueueWithProperties(context_,
					devices_[device_indx],
					prop,
					&status);
			}
			else
			{
				cQueue = clCreateCommandQueue(context_,
					devices_[device_indx],
					0,
					&status);
			}

		}
		if (cQueue)
		{
			if (queues_.empty())
			{
				std::vector<cl_command_queue> t;
				queues_[devices_[device_indx]] = t;
			}
			queues_[devices_[device_indx]].push_back(cQueue);
		}

		return(cQueue);
	}

	cl_command_queue CaLibsOCL::createClQueue(cl_device_id deviceId, const cl_command_queue_properties * prop)
	{
		cl_command_queue ret = 0;
		int indx = findDeviceIndxById(deviceId);
		if (indx != -1)
		{
			ret = createClQueue(indx, prop);
		}
		return(ret);
	}


	cl_command_queue CaLibsOCL::getClQueue(int device_indx, int queue_indx)
	{
		cl_command_queue cQueue = 0;
		if (device_indx < devices_.size())
		{
			cQueue = getClQueue(devices_[device_indx], queue_indx);
		}
		return cQueue;
	}

	cl_command_queue CaLibsOCL::getClQueue(cl_device_id deviceId, int queue_indx)
	{
		cl_command_queue ret = 0;
		if (!queues_[deviceId].empty())
		{
			ret = queues_[deviceId][queue_indx];
		}

		return(ret);
	}

	bool CaLibsOCL::isDeviceCL20(int device_indx) const
	{
		bool ret = false;
		int majorRev, minorRev;

		if (device_infors_.size() > device_indx && (sscanf((*device_infors_[device_indx]).deviceVersion, "OpenCL %d.%d", &majorRev, &minorRev) == 2) && majorRev >= 2)
		{
			ret = true;
		}
		return(ret);
	}

	int CaLibsOCL::findDeviceIndxById(cl_device_id deviceId) const
	{
		int ret = -1;
		for (int i = 0; i < devices_.size(); ++i)
		{
			if (devices_[i] == deviceId)
			{
				ret = i;
				break;
			}
		}
		return(ret);
	}
}
/*  IN PROGRESS (see T.cpp) --------------------------------------------------
#include "CABuf.hpp"
template class CABuf<float>;
template class CABuf<double>; 
*/ 

template <typename T> CABuf<T>::CABuf(cl_context _context)
{
  context_ = _context;
  mappingQ_ = 0;
  sys_ptr_ = 0;
  map_ptr_ = 0;
  buf_ = 0;
  len_ = 0;
  flags_ = 0;
  sys_own_ = false;
  cl_own_  = false;
  offset_ = 0;
}

//virtual
template <typename T>  CABuf<T>::~CABuf(void)
  {
    release();
  }


  /// TODO : correct copy costractor, operator =, clone()
template <typename T>   CABuf<T>::CABuf(const CABuf & _src)  { }

/** TODO CABuf clone(void)
    {
    CABuf new_buf(context_);
    return(new_buf);
    }
*/

template <typename T> int CABuf<T>::create(size_t _sz, uint _flags, const T *_buf)
{
    int ret = ALIBS_SUCCESS;
    if (_sz == 0 )
      {
	ret = ALIBS_FAILURE;
	return ret;
      }

    T * buf = (T *)_buf;

    if (_buf == 0 && !(_flags & _CBUF_MEM_OCL_ONLY) )
      {
	sys_ptr_ = new T[_sz];
	if (!sys_ptr_)
	  {
	    printf("error creating bufffer size %d: %d \n", _sz, ret);
	    return ret;
	  }
      }
    if (!(_flags & _CBUF_MEM_SYS_ONLY))
      {
// remove our flags
	uint flags = _flags & 0xffffff;

	buf_ = clCreateBuffer(context_, flags, _sz * sizeof(T), buf, &ret);
	if (ret != CL_SUCCESS)
	  {
	    if (sys_ptr_)
	      {
		delete[] sys_ptr_;
		sys_ptr_ = 0;
	      }
	    printf("error creating bufffer size %d: %d \n", _sz, ret);
	    return ret;
	  }
      }
    len_ = _sz;

    sys_own_ = (_buf == 0 && sys_ptr_)? true : false;
    cl_own_ = (buf_) ? true : false;
    flags_ = _flags;

    return(ret);
}

template<typename T> int CABuf<T>::create(size_t _sz, uint _flags)
  {
    int ret = create(_sz, _flags, 0);
    return(ret);
  }

// TODO :: CORRECT 
template<typename T> int CABuf<T>::attach(const T *_buf, size_t _sz)
{
  int ret = ALIBS_SUCCESS;
  uint flags = flags_;
  bool old_sys_own = sys_own_;
  T * old_ptr = sys_ptr_;

  if ( _sz > len_ )
    {
      release();
      create(_sz, flags);
    }

  if ( sys_own_ && sys_ptr_  && old_ptr != _buf)
    {
      delete [] sys_ptr_;
      sys_ptr_ = 0;
    }

  sys_ptr_ = (T*)_buf;
  len_ = _sz;
  sys_own_ = (old_ptr != _buf) ? false : old_sys_own;

  return(ret);
}

// TODO : CORRECT 

template<typename T> int CABuf<T>::attach(cl_mem _buf, size_t _sz)
{
  int ret = ALIBS_SUCCESS;

  if ( _buf != buf_ || _sz > len_ )
    {
      release();
      buf_ = _buf;
      len_ = _sz;
      cl_own_ = false;
    }
  return(ret);
}

template<typename T> T* CABuf<T>::map(uint _flags, cl_command_queue _mappingQ)
{
  T* ret = 0;
  int status = 0;
  if (buf_ && !map_ptr_  && !(flags_ & _CBUF_MEM_SYS_ONLY) )
    {
      mappingQ_ = (_mappingQ==NULL)? getaDNNOCL().getClQueue() : _mappingQ;
      ret = map_ptr_ = (T *)clEnqueueMapBuffer (mappingQ_,
						buf_,
						CL_TRUE,
						_flags, //CL_MAP_WRITE_INVALIDATE_REGION,
						0,
						len_*sizeof(T),
						0,
						NULL,
						NULL,
						&status);
    }
  else
  if (flags_ & _CBUF_MEM_SYS_ONLY)
    {
      ret = sys_ptr_;
    }
  return(ret);
}

template<typename T> T* CABuf<T>::mapA(cl_command_queue _mappingQ, uint _flags, cl_event *_wait_event, cl_event *_set_event )
{
  T* ret = 0;
  int status = 0;
  if ( buf_ && !map_ptr_ )
    {
      int n_wait_events = 0;
      cl_event * p_set_event = _set_event;
      cl_event * p_wait_event = _wait_event;
      if (_wait_event != NULL)
	{
	  n_wait_events = 1;
	}

      mappingQ_ = (_mappingQ==NULL)? getaDNNOCL().getClQueue() : _mappingQ;
	
      ret = map_ptr_ = (T *)clEnqueueMapBuffer (mappingQ_,
						buf_,
						CL_FALSE,
						_flags, //CL_MAP_WRITE_INVALIDATE_REGION,
						0,
						len_*sizeof(T),
						n_wait_events,
						p_wait_event,
						p_set_event,
						&status);
      if (_wait_event != NULL)
	{
	  clReleaseEvent(*_wait_event);
	  *_wait_event = NULL;
	}
    }
  return(ret);
}

template<typename T> int CABuf<T>::unmap(cl_event *_wait_event, cl_event *_set_event)
{
  int ret = ALIBS_SUCCESS;
  if (buf_ && map_ptr_ && mappingQ_ && !(flags_ & _CBUF_MEM_SYS_ONLY))
    {
      int n_wait_events = 0;
      cl_event * p_set_event = _set_event;
      cl_event * p_wait_event = _wait_event;
      if (_wait_event != NULL)
	{
	  n_wait_events = 1;
	}
      ret = clEnqueueUnmapMemObject(mappingQ_,
				    buf_,
				    map_ptr_,
				    n_wait_events,
				    p_wait_event,
				    p_set_event);
      
      if (_wait_event != NULL)
	{
	  clReleaseEvent(*_wait_event);
	  *_wait_event = NULL;
	}

      map_ptr_ = 0;
      mappingQ_ = 0;
    }
  return(ret);
}

template<typename T> int CABuf<T>::copyToDevice(uint _flags, const T* _data, size_t _len, cl_command_queue _commandQueue)
{
  int err = ALIBS_SUCCESS;

  if( (len_ > 0  && !sys_ptr_) || (_len > 0 && (_len > len_ ||  !_data) ) )
    {
      printf("wrong data\n");
      return(-1);
    }
    
  if ( !buf_ )
    {
      flags_ = _flags;
      buf_ = clCreateBuffer(context_, _flags, len_*sizeof(T), NULL, &err);
      if(err != CL_SUCCESS)
	{
	  printf("error creating bufffer: %d\n", err);
	  return err;
	}
      cl_own_ = true;
    }

  _commandQueue = (_commandQueue == NULL) ? getaDNNOCL().getClQueue() : _commandQueue;
  size_t len =(_data != NULL )? _len : len_; 
  const T * sys_ptr = (_data != NULL) ? _data : sys_ptr_;
  err = clEnqueueWriteBuffer(_commandQueue, buf_, CL_TRUE,0, len * sizeof(T), sys_ptr, 0, NULL, NULL);
  
  if(err != CL_SUCCESS)
    {
      printf("error writing data to device: %d\n", err);
      return err;
    }
  return(err);
}

template<typename T> int CABuf<T>::copyToDeviceA(uint _flags, const T* _data, size_t _len, cl_command_queue _commandQueue)
{
  int err = ALIBS_SUCCESS;
  
  if( (len_ > 0  && !sys_ptr_) || (_len!=-1 && (_len > len_ ||  !_data) ) )
    {
      printf("wrong data\n");
      return(-1);
    }   
  if ( !buf_ )
    {
      flags_ = _flags;
      buf_ = clCreateBuffer(context_, _flags, len_*sizeof(T), NULL, &err);
      if(err != CL_SUCCESS)
	{
	  printf("error creating bufffer: %d\n", err);
	  return err;
	}
      cl_own_ = true;
    }
  _commandQueue = (_commandQueue == NULL) ? getaDNNOCL().getClQueue() : _commandQueue;

  size_t len = (_len!=-1 )? _len : len_; 
  const T * sys_ptr = (_len!=-1) ? _data : sys_ptr_;
  err = clEnqueueWriteBuffer(_commandQueue, buf_, CL_FALSE,0, len * sizeof(T), sys_ptr, 0, NULL, NULL);
  
  if(err != CL_SUCCESS)
    {
      printf("error writing data to device: %d\n", err);
      return err;
    }
  return(err);
}

template<typename T> int CABuf<T>::copyToHost(cl_command_queue _commandQueue)
{
  int err = ALIBS_SUCCESS;

  if(len_ == 0 || !buf_)
    {
      printf("wrong data\n");
      return(-1);
    }
  
  if ( !sys_ptr_ )
    {
      sys_ptr_ = new T[len_];
      if(!sys_ptr_ )
	{
	  err = ALIBS_FAILURE;
	  printf("error creating bufffer: %d\n", err);
	  return err;
	}
      sys_own_ = true;
    }
  _commandQueue = (_commandQueue == NULL) ? getaDNNOCL().getClQueue() : _commandQueue;

  err = clEnqueueReadBuffer(_commandQueue, buf_, CL_TRUE,0, len_ * sizeof(T), sys_ptr_, 0, NULL, NULL);

  if(err != CL_SUCCESS)
    {
      printf("error writing data to device: %d\n", err);
      return err;
    }
  return(err);
}

template<typename T> int CABuf<T>::copy(CABuf<T> & _src, size_t _src_offset, size_t _dst_offset, cl_command_queue _commandQueue)
{
  int status;
  _commandQueue = (_commandQueue == NULL) ? getaDNNOCL().getClQueue() : _commandQueue;

  status = clEnqueueCopyBuffer(_commandQueue,
			       _src.getCLMem(),
			       getCLMem(),
			       _src_offset,
			       _dst_offset,
			       len_ * sizeof(T),
			       0,
			       NULL,
			       NULL);
  return (status);
}

template<typename T> int CABuf<T>::setValue(T _val, cl_command_queue _commandQueue)
{
  int err = ALIBS_SUCCESS;
  _commandQueue = (_commandQueue == NULL) ? getaDNNOCL().getClQueue() : _commandQueue;
  
  T * map_ptr = map(_commandQueue, CL_MAP_WRITE_INVALIDATE_REGION);
  for( int i = 0; i < len_; i++)
    {
      map_ptr[i] = _val;
      if ( sys_ptr_ )
	{
	  sys_ptr_[i] = _val;
	}
    }
  unmap();
  return(err);
}

// ?? What is difference - where are the comments
  
template<typename T> int CABuf<T>::setValue2(T _val, cl_command_queue _commandQueue)
{
  int err = ALIBS_SUCCESS;
  _commandQueue = (_commandQueue == NULL) ? getaDNNOCL().getClQueue() : _commandQueue;
  std:: string kernel_file = "GraalUtil.cl", kernel_name = "amdSetValue", comp_options;
  std:: string type_nm(typeid( _val ).name());
  
  if (type_nm == "unsigned int" )
    {
      type_nm = "uint";
    }
  comp_options = std::string("-D __TYPE__=") + type_nm;

  cl_kernel setValueKernel = getaDNNOCL().getKernel(kernel_file, kernel_name, comp_options);

  int n_arg = 0;

  err = clSetKernelArg(setValueKernel, n_arg++, sizeof(cl_mem), &buf_);
  err |= clSetKernelArg(setValueKernel, n_arg++, sizeof(T), &_val);
  size_t l_wk[3] = {256,1,1};
  size_t g_wk[3] = {1,1,1};
  g_wk[0] = len_;
  err |= clEnqueueNDRangeKernel(_commandQueue, setValueKernel, 1, NULL, g_wk, l_wk, 0, NULL, NULL);
  //TT: ?? do each of the above cl methods return a different bit region error
  clFinish(_commandQueue);
  for( int i = 0; sys_ptr_ && i < len_; i++)
    {
      sys_ptr_[i] = _val;
    }
  return(err);
}

template<typename T> int CABuf<T>::release(void)
{
  int ret = ALIBS_SUCCESS;
  if ( sys_own_ && sys_ptr_) 
    {
      delete [] sys_ptr_;
      sys_ptr_ = 0;
    }
  sys_own_  = false;

  if ( cl_own_ ) 
    {
      unmap();
      if ( buf_ )
	{
	  ret = clReleaseMemObject(buf_);
	}
      buf_ = 0;
    }

  cl_own_ = false;
  len_ = 0;

  return(ret);
}

template<typename T> CASubBuf<T> :: CASubBuf(CABuf<T> & _base) : CABuf<T>(_base.getContext())
{
  base_buf_ = _base.getCLMem();
  sys_ptr_ = _base.getSysMem();
  assert(base_buf_);
}

template<typename T> int CASubBuf<T>::create(size_t _offset, size_t _sz, uint _flags)
{
  int ret = ALIBS_SUCCESS;
  if (_sz == 0 )
    {
      ret = ALIBS_FAILURE;
      return ret;
    }

  if ( sys_ptr_ )
    {
      sys_ptr_ += _offset;
    }

  cl_buffer_region sub_buf;

  sub_buf.origin = _offset* sizeof(T);
  sub_buf.size = _sz * sizeof(T);

  buf_ = clCreateSubBuffer (base_buf_,
			    _flags,
			    CL_BUFFER_CREATE_TYPE_REGION,
			    &sub_buf,
			    &ret);

  if(ret != CL_SUCCESS)
    {
      printf("error creating bufffer: %d\n", ret);
      return ret;
    }

  len_ = _sz;
  cl_own_ = true;
  flags_ = _flags;
	
  return(ret);
}

template<typename T> int CASubBuf<T>::create(size_t _sz, uint _flags)
{
  create(0, _sz, _flags);
  // TT: what return should be here
}

// END 
