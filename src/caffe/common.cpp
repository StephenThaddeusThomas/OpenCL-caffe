// File:common.cpp
// Path:/c/AMD/MLopen/caffe/src/caffe
// Host:thaddeus-nn (TheBeast)
// Date:160228
// What:Stage1 Adding in the aDNN include file AMDnn.h and AMDnn.hpp.
// These include files originally where in apcLibs/aDNN/aLibDNN/
// these include files are linked (hard) into MLopen/incl/aDNN (keeping with caffe)
// Stage1 Added in call to create the aDNN library, print its name, then it gets deleted
// Stage1 Purpose is to see if this will compile and link
// Stage2 ADNN elements are now part of Caffe class.  ADNN library is now ccreated in Constructor
// Stage2 Moved to Device 

#include <glog/logging.h>
#include <cstdio>
#include <ctime>

// TT: Test1 160228 - I've added these in - in hardcoded ("") vs softcode (<>) because I have yet to modify the cmake files that will set the include directories
//#include "/c/AMD/MLopen/incl/aDNN/AMDnn.h"
#include <CL/opencl.h>

#include "caffe/common.hpp"
#include "caffe/util/rng.hpp"

namespace caffe
{
  shared_ptr<Caffe> Caffe::singleton_;

  // random seeding
  int64_t cluster_seedgen(void)
  {
    //To fix: for now we use fixed seed to get same result each time
    int64_t s, seed, pid;
    FILE* f = fopen("/dev/urandom", "rb");
    if (f && fread(&seed, 1, sizeof(seed), f) == sizeof(seed))
      {
	fclose(f);
	return seed;
      }

    LOG(INFO) << "System entropy source not available, using fallback algorithm to generate seed instead.";
    if (f)
      fclose(f);

    pid = getpid();
    s = time(NULL);
    seed = abs(((s * 181) * ((pid - 83) * 359)) % 104729);
    return seed;
  }

  void GlobalInit(int* pargc, char*** pargv)
  {
    // Google flags.
    ::gflags::ParseCommandLineFlags(pargc, pargv, true);

    // Google logging.
    ::google::InitGoogleLogging(*(pargv)[0]);

    // Provide a backtrace on segfault.
    ::google::InstallFailureSignalHandler();

printf("GlobalInit:%i",__LINE__);
#ifdef ADNN_INTEGRATION_POSITION_1
    // see quick_hybrid_gpu.out for this location
    // Now moving this code to Main in tools/caffe.cpp
    // TT:160228 - Stage 1 (Pizza) instantiate the aDNN library
    // This bit of code will prove/test ability to compile and link
    // This same code is in /c/AMD/MLopen/caffe/src/caffe/test/test_caffe_adnn_main.cpp
    // thaddeus-nn (the host) exports /c
    // the developer machine mounts the directory on /mnt/tnnc
  
//    adnn_lib_parameters  lib_params;
//    memset(&lib_params, 0, sizeof(adnn_lib_parameters));
//    lib_params.accel_type = CL_DEVICE_TYPE_GPU;
//    lib_params.ocl_kernels_path = "/c/AMD/MLopen/addn/ocl"; // TT: need command line arg or prototxt entry 
//    alib_obj aLib = ADNNLibCreate(&lib_params);
//    printf("Created ADNN library Common %s\n", ADNNLibGetName(aLib));
//    ADNNLibDestroy(&aLib);
//    printf("Destroyed ADNN library Common \n");
#error WHAT 
#endif
}

#ifdef CPU_ONLY  // CPU-only Caffe.

Caffe::Caffe()
: random_generator_(), mode_(Caffe::CPU)
{
LOG(INFO) << "+Caffe ModeCPU"; 
//  Stage2 Test2 - split data from methods - data in Caffe class in common.hpp
//  memset(&lib_params, 0, sizeof(adnn_lib_parameters));
//  lib_params.accel_type = CL_DEVICE_TYPE_GPU;
//  lib_params.ocl_kernels_path = "/c/AMD/MLopen/addn/ocl"; // TT: need command line arg or prototxt entry 
//  aLib = ADNNLibCreate(&lib_params);
//  printf("Created ADNN library %s\n", ADNNLibGetName(aLib));
}

Caffe::~Caffe()
{
//  Stage2 Test2 - NOTE: this print was not seen in the output 
//  ADNNLibDestroy(&aLib);
//  printf("Destroyed ADNN library\n");
}

void Caffe::set_random_seed(const unsigned int seed) {
  // RNG seed
  Get().random_generator_.reset(new RNG(seed));
}

void Caffe::SetDevice(const int device_id) {
  NO_GPU;
}

void Caffe::DeviceQuery() {
  NO_GPU;
}

class Caffe::RNG::Generator {
  public:
  Generator() : rng_(new caffe::rng_t(cluster_seedgen())) {}
  explicit Generator(unsigned int seed) : rng_(new caffe::rng_t(seed)) {}
  caffe::rng_t* rng() {return rng_.get();}
  private:
  shared_ptr<caffe::rng_t> rng_;
};

Caffe::RNG::RNG() : generator_(new Generator()) {}

Caffe::RNG::RNG(unsigned int seed) : generator_(new Generator(seed)) {}

Caffe::RNG& Caffe::RNG::operator=(const RNG& other) {
  generator_ = other.generator_;
  return *this;
}

void* Caffe::RNG::generator() {
  return static_cast<void*>(generator_->rng());
}

#else  // Normal GPU + CPU Caffe.

Caffe::Caffe() 
{
  LOG(INFO) << "+Caffe";
                        // TT: Junli - This is what needs to be changed to interface to Alex code 
  amdDevice.Init();  	// Need to save the Handle of the Library, this Handle has to be passed throughout Caffe, and used to create each layer 
		     	// for example  lib_handle=amdDevice.Init(); 

  cl_int err = clblasSetup();
  if (err != CL_SUCCESS) 
    {
     LOG(ERROR) << "clBLAS setup failed " << err;
    }
}

Caffe::~Caffe() 
{
  LOG(INFO) << "-Caffe"; 
  // TT: assume this is where we will call something to 'release' Alex's library 
  // TODO

  // BLAS destructor
  clblasTeardown();
}

void Caffe::set_random_seed(const unsigned int seed) 
{
  // RNG (Random Number Generoator) seed
  Get().random_generator_.reset(new RNG(seed));
  caffe_gpu_uniform(0, NULL, seed);
  caffe_gpu_uniform((float*)NULL, 0, (float)0.0, (float)1.0, seed);
}

void Caffe::SetDevice(const int device_id) {
  if (amdDevice.GetDevice() == device_id) {
    return;
  }
  amdDevice.Init(device_id);
}

void Caffe::DeviceQuery() {
  amdDevice.DeviceQuery();
}

class Caffe::RNG::Generator {
  public:
    Generator()
        : rng_(new caffe::rng_t(cluster_seedgen())) {
    }
    explicit Generator(unsigned int seed)
        : rng_(new caffe::rng_t(seed)) {
    }
    caffe::rng_t* rng() {
      return rng_.get();
    }
  private:
    shared_ptr<caffe::rng_t> rng_;
};

Caffe::RNG::RNG()
    : generator_(new Generator()) {
}

Caffe::RNG::RNG(unsigned int seed)
    : generator_(new Generator(seed)) {
}

Caffe::RNG& Caffe::RNG::operator=(const RNG& other) {
  generator_.reset(other.generator_.get());
  return *this;
}

void* Caffe::RNG::generator() {
  return static_cast<void*>(generator_->rng());
}

#endif  // CPU_ONLY

}  // namespace caffe
