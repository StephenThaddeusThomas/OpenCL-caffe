// test_caffe_adnn_main.cpp
// from MLopen/caffe/src/caffe/test/test_caffe_main.cpp
// 140301

// The main caffe test code. Your test cpp code should include this hpp
// to allow a main function to be compiled into the binary.

// TT: these files can be found in /c/AMD/MLopen/caffe/include    <<<--- working location
// TT: or in /repo/stt/OpenCL-caffe/include                       <<<--- Repository

#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include "caffe/test/test_caffe_main.hpp"

// TT: see below
#include "/c/AMD/MLopen/incl/aDNN/AMDnn.h"


int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  caffe::GlobalInit(&argc, &argv);

#ifndef CPU_ONLY
  // Before starting testing, let's first print out a few cuda defice info.
  int device = 0;
  if (argc > 1)
    {
      // Use the given device
      device = atoi(argv[1]);
      caffe::amdDevice.Init(device);
      cout << "Setting to use device " << device << endl;
    }
  else
  if (OPENCL_TEST_DEVICE >= 0)
    {
      // Use the device assigned in build configuration; but with a lower priority
      device = OPENCL_TEST_DEVICE;
    }
  cout << "Current device id: " << device << endl;
  caffe::amdDevice.Init();
  
#endif

  // TT: 160301 - this same code is in: /c/AMD/MLopen/caffe/src/caffe/common.cpp
  // TT: 160228 - Stage 1 (Pizza) instantiate the aDNN library
  // TT: This bit of code will prove/test ability to compile and link
  adnn_lib_parameters  lib_params;
  memset(&lib_params, 0, sizeof(adnn_lib_parameters));
  lib_params.accel_type = CL_DEVICE_TYPE_GPU;
  lib_params.ocl_kernels_path = "/c/AMD/MLopen/addn/ocl"; // TT: need command line arg or prototxt entry 
  alib_obj aLib = ADNNLibCreate(&lib_params);
  printf("Created ADNN library %s\n", ADNNLibGetName(aLib));
  ADNNLibDestroy(&aLib);


  
  // invoke the test.
  return RUN_ALL_TESTS();
}
