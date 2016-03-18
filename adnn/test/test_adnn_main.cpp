// File:test_adnn_main.cpp
// Date:160302
// Time:0113
// from MLopen/caffe/src/caffe/test/test_caffe_adnn_main.cpp


#include "/c/AMD/MLopen/incl/aDNN/AMDnn.h"


int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);

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
