// File:test_caffe_adnn_lib.cpp   Instantiate Caffe then Adnn Using Test Fixture 
// NOTE:Thinking about creating another version where the instantiation order of caffe comes AFTER adnn
// Merge of MLopen/adnn/gtest/test_adnn_lib.cpp and MLopen/caffe/src/caffe/test/test_caffe_main.cpp
// Path:MLopen/caffe/src/caffe/test/
// Date:160302
// Time:1604
// Uses:test_adnn_lib_fixture.hpp if compiled wiht -DUSE_GOOGLE_TEST_FIXTURE
// Locations:
// /repo/google/googletest/include/gtest/internal/custom/gtest.h
// /repo/googletest/googletest/include/gtest/gtest.h  <<<< the public one
// /repo/google/googletest/build/libgtest.a
// -rw-rw-r-- 1 thaddeus amdnnet 1755538 Mar  215:15 libgtest.a

// CommandLine Build: (all you need in one simple easy line)

// For building with FIXTURE (TEST_F)   >>>> These are INCOMPLETE see >>> compile.* in /c/AMD/MLopen/caffe/src/caffe/test/ <<<<<
// g++ -o test_adnn_lib_fixture.alnx -Wall -Wshadow -DGTEST_HAS_PTHREAD=1 -fexceptions -Wextra -Wno-missing-field-initializers  -I/c/AMD/MLopen/caffe/build/external/gflags-install/include -I/c/AMD/MLopen/caffe/include -I/c/AMD/MLopen/caffe/build/include -I/repo/google/googletest/include -DUSE_GOOGLE_TEST_FIXTURE test_caffe_adnn_lib.cpp /c/AMD/MLopen/adnn/lib/libaDNN.a /repo/google/googletest/build/libgtest.a /c/AMD/MLopen/appsdk/lib/x86_64/libOpenCL.so -lpthread /opt/clBLAS-2.3/lib64/libclBLAS.so

// For building without Fixture
// g++ -o test_adnn_lib.alnx -Wall -Wshadow -DGTEST_HAS_PTHREAD=1 -fexceptions -Wextra -Wno-missing-field-initializers -I/c/AMD/MLopen/caffe/include -I/c/AMD/MLopen/caffe/build/external/gflags-install/include -I/repo/google/googletest/include  test_caffe_adnn_lib.cpp /c/AMD/MLopen/adnn/lib/libaDNN.a /repo/google/googletest/build/libgtest.a /c/AMD/MLopen/appsdk/lib/x86_64/libOpenCL.so -lpthread /opt/clBLAS-2.3/lib64/libclBLAS.so

// CAFFE First
#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include "caffe/test/test_caffe_main.hpp"

// Google Test
#include "/repo/google/googletest/include/gtest/gtest.h"

// ADNN 
#include "/c/AMD/AppSDK/include/CL/opencl.h"
#include "/c/AMD/MLopen/adnn/inc/AMDnn.h"

// LIB NAME is what should be returned from ADNNLibGetName
#define  LIB_NAME "ADNN Library v0.01"

// KERNEL PATH should become a command line argument, this is the default..
#define  KERNEL_PATH "/c/AMD/MLopen/adnn/ocl"

static
void caffe_init(int argc, char **argv)
 {
   cout << "CaffeInit" << endl; 
#ifdef USE_GPU
   using namespace std;
   cout << "GPU ENABLED" << endl; 
   // Before starting testing, let's first print out a few cuda defice info.
   int device = 0;
   if(argc>1)
     {
       // Use the given device
       device = atoi(argv[1]);
       caffe::amdDevice.Init(device);
       cout << "Setting to use device " << device << endl;
     }
   else
   if(OPENCL_TEST_DEVICE >= 0)
     {
       // Use the device assigned in build configuration; but with a lower priority
       device = OPENCL_TEST_DEVICE;
     }
   cout << "Current device id: " << device << endl;
   caffe::amdDevice.Init();     // 160303 THIS IS FAILING - trying test_caffe_main.cpp 
#else
   cout << "CPU ONLY" << endl; 
#endif
 }

// GoogleTest has two ways of defining Unit test.
// Plain test (TEST) and Fixture Test. (TEST_F), the program can be compiled for either
// TEST_F is a TEST like macro that uses Fixtures which are instantiated for EACH unit test
// TEST is A unit test, each statement is part of that Unit Test - if any fail, the test fails

#ifdef USE_GOOGLE_TEST_FIXTURE

namespace {
  
 #include "/c/AMD/MLopen/adnn/test/test_adnn_lib_fixture.hpp"

 TEST_F (adnn_lib_test_fixture, unit_test)
  {
   // To test that the ADNN Library object is functional we get its name and compare that to the name we expect
   // These use C Strings  (line 19286 in gtest.h) 
   EXPECT_STREQ(LIB_NAME, alw.name());
  }

} // anonymous namespace

 int main(int argc, char **argv)
  {
    std::cerr << *argv << " FIXTURES line:" << __LINE__ << std::endl;
    ::testing::InitGoogleTest(&argc, argv);
    caffe::GlobalInit(&argc, &argv);
    caffe_init(argc,argv);
    return RUN_ALL_TESTS() ;
  }

#else

 // For plain TEST (non-fixture)
 static adnn_lib_parameters lib_prm;
 static alib_obj            lib_obj;

 // This is a standard Unit TEST
 // To disable this test, prefix with DISABLE_  (eg. DISABLE_adnn_lib_test, get_lib_name) 
TEST (adnn_lib_test, get_lib_name)
{
  // To test that the ADNN Library object is functional we get its name and compare that to the name we expect
  // These use C Strings  (line 19286 in gtest.h) 
  EXPECT_STREQ(LIB_NAME, ADNNLibGetName(lib_obj));
}

static
void adnn_create(void)
{
  std::cout << "ADNN Create" << std::endl;
  memset(&lib_prm, 0, sizeof(adnn_lib_parameters));
  lib_prm.accel_type = CL_DEVICE_TYPE_GPU;
  lib_prm.ocl_kernels_path = KERNEL_PATH ; 
  lib_obj = ADNNLibCreate(&lib_prm);
}

 static
 void adnn_destroy(void)
 {
   std::cout << "ADNN Destroy" << std::endl;  
   ADNNLibDestroy(&lib_obj);
 }

int main(int argc, char **argv)
{
  std::cerr << *argv << " Standard line:" << __LINE__ << std::endl;
  ::testing::InitGoogleTest(&argc, argv);
  caffe::GlobalInit(&argc, &argv);
  caffe_init(argc,argv);
  adnn_create();
  int rc=RUN_ALL_TESTS();
  adnn_destroy();
  return(rc);
}

#endif


/* ----- OUTPUT ---------------------------------------------------------------

** ----------------------------------------------------------------------------*/
// END
