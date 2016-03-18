// File:test_adnn_lib.cpp
// Path:MLopen/adnn/gtest
// From:MLopen/caffe/src/caffe/test/test_caffe_adnn_lib.cpp 
// Ref_:http://www.ibm.com/developerworks/aix/library/au-googletestingframework.html
// Date:160301
// Time:2255
// Uses:test_adnn_lib_fixture.hpp  
// /repo/google/googletest/include/gtest/internal/custom/gtest.h
// /repo/googletest/googletest/include/gtest/gtest.h  <<<< the public one

#include "/repo/google/googletest/include/gtest/gtest.h"

#include "/c/AMD/MLopen/incl/aDNN/AMDnn.h"

// LIB NAME is what should be returned from ADNNLibGetName
#define  LIB_NAME "ADNN Library v0.01"

// KERNEL PATH should become a command line argument, this is the default..
#define  KERNEL_PATH "/c/AMD/MLopen/addn/ocl"

// GoogleTest has two ways of defining Unit test.
// Plain test (TEST) and Fixture Test. (TEST_F), the program can be compiled for either
// TEST_F is a TEST like macro that uses Fixtures which are instantiated for EACH unit test
// TEST is A unit test, each statement is part of that Unit Test - if any fail, the test fails

#ifdef USE_GOOGLE_TEST_FIXTURE

 #include "test_adnn_lib_fixture.hpp"

 TEST_F (adnn_lib_test_fixture, unit_test)
  {
   // To test that the ADNN Library object is functional we get its name and compare that to the name we expect
   // These use C Strings  (line 19286 in gtest.h) 
   EXPECT_STREQ(LIB_NAME, alw.name());
  }

 int main(int argc, char **argv)
  {
   ::testing::InitGoogleTest(&argc,argv);
   return(RUN_ALL_TESTS);
  }

#else

 // For plain TEST (non-fixture)
 static adnn_lib_paramters lib_prm;
 static alib_obj           lib_obj;

 // This is a standard Unit TEST
 // To disable this test, prefix with DISABLE_  (eg. DISABLE_adnn_lib_test, get_lib_name) 
 TEST (adnn_lib_test, get_lib_name)
  {
   // To test that the ADNN Library object is functional we get its name and compare that to the name we expect
   // These use C Strings  (line 19286 in gtest.h) 
   EXPECT_STREQ(LIB_NAME, ADNNLibGetName(libobj));
  }

 void create(void)
  {
    memset(&lib_prm, 0, sizeof(adnn_lib_parameters));
    lib_prm.accel_type = CL_DEVICE_TYPE_GPU;
    lib_prm.ocl_kernels_path = KERNEL_PATH ; 
    lib_obj = ADNNLibCreate(&lib_prm);
  }

 void destroy(void)
  {
    ADNNLibDestroy(&lib_obj);
  }

 int main(int argc, char **argv)
  {
   create();
   ::testing::InitGoogleTest(&argc,argv);
   int rc=(RUN_ALL_TESTS);
   destroy();
   return(rc);
  }

#endif
// END
