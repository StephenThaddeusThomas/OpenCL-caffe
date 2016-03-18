// File:test_adnn_lib_fixture.hpp
// Path:MLopen/caffe/include/caffe/test
// From:http://www.ibm.com/developerworks/aix/library/au-googletestingframework.html
// Date:160301
// Time:2243
// Test:test_adnn_lib.cpp in MLopen/caffe/src/caffe/test 

// NOTE: The same test fixture is not used across multiple tests.  For every new unit test, the google test framework
// creates a new test fixture (that is constructs a adnn_lib_test_fixture object).
// THIS works well with our particular case, since we only can create ADNN Library Object    <<< add ProperName here

// See what Alex has for the C++ Library Object and use that in place of this
class adnn_lib_wrapper
{
public:

  adnn_lib_wrapper() { }
 ~adnn_lib_wrapper() { }

  void   init(void)
  {
    memset(&lib_prm, 0, sizeof(adnn_lib_parameters));
    lib_prm.accel_type = CL_DEVICE_TYPE_GPU;
    lib_prm.ocl_kernels_path = "/c/AMD/MLopen/addn/ocl"; // TT: need command line
  }
  void   create(void)   { lib_obj = ADNNLibCreate(&lib_prm); }
  char * name(void)     { return(ADNNLibGetName(lib_obj));   }  // called via TEST_F 
  void   destroy(void)  { ADNNLibDestroy(&lib_obj); }
  void   term(void)     {  }
  
protected:
  adnn_lib_paramters lib_prm;
  alib_obj           lib_obj;
};

/*
We have TWO ways to go
1. We make two more methods in adnn_lib_wrapper  (init(), term() and from those create and destroy the lib object)
   These methods would be called from SetUp() and TearDown() below
2. In the adnn_lib_wrapper ctor we call create () and in the dtor we call ADDNLibDestroy
*/

class adnn_lib_test_fixture : public ::testing::test
{
public:
  adnn_lib_test_fixture()
  {
  }

  // Virtual Interface to Test Fixture
  void SetUp()
  {
    // here we need to create an adnn library object
    alw.create();
  }

  void TearDown()
  {
    // here we need to destroy the adnn library object
    alw.destroy();
  }

  ~adnn_lib_test_fixture()
  {
    // note: no exceptions allowed
  }

protected:
  
  adnn_lib_wrapper alw;
  
};

