// File:test_adnn_lib_fixture.hpp
// Path:MLopen/adnn/test/
// From:MLopen/caffe/include/caffe/test   <<<< could link 
// From:http://www.ibm.com/developerworks/aix/library/au-googletestingframework.html
// Date:160301
// Time:2243
// Test:test_adnn_lib.cpp in MLopen/caffe/src/caffe/test 

// NOTE: The same test fixture is not used across multiple tests.
// For every new unit test, the google test framework creates a new test fixture
// (that is constructs a adnn_lib_test_fixture object).
// THIS works well with our particular case, since we only can create ADNN Library Object

// See what Alex has for the C++ Library Object and use that in place of this ...
class adnn_lib_wrapper
{
public:

  adnn_lib_wrapper() { }
 ~adnn_lib_wrapper() { }

  void   init(void)
  {
    memset(&lib_prm, 0, sizeof(adnn_lib_parameters));
    lib_prm.accel_type = CL_DEVICE_TYPE_GPU;
    lib_prm.ocl_kernels_path = KERNEL_PATH;
  }
  void   create(void)   { lib_obj = ADNNLibCreate(&lib_prm); }
  const char * name(void)     { return(ADNNLibGetName(lib_obj));   }  // called via TEST_F 
  void   destroy(void)  { ADNNLibDestroy(&lib_obj);          }
  void   term(void)     {  }
  
protected:
  
  adnn_lib_parameters lib_prm;
  alib_obj            lib_obj;
};


class adnn_lib_test_fixture : public ::testing::Test
{
public:
  adnn_lib_test_fixture()
  {
    alw.init();
  }

  // Virtual Interface to Test Fixture
  virtual void SetUp()
  {
    // here we need to create an adnn library object
    alw.create();
  }

  virtual void TearDown()
  {
    // here we need to destroy the adnn library object
    alw.destroy();
  }

  ~adnn_lib_test_fixture()
  {
    // note: no exceptions allowed
    alw.term();
  }

protected:
  
  adnn_lib_wrapper alw;
  
};

