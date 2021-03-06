############################################################################################################
# File: /hybrid/stage4/src/makefile (link to makefile.stage4)
# From: /AMD/MLopen/hybrid/SingleConvLayer/makefile.SCL (SingleConvLayer --> stage3/src) 
# date: 160315
############################################# Linux GCC Environment ########################################
# gcc --version
# gcc (Ubuntu/Linaro 4.4.4-14ubuntu5) 4.4.5
# Linux man page for ptheads
#   Compiling on Linux
#       On Linux, programs that use the Pthreads API should be compiled using cc -pthread.
############################################################################################################
OS=linux
app=stg4
lib=libBrdg.a
v=.stage4
t=.alnx

ROOT=/c/AMD
PROJ=MLopen/hybrid
CAFFE=MLopen/caffe
STAGE=stage4

# 160310 : updated these (Root)/Include --> (Root)/incl and AMDSDK is not on /opt/AMDAPPSDK-3.0
# LOCAL is for common items that are not in development
# STAGE is for libraries, objects, binaries that are being tested with this stage 
LOCAL_PATH=$(ROOT)/$(PROJ)
LOCAL_BIN_PATH = $(LOCAL_PATH)/bin
LOCAL_LIB_PATH = $(LOCAL_PATH)/lib
LOCAL_SRC_PATH = $(LOCAL_PATH)/src
LOCAL_INC_PATH = $(LOCAL_PATH)/inc
LOCAL_OBJ_PATH = $(LOCAL_PATH)/obj
STAGE_PATH=$(ROOT)/$(PROJ)/$(STAGE)
STAGE_BIN_PATH = $(STAGE_PATH)/bin
STAGE_LIB_PATH = $(STAGE_PATH)/lib
STAGE_SRC_PATH = $(STAGE_PATH)/src
STAGE_INC_PATH = $(STAGE_PATH)/inc
STAGE_OBJ_PATH = $(STAGE_PATH)/obj

COMMON_INC_PATH= $(ROOT)/incl
AMDSDK_INC_PATH= /opt/AMDAPPSDK-3.0
CAFFE_INC_PATH = $(ROOT)/$(CAFFE)/include
GFLAG_INC_PATH = $(ROOT)/$(CAFFE)/external/gflags-install/include
BUILD_INC_PATH = $(ROOT)/$(CAFFE)/build/include
#Note: the contents (files/directories) of BUILD_INC are only generated by running
# cmake ; make ;  and is where the caffe.pb.h file lives which is needed to parse
# the prototext buffers

# 160310 : Roles are reversed, Local<~>Remote
LOCAL_HOST=thaddeus-nn
LOCAL_REPO=/repo/stt
REMOTE_HOST=TT
REMOTE_REPO_PATH = $(REMOTE_HOST):/repo/stt
GITHUB_APC_FORK = https://github.com/StephenThaddeusThomas/apcLibs.git
GITHUB_CAFFE_FORK = https://github.com/StephenThaddeusThomas/OpenCL-caffe.git

########################################################################################
# COMPILE 
# The following is from /c/Make/makefile.notes and vlp0:/c/test/14.8/makefile.test.irxos
########################################################################################

# REMOVED -DUSE_PTHREADS (i think this was only needed for SGI)
# -traditional-cpp was to see if we can work around <iostream> (w/o .h)
# -frepo implies -fno-implicit-templates see ProgDvlp.110118
# -std=c++98
# -falt-external-templates
# -fexternal-templates
# -W
# -fPIC
# 160310 : Removed  -fno-implicit-templates and Changed c++0x-->c++11 
C= g++ -mglibc -pipe -pthread -c -D_REENTRANT -std=c++11 -frepo
c= gcc -mglibc -pipe -c  

# FLAGS
# These compiler options are listed in three DBG levels
# Level 1 is to get things up and running (espeically the -fpermissive) with checks that make sense when porting form AIX/IRIX to Linux
# Level 2 is to get more strict, after we have some time to test and optimize (save space)
# Level 3 is all warnings
# The Flag ($F) can be concatenated.  eg F=$(C_DBG_1)$(C_DBG_2) or F=$(C_DBG_1)$(C_DBG_3) or when done, F=$(C_OPT)
# -Wall includes many of the -w's defined so using -Wall is a bit redundant - the -W's defined above I think should be used, so  
# -Wimplicit ==  -Wimplicit-int -Wimplicit-function-declaration
C_DBG_0 = -w
C_DBG_1 = -Wctor-dtor-privacy -Wnon-virtual-dtor -Wreorder -fpermissive -Wno-deprecated -Wno-non-template-friend  -Wswitch-enum -Wswitch-default -Wcast-align 
C_DBG_2 = -Wextra -Weffc++ -Wfatal-errors -Wmissing-include-dirs  -Wundef -Wunused -Wunused-function -Wunused-label -Wunused-parameter -Wunused-macros -Wundef -Wlogical-op -Waggregate-return -Wredundant-decls -Wtraditional -Wimplicit -Wsequence-point -Wfloat-equal 
C_DBG_3 = -Wall
C_OPT   = -O2
F= $(C_DBG_1) -Wno-write-strings  -Wfatal-errors
E= -DLNX -DVLP -time 
G= -DVERBOSE -fmessage-length=0 -g2 -ggdb 
I= -I$(STAGE_INC_PATH) -I$(CAFFE_INC_PATH) -I$(AMDSDK_INC_PATH)
L_OPT = -O2 -s
L_DBG = -g 
L= -L$(STAGE_LIB_PATH)
# 160310 removed:  -shared-libgcc
K= g++ -m64 -o
P= -fprofile-arcs
R= -fno-rtti
X= -Q -H

## STAGE4
#  There are two sets of files
#1. those that go into the Bridge (so caffe can call ADNN)
#2. those that (eventually) will be part of caffe, but for now, are compiled outside of it 

#################################### COMPONENTS/OBJECTS ###########################################
# Path: /c/AMD/MLopen/hybrid/stage4/src

# Caffe (#2)
adnn_conv_layer.o    : adnn_conv_layer.cpp 		; $C $E $F $G $I -I$(GFLAG_INC_PATH) -I$(BUILD_INC_PATH) $<

# Command Line (standalone)
main4infer.o         : main4.cpp	   		; $C -o main4infer.o -DTEST_INFER $E $F $G $I -I$(GFLAG_INC_PATH) -I$(BUILD_INC_PATH) $<
main4train.o	     : main4.cpp			; $C -o main4train.o -DTEST_TRAIN $E $F $G $I -I$(GFLAG_INC_PATH) -I$(BUILD_INC_PATH) $<

# Files that Use Blobs need GFLAG_ and BUILD_INC_PATH
AdnnSetupLayer.o     : AdnnSetupLayer.cpp               ; $C $E $F $G $I  $< 
AdnnConvLayerInfer.o : AdnnConvLayerInfer.cpp 		; $C $E $F $G $I -I$(GFLAG_INC_PATH) -I$(BUILD_INC_PATH) $<
AdnnConvLayerTrain.o : AdnnConvLayerTrain.cpp		; $C $E $F $G $I -I$(GFLAG_INC_PATH) -I$(BUILD_INC_PATH) $<
PrepareConvNode.o    : PrepareConvNode.cpp 		; $C $E $F $G $I  $<

#################################### Target and Parent OBJS ########################################

# CaffeModules (objects)
caffe_obj = adnn_conv_layer.o

# There are TWO versions of this - it all depends which one you linke with see makefile.SCL
lib_infer_obj = AdnnSetupLayer.o AdnnConvLayerInfer.o PrepareConvNode.o  # Infer
lib_train_obj = AdnnSetupLayer.o AdnnConvLayerTrain.o PrepareConvNode.o  # Train

# First Version is for a Foward Only Single Convolutions
app_infer_obj = main4infer.o $(lib_infer_obj)
app_train_obj = main4train.o $(lib_train_obj)

# 160315 : removed $L and put in hard paths to developmental libraries
apptrain : $(app_train_obj)
	$K $(STAGE_BIN_PATH)/train$v$t $E $F $G $(app_train_obj) $I  -lm  $(STAGE_LIB_PATH)/libcaffe$v.so $(LOCAL_LIB_PATH)/libaDNN.a /c/AMD/AppSDK/lib/x86_64/libOpenCL.so  /opt/clBLAS-2.3/lib64/libclBLAS.so

# 160315 : removed $L
appinfer : $(app_infer_obj)
	$K $(STAGE_BIN_PATH)/infer$v$t $E $F $G $(app_infer_obj) $I  -lm  $(STAGE_LIB_PATH)/libcaffe$v.so $(LOCAL_LIB_PATH)/libaDNN.a /c/AMD/AppSDK/lib/x86_64/libOpenCL.so  /opt/clBLAS-2.3/lib64/libclBLAS.so

libtrain : $(lib_train_obj)
	ar -v -r $(STAGE_LIB_PATH)/libtrain.a $(lib_train_obj)

libinfer : $(lib_infer_obj)
	ar -v -r $(STAGE_LIB_PATH)/libinfer.a $(lib_infer_obj)

likenew     : clean fresh lib

clean:
	/bin/rm -f *.o *.rpo 
	/bin/rm -f *~ 

fresh:
	clear; clear > /dev/pts/1

# before 'make caffe' cd to caffe/build and run make there - it will be missing a few things (no worry) see Procedures.stage4.notes
# note that hybrid/lib/libcaffe.so is linked to libcaffe.stage#.so 
caffe :
	cp /c/AMD/MLopen/caffe/build/tools/CMakeFiles/caffe.bin.dir/caffe.cpp.o $(STAGE_OBJ_PATH)/caffe$v.o
	cp /c/AMD/MLopen/caffe/build/lib/libcaffe.so $(STAGE_LIB_PATH)/libcaffe$v.so
	sum /c/AMD/MLopen/caffe/build/lib/libcaffe.so $(STAGE_LIB_PATH)/libcaffe$v.so


# In this stage, there are two types of 'bridge' libraries
# 1. libinfer.a : for use when running "caffe  test ...."
# 2. libtrain.a : for use when running "caffe train -solver=..... "
# For what we need now for testing is the 'train' version which does forward and backward propogateion

#ERROR : /opt/AMDAPPSDK-3.0 does not contain libOpenCL.so - why did this not install correctly, using /c/AMD/AppSDK/lib/x86_64
#NOTE: this requires the special build of caffe.so (called libcaffe+stage3.so) this is done by following Procedures.stage4.notes 
train : libtrain
	g++ $(STAGE_OBJ_PATH)/caffe$v.o -o $(STAGE_BIN_PATH)/caffeTRAIN$v$t -std=c++11 \
$(STAGE_LIB_PATH)/libcaffe$v.so \
$(STAGE_LIB_PATH)/libtrain.a \
$(LOCAL_LIB_PATH)/libaDNN.a \
$(LOCAL_LIB_PATH)/libgtest.a  $(LOCAL_LIB_PATH)/libgflags.a $(LOCAL_LIB_PATH)/libproto.a \
/usr/local/lib/libglog.so /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5_hl.so /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5.so \
/c/AMD/AppSDK/lib/x86_64/libOpenCL.so /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.9 /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.9 \
/usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.9 /opt/clBLAS-2.3/lib64/libclBLAS.so \
-lpthread -lprotobuf -lz -ldl -lm -llmdb -lleveldb -lsnappy -llapack_atlas -lcblas -latlas -lboost_system -lboost_thread 

infer : libinfer
	g++ $(STAGE_OBJ_PATH)/caffe$v.o -o $(STAGE_BIN_PATH)/caffeINFER$v$t -std=c++11 \
$(STAGE_LIB_PATH)/libcaffe$v.so \
$(STAGE_LIB_PATH)/libinfer.a \
$(LOCAL_LIB_PATH)/libaDNN.a \
$(LOCAL_LIB_PATH)/libgtest.a  $(LOCAL_LIB_PATH)/libgflags.a $(LOCAL_LIB_PATH)/libproto.a \
/usr/local/lib/libglog.so /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5_hl.so /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5.so \
/c/AMD/AppSDK/lib/x86_64/libOpenCL.so /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.9 /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.9 \
/usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.9 /opt/clBLAS-2.3/lib64/libclBLAS.so \
-lpthread -lprotobuf -lz -ldl -lm -llmdb -lleveldb -lsnappy -llapack_atlas -lcblas -latlas -lboost_system -lboost_thread 

# NOTE only can install one version of caffe at a time. 
install_train: 
	ln -f  $(STAGE_BIN_PATH)/caffeTRAIN$v$t /c/AMD/MLopen/caffe/build/tools/caffe

install_infer: 
	ln -f  $(STAGE_BIN_PATH)/caffeINFER$v$t /c/AMD/MLopen/caffe/build/tools/caffe


run:
	dotl.aix > /dev/pts/1

runsilent:
	dotl.aix > /dev/null

# end 


