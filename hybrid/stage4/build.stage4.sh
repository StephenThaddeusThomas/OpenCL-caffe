#!/bin/bash
g++ obj/caffe.stage4.o -o caffe-stage4 \
-std=c++11 \
../lib/libcaffe.stage4.so   \
../lib/libadnnback.a  \
../lib/libaDNN.a       \
../lib/libgtest.a       \
../lib/libgflags.a       \
../lib/libproto.a         \
/usr/local/lib/libglog.so  \
-lpthread \
-lprotobuf \
-lz -ldl -lm -llmdb -lleveldb -lsnappy \
/usr/local/lib/libglog.a \
/usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5_hl.so \
/usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5.so \
/c/AMD/AppSDK/lib/x86_64/libOpenCL.so \
/usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.9 \
/usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.9 \
/usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.9 \
/opt/clBLAS-2.3/lib64/libclBLAS.so \
-llapack_atlas \
-lcblas \
-latlas \
-lboost_system \
-lboost_thread 


