# /c/AMD/MLopen/aDNN/src/makefile.lib
# date: 160228 aDNN-->adnn
# date: 160416
# from: /c/dotland/src/makefile link to Makefile.dotland.lnxos (version 8)
# gcc --version
# gcc (Ubuntu/Linaro 4.4.4-14ubuntu5) 4.4.5
# Linux man page for ptheads
#   Compiling on Linux
#       On Linux, programs that use the Pthreads API should be compiled using cc -pthread.
############################################# Linux GCC Environment ########################################

V=linux
v=.1
T=.lnx
l=.plst
H=TT
APP=aDNNlib
ROOT=/c/AMD
PROJ=MLopen/adnn
BIN=$(ROOT)/$(PROJ)/bin/
REMOTE_BIN_PATH = git-server:/c/bin
LOCAL_BIN_PATH = $(ROOT)/$(PROJ)/bin
LOCAL_LIB_PATH = $(ROOT)/$(PROJ)/lib
LOCAL_SRC_PATH = $(ROOT)/$(PROJ)/src
COMMON_INC_PATH= $(ROOT)/MLopen/incl
AMDSDK_INC_PATH= $(COMMON_INC_PATH)/APPSDK
COMMON_LIB_PATH= $(ROOT)/lib

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
# -fno-implicit-templates
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
# It may make more sense to use -O2 and -g2 -g1 -O1 etc with the $G option and save $F for warnings and error flags
# -Woverloaded-virtual-macros <<<-- unrecognized
C_DBG_0 = -w
C_DBG_1 = -Wctor-dtor-privacy -Wnon-virtual-dtor -Wreorder -fpermissive -Wno-deprecated -Wno-non-template-friend  -Wswitch-enum -Wswitch-default -Wcast-align 
C_DBG_2 = -Wextra -Wfatal-errors -Wmissing-include-dirs  -Wundef -Wunused -Wunused-function -Wunused-label -Wunused-parameter -Wunused-macros -Wlogical-op -Waggregate-return -Wredundant-decls -Wsequence-point -Wfloat-equal
# -Wtraditional
C_DBG_3 = -Wall
C_OPT   = -O2

F= $(C_DBG_1) -Wno-write-strings -Wfatal-errors -fPIC

# 151120 : Add 3DX_DRIVER - but ... <command-line>:0:1: error: macro names must be identifiers
# SOLUTION - can not start a macro with a NUMBER - so SPACEPILOT_3DX_DRIVER will work 
E= -DLNX -DVLP -time 
G= -DVERBOSE -fmessage-length=0 -g2 -ggdb 
I=  -I$(LOCAL_SRC_PATH) -I../ocl -I$(COMMON_INC_PATH)/aDNN -I$(AMDSDK_INC_PATH)
P= -fprofile-arcs
R= -fno-rtti
X= -Q -H

########################################################################################
# LINK 
# handcrafted using /v/d/GCC.help and /oe/Atlis/DomParser/dvlp/8.2/makefile.linux.2
########################################################################################

L_OPT = -O2 -s
L_DBG = -g 
L= -L$(LOCAL_LIB_PATH)

# tried: -no-shared and both -static and -shared-libgcc
# works: -static -static -m64 -o  but problem now is templates
# unsure: -fno-implicit-templates
# -Wl  <--- this was fucking things up
# --no-undefined
# -std=c++98
# -frepo
# -static
# -fPIC -m64  -rdynamic
# -static <<<--- with this we get: crtbeginT.o: relocation R_X86_64_32 against `__DTOR_END__
# The following works for dsxstub (dsxs) ...
# K=g++ -m64 -static-libgcc -std=c++98 -o
# Removing to be compliant to the video makefile : -m64 -static-libgcc -std=c++98
# Tried... -std=c++0x - nope.
K=g++ -m64 -shared-libgcc -o

#################################### Target and Parent OBJS ########################################

# Component Object Sets
common_include = 
common_object  = error.o dump.o # reporterror.o

# App is for driver
app_obj  = 
lib_obj  = aDNNConvLayer.o aDNNConvLayerWin.o aDNNFullyConnectLayer.o aDNNInternal.o aDNNLayer.o aDNNLRNLayer.o aDNNNeuronLayer.o\
aDNNPoolingLayer.o  aDNNSoftMaxLayer.o aLibDNNInternal.o AMDnnImpl.o AMDnnTensor.o  aDNNOCL.o

aDNNConvLayer.o 	: aDNNConvLayer.cpp 		; $C $E $F $G $I $<
aDNNConvLayerWin.o      : aDNNConvLayerWin.cpp 		; $C $E $F $G $I $<
#160221 Alex "This file is not in the solution, should not be here
#aDNN.o 			: aDNN.cpp 			; $C $E $F $G $I $<
aDNNFullyConnectLayer.o : aDNNFullyConnectLayer.cpp 	; $C $E $F $G $I $<
aDNNInternal.o          : aDNNInternal.cpp 		; $C $E $F $G $I $<
aDNNLayer.o        	: aDNNLayer.cpp 		; $C $E $F $G $I $<
aDNNLRNLayer.o     	: aDNNLRNLayer.cpp 		; $C $E $F $G $I $<
aDNNNeuronLayer.o  	: aDNNNeuronLayer.cpp 		; $C $E $F $G $I $<
aDNNOCL.o          	: aDNNOCL.cpp 			; $C $E $F $G $I $<
CABuf.o			: CABuf.cpp     		; $C $E $F $G $I -fno-rtti  $<  #-Weffc++ <--- Try
aDNNPoolingLayer.o 	: aDNNPoolingLayer.cpp 		; $C $E $F $G $I $<
aDNNSoftMaxLayer.o 	: aDNNSoftMaxLayer.cpp 		; $C $E $F $G $I $<
aLibDNNInternal.o  	: aLibDNNInternal.cpp 		; $C $E $F $G $I $<
#aLibDNNObj.o       	: aLibDNNObj.cpp 		; $C $E $F $G $I $<
AMDnnImpl.o        	: AMDnnImpl.cpp 		; $C $E $F $G $I $<
AMDnnTensor.o      	: AMDnnTensor.cpp  		; $C $E $F $G $I $<

T.o			: T.cpp     		; $C $E $F $G $I -fno-rtti  $<  #-Weffc++ <--- Try

#  -fpermissive

# The specific targets - add to all: line
lib : $(lib_obj)
	ar -v -r $(LOCAL_LIB_PATH)/libaDNN.a  $(lib_obj)
	ln -f $(LOCAL_LIB_PATH)/libaDNN.a $(COMMON_LIB_PATH)/libaDNN.a 

app : # TODO
	$K dot $E $F $G $(lib_obj) $I $L -laDNN -lm


# The main target (when make is run without an argument, is the first one detected, ie 'all')
likenew     : clean fresh lib

# utilities
install     : $(APP)$T
	  	strip $(APP)$T
		/bin/cp -p $(APP)$T $/usr/local/bin
## TODO need to install the AMDnn.h and AMDnn.hpp to /c/AMD/MLopen/incl/aDNN
## TODO need to install the library to /c/AMD/MLopen/lib or lib43 TBD Mikael and Alex

clean:
	/bin/rm -f *.o 
	/bin/rm -f *~	# emacs

fresh:
	clear; clear > /dev/pts/1

run:
	dotl.aix > /dev/pts/1

runsilent:
	dotl.aix > /dev/null

# end 


