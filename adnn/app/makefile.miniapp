############################################################################################################
# /c/AMD/MLopen/aDNN/app/makefile.miniapp (link to makefile for easy)
# date: 160227
# HAD TO CHANGE the Include and Library to the repo  (see /c/AMD/MLopen/BufferList*refactoring)
# The repo was copied from thaddeus-nn (the beast) from build made Feb 26 on Linux
# thaddeus-nn:/repo/stt/apcLibs --> TT:/c/AMD/MLopen/apcLibs
############################################# Linux GCC Environment ########################################
# gcc --version
# gcc (Ubuntu/Linaro 4.4.4-14ubuntu5) 4.4.5
# Linux man page for ptheads
#   Compiling on Linux
#       On Linux, programs that use the Pthreads API should be compiled using cc -pthread.
############################################################################################################
OS=linux
app=miniapp
v=.1
t=.lnx

ROOT=/c/AMD
PROJ=MLopen/adnn

LOCAL_PATH=$(ROOT)/$(PROJ)
LOCAL_BIN_PATH = $(LOCAL_PATH)/bin
LOCAL_LIB_PATH = $(LOCAL_PATH)/lib
LOCAL_SRC_PATH = $(LOCAL_PATH)/src
LOCAL_INC_PATH = $(LOCAL_PATH)/inc
COMMON_INC_PATH= $(ROOT)/Include
AMDSDK_INC_PATH= $(COMMON_INC_PATH)/AMDSDK

LOCAL_HOST=TT
REMOT_HOST=thaddeus-nn
REMOTE_REPO_PATH = thaddeus-nn:/repo/stt
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
# do NOT use  -fno-implicit-templates (but see above)
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
I= -I$(LOCAL_INC_PATH) -I$(AMDSDK_INC_PATH)
L_OPT = -O2 -s
L_DBG = -g 
L= -L$(LOCAL_LIB_PATH)
K= g++ -m64 -shared-libgcc -o
P= -fprofile-arcs
R= -fno-rtti
X= -Q -H

#################################### COMPONENTS/OBJECTS ###########################################

miniapp_main.o            : miniapp_main.cpp	   		; $C $E $F $G $I $<
RunSingleConvLayerInfer.o : RunSingleConvLayerInfer.cpp		; $C $E $F $G $I $<
RunSingleConvLayerTrain.o : RunSingleConvLayerTrain.cpp		; $C $E $F $G $I $<
PrepareConvNode.o         : PrepareConvNode.cpp 		; $C $E $F $G $I $<

#################################### Target and Parent OBJS ########################################

# First Version is for a Foward Only Single Convolutions
app_obj  = miniapp_main.o RunSingleConvLayerInfer.o PrepareConvNode.o 

miniapp : $(app_obj)
	$K $(app)$v$t $E $F $G $(app_obj) $I $L -laDNN -lclBLAS -lm -lOpenCL 

likenew     : clean fresh lib

clean:
	/bin/rm -f *.o *.rpo 
	/bin/rm -f *~ 

fresh:
	clear; clear > /dev/pts/1

install: $(app)$v$t
#	mkdir $(LOCAL_BIN_PATH)
	ln $(app)$v$t $(LOCAL_BIN_PATH)/$(app)

run:
	$(LOCAL_BIN_PATH)/$(app) -bz 100 -li 200 -ni 3 -n -l -iw 32 -ih 32 -ic0 32 -oc0 64 -fs 3 -sd 1 -fw -cnv
# end 


