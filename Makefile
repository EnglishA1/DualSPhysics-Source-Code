#DualSPHysics MultiPhase GPU v3.0 (15-09-2015)

#=============== Compilation Options ===============
USE_DEBUG=NO
USE_FAST_MATH=YES
USE_NATIVE_CPU_OPTIMIZATIONS=NO

#=============== CUDA toolkit directory (make appropriate for local CUDA installation) ===============
DIRTOOLKIT=/usr/local/cuda
DIRTOOLKIT=/exports/opt/NVIDIA/cuda-5.5

#=============== Files to compile ===============
OBJ_BASIC=main.o Functions.o JCfgRun.o JException.o JFloatingData.o JLog2.o JObject.o JPartData.o JPartsOut.o JSpaceCtes.o JSpaceEParms.o JSpaceParts.o JSph.o JSphDtFixed.o JSphVarAcc.o JVarsAscii.o
OBJ_CPU_SINGLE=JPartsLoad.o
OBJ_GPU=JCellDivGpu.o JGpuArrays.o JObjectGpu.o JPeriodicGpu.o JPtxasInfo.o JSphGpu.o
OBJ_GPU_SINGLE=JCellDivGpuSingle.o JSphGpuSingle.o
OBJ_CUDA=JCellDivGpu_ker.o JSphGpu_ker.o JPeriodicGpu_ker.o
OBJ_CUDA_SINGLE=JCellDivGpuSingle_ker.o
OBJECTS=$(OBJ_BASIC) $(OBJ_CPU_SINGLE) $(OBJ_GPU) $(OBJ_CUDA) $(OBJ_GPU_SINGLE) $(OBJ_CUDA_SINGLE)

#=============== Select GPU architectures ===============
GENCODE:=$(GENCODE) -gencode=arch=compute_12,code=\"sm_12,compute_12\"
##GENCODE:=$(GENCODE) -gencode=arch=compute_13,code=\"sm_13,compute_13\"
GENCODE:=$(GENCODE) -gencode=arch=compute_20,code=\"sm_20,compute_20\"
##GENCODE:=$(GENCODE) -gencode=arch=compute_30,code=\"sm_30,compute_30\"
##GENCODE:=$(GENCODE) -gencode=arch=compute_35,code=\"sm_35,compute_35\"

#=============== DualSPHysics libs to be included ===============
ifeq ($(USE_DEBUG), NO)
  JLIBS=-L./ -ljxml_64 -ljformatfiles2_64 -ljsphmotion_64
else
  JLIBS=-L./ -ljxml_64_debug -ljformatfiles2_64_debug -ljsphmotion_64_debug
endif

#=============== CPU Code Compilation (make appropriate for chosen compiler) ===============
### with GCC compiler
CC=g++
ifeq ($(USE_DEBUG), NO)
  CCFLAGS=-c -O3 -D_WITHGPU
else
  CCFLAGS=-c -O0 -g -Wall -D_WITHGPU
endif
CCLINKFLAGS=

### with Intel C++ Compiler
##CC=icpc
##ifeq ($(USE_DEBUG), NO)
##  CCFLAGS=-c -c -O3 -openmp -D_WITHGPU
##else
##  CCFLAGS=-c -O0 -g -Wall -openmp -D_WITHGPU
##endif
##CCLINKFLAGS=-openmp

ifeq ($(USE_FAST_MATH), YES)
  CCFLAGS+= -ffast-math
endif
ifeq ($(USE_NATIVE_CPU_OPTIMIZATIONS), YES)
  CCFLAGS+= -march=native
endif

#=============== GPU Code Compilation ===============
CCFLAGS := $(CCFLAGS) -I./ -I$(DIRTOOLKIT)/include
CCLINKFLAGS := $(CCLINKFLAGS) -L$(DIRTOOLKIT)/lib64 -lcudart
NCC=nvcc
ifeq ($(USE_DEBUG), NO)
  NCCFLAGS=-c $(GENCODE) -O3
else
  NCCFLAGS=-c $(GENCODE) -O0 -g
endif
ifeq ($(USE_FAST_MATH), YES)
  NCCFLAGS+= -use_fast_math
endif

all:DualSPHysics_linux64 
	rm -rf *.o
ifeq ($(USE_DEBUG), NO)
	@echo "  --- Compiled Release GPU version ---"
else
	@echo "  --- Compiled Debug GPU version ---"
	mv DualSPHysics_linux64 DualSPHysics_linux64_debug
	mv DualSPHysics_linux64_ptxasinfo DualSPHysics_linux64_debug_ptxasinfo
endif

DualSPHysics_linux64:  $(OBJECTS)
	$(CC) $(OBJECTS) $(CCLINKFLAGS) -o $@ $(JLIBS)

.cpp.o: 
	$(CC) $(CCFLAGS) $< 

JSphGpu_ker.o: JSphGpu_ker.cu
	$(NCC) $(NCCFLAGS) --ptxas-options -v JSphGpu_ker.cu 2>DualSPHysics_linux64_ptxasinfo

JCellDivGpu_ker.o: JCellDivGpu_ker.cu
	$(NCC) $(NCCFLAGS) JCellDivGpu_ker.cu

JCellDivGpuSingle_ker.o: JCellDivGpuSingle_ker.cu
	$(NCC) $(NCCFLAGS) JCellDivGpuSingle_ker.cu

JCellDivGpuMpi_ker.o: JCellDivGpuMpi_ker.cu
	$(NCC) $(NCCFLAGS) JCellDivGpuMpi_ker.cu

JPeriodicGpu_ker.o: JPeriodicGpu_ker.cu
	$(NCC) $(NCCFLAGS) JPeriodicGpu_ker.cu

clean:
	rm -rf *.o DualSPHysics_linux64 DualSPHysics_linux64_ptxasinfo DualSPHysics_linux64_debug DualSPHysics_linux64_debug_ptxasinfo

