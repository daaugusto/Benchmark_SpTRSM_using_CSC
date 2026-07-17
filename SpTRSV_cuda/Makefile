#compilers
CC=nvcc

#GLOBAL_PARAMETERS
VALUE_TYPE = double

#CUDA_PARAMETERS
NVCC_FLAGS = -O3 -w -m64 -Xptxas -dlcm=cg -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_89,code=compute_89

#ENVIRONMENT_PARAMETERS
CUDA_INSTALL_PATH = /usr/lib/cuda

#includes
INCLUDES = -I$(CUDA_INSTALL_PATH)/include

#libs
#CLANG_LIBS = -stdlib=libstdc++ -lstdc++
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib64  -lcudart -lcusparse
LIBS = $(CUDA_LIBS)

#options
#OPTIONS = -std=c99

make:
	$(CC) $(NVCC_FLAGS) main.cu -o sptrsv $(INCLUDES) $(LIBS) $(OPTIONS) -D VALUE_TYPE=$(VALUE_TYPE)
