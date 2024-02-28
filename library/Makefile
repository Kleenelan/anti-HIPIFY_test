
LIB := libbitla.so

INC := -I./src -I./src/include -I../common/include -I/opt/rocm/include

LD_FLAGS := -L /opt/rocm/lib -lrocblas -lamdhip64 -lrocsparse

LIB_SRC_CPP := 	src/lapack/roclapack_geqrf.cpp     	\
				src/auxiliary/rocauxiliary_larf.cpp	\
				src/auxiliary/rocauxiliary_larfg.cpp \
				src/auxiliary/rocauxiliary_larft.cpp \
				src/auxiliary/rocauxiliary_lacgv.cpp \
				src/common/rocsolver_logger.cpp		\
				src/common/buildinfo.cpp			\
				../common/src/common_host_helpers.cpp

LIB_OBJ_CPP := $(LIB_SRC_CPP:.cpp=.o)# $(CU_SRC:.cu=.o)

HIPCC_FLAGS := -DFMT_HEADER_ONLY=1 -DOPTIMAL -DROCBLAS_INTERNAL_API -DROCM_USE_FLOAT16 -DROCSOLVER_LIBRARY -DROCSOLVER_VERIFY_ASSUMPTIONS -DUSE_PROF_API=1 -D_FORTIFY_SOURCE=1 -D_GLIBCXX_ASSERTIONS -D__HIP_HCC_COMPAT_MODE__=1 -D__HIP_PLATFORM_AMD__=1 -Drocsolver_EXPORTS

%.o: %.cpp
	hipcc -g -fPIC $(HIPCC_FLAGS)  -std=c++17 $< $(INC) -c -o $@

$(LIB): $(LIB_OBJ_CPP)
	g++ -g $^ -shared   $(LD_FLAGS) -o $@


.PHONY: clean
clean:
	${RM} $(LIB) $(LIB_OBJ_CPP)
