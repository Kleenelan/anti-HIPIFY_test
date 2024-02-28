EXE := rocsolverDgeqrfExe
 
all: $(EXE)
 
INC := -I/opt/rocm/include   -D__HIP_PLATFORM_AMD__
#LD_FLAGS := -L/home/hipper/ex_transplant_roc_2_ix/rocSOLVER/build/debug/rocsolver-install/lib -lamdhip64 -lrocsolver -L/opt/rocm/lib -lrocblas
LD_FLAGS := -L../../library  -lbitla -L/opt/rocm/lib -lrocblas -lamdhip64
 
$(EXE).o: $(EXE).cpp
	g++ -g $< $(INC) -c -o $@
 
$(EXE): $(EXE).o
	g++ -g $< $(LD_FLAGS) -o $@
 
 
.PHONY: clean
clean:
	${RM} *.o $(EXE)
