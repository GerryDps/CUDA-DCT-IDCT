NVCC = nvcc
ARCH = -arch=sm_75
LIBS = --library=cublas --library=jpeg
MAINS = main_cublass main_cublass_2 main_fastAppr main_newAppr

all: $(MAINS)

%: %.o utils.o utils_kernels.o
	$(NVCC) $(ARCH) $^ -o $@ $(LIBS)

main_cublass.o: main_cublass.cu utils.cuh utils_kernels.cuh
	$(NVCC) $(ARCH) -c $< -o $@

main_cublass_2.o: main_cublass_2.cu utils.cuh utils_kernels.cuh
	$(NVCC) $(ARCH) -c $< -o $@

main_fastAppr.o: main_fastAppr.cu utils.cuh utils_kernels.cuh
	$(NVCC) $(ARCH) -c $< -o $@

main_newAppr.o: main_newAppr.cu utils.cuh utils_kernels.cuh
	$(NVCC) $(ARCH) -c $< -o $@

utils.o: utils.cu utils.cuh
	$(NVCC) $(ARCH) -c $< -o $@ --library=jpeg

utils_kernels.o: utils_kernels.cu utils_kernels.cuh
	$(NVCC) $(ARCH) -c $< -o $@

clean:
	rm -f *.o $(MAINS)