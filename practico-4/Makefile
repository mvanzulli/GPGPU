all: clean
	nvcc \
	\
	-arch=sm_50 \
	-Xptxas \
	-dlcm=ca \
	\
	main.cpp \
	transpose_cpu.cpp \
	transpose_gpu.cu \
	blur_gpu.cu \
	blur_cpu.cpp \
	\
	-o main \
	\
	-O3 \
	-L/usr/X11R6/lib \
	-lm \
	-lpthread \
	-lX11

clean: 
	rm -f blur
