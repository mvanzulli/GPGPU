all: clean
	nvcc \
	\
	-arch=sm_50 \
	-Xptxas \
	-dlcm=cg \
	\
	main.cpp \
	blur.cu \
	bright.cu \
	\
	-o blur \
	\
	-O3 \
	-L/usr/X11R6/lib \
	-lm \
	-lpthread \
	-lX11

clean: 
	rm -f blur
