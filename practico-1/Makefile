all: clean
	gcc	\
	\
	-march=native \
	-O3 \
	-mtune=native \
	-pg \
	\
	main.c \
	sum.c	\
	product.c	\
	aux.c	\
	\
	-o main
	
clean:
	rm -f main