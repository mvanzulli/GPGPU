#include "util.h"
#include "CImg.h"

using namespace cimg_library;

void transpose_cpu (float * image_in, int width, int height, float * image_out);
void transpose_gpu (float * image_in, int width, int height, float * image_out, int threadPerBlockx, int threadPerBlocky);
void blur_cpu (float * image_in, int width, int height, float * image_out, float * mask, int m_size);
void blur_gpu (float * image_in, int width, int height, float * image_out, float * mask, int m_size, int threadPerBlockx, int threadPerBlocky);


int main(int argc, char** argv) {

	const char * path;
	int n_mask;

	if (argc < 4) {
		printf("Se deben ingresar 3 parámetros:\n 1) la ruta del archivo que contiene la imagen a procesar\n 2) el número de threads por bloque en la dirección x\n 3) el número de threads por bloque en la dirección y\n");
		exit(1);
	} 
	else
		path = argv[argc-3];

	int threadPerBlockx = atoi(argv[2]);
	int threadPerBlocky = atoi(argv[3]);
	CImg<float> image(path);
	CImg<float> image_out(image.width(), image.height(),1,1,0);
	CImg<float> image_out_t(image.height(), image.width(),1,1,0);

	float *img_matrix = image.data();
    float *img_out_matrix = image_out.data();
    float *img_out_matrix_t = image_out_t.data();

	float elapsed = 0;

	printf("Image dimensions:\n");
    printf("width: %d px\n", image.width());
    printf("height: %d px\n", image.height());
    printf("\n");

	//Defino la mascara
    float mascara[25]={1, 4, 6, 4, 1,
						4,16,24,16, 4,
						6,24,36,24, 6,
						4,16,24,16, 4,
						1, 4, 6, 4, 1};
	n_mask =  sqrt( sizeof(mascara) / sizeof(mascara[0]));

	// Transpose - Ejercicio 1
	transpose_cpu(img_matrix, image.width(), image.height(), img_out_matrix_t);
	image_out_t.save("output_transpose_cpu.ppm");
	
	transpose_gpu(img_matrix, image.width(), image.height(), img_out_matrix_t, threadPerBlockx, threadPerBlocky);
	image_out_t.save("output_transpose_gpu.ppm");

	// Blur - Ejercicio 2
	blur_cpu(img_matrix, image.width(), image.height(), img_out_matrix, mascara, n_mask);
	image_out.save("output_blur_CPU.ppm");

	blur_gpu(img_matrix, image.width(), image.height(), img_out_matrix, mascara, n_mask, threadPerBlockx, threadPerBlocky);
	image_out.save("output_blur_GPU.ppm");

    return 0;
}

