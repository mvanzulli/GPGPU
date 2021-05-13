#include "util.h"
#include "CImg.h"

using namespace cimg_library;

void blur_cpu(float * image_in, int width, int height, float * image_out, float * mask, int m_size);
void blur_gpu(float * image_in, int width, int height, float * image_out, float * mask, int m_size, int threadPerBlockx, int threadPerBlocky);
void ajustar_brillo_cpu(float * img_in, int width, int height, float * img_out, float coef);
void ajustar_brillo_gpu(float * img_in, int width, int height, float * img_out, float coef, int coalesced, int threadPerBlockx, int threadPerBlocky);
    
int main(int argc, char** argv){


	const char * path;
	int coalesced;
	const char * no_coalesced;


	if (argc < 5) {
		printf("Se deben ingresar 4 parámetros:\n 1) la ruta del archivo que contiene la imagen a procesar\n 2) 1 o 0 dependiendo si se quiere un acceso coalesced o no coalesced para la función ajustar_brillo_gpu\n 3) el número de threads por bloque en la dirección x\n 4) el número de threads por bloque en la dirección y\n");
		exit(1);
	} 
	else
		path = argv[argc-4];

	int threadPerBlockx = atoi(argv[3]);
	int threadPerBlocky = atoi(argv[4]);

	if (atoi(argv[2]) == 1) {
		coalesced = 1;
	} else if (atoi(argv[2]) == 0) {
		coalesced = 0;
	}

    //inicializamos la mascara
    float mascara[25]={1, 4, 6, 4, 1,
						4,16,24,16, 4,
						6,24,36,24, 6,
						4,16,24,16, 4,
						1, 4, 6, 4, 1};

	CImg<float> image(path);
	CImg<float> image_out(image.width(), image.height(),1,1,0);

	float *img_matrix = image.data();
    float *img_out_matrix = image_out.data();

	float elapsed = 0;

	// ajustar_brillo_cpu(img_matrix, image.width(), image.height(), img_out_matrix, 100);

	ajustar_brillo_gpu(img_matrix, image.width(), image.height(), img_out_matrix, 100, coalesced, threadPerBlockx, threadPerBlocky);
   	image_out.save("output_brillo.ppm");

	 blur_cpu(img_matrix, image.width(), image.height(), img_out_matrix, mascara, 5);
   	image_out.save("output_blur_CPU.ppm");

	 blur_gpu(img_matrix, image.width(), image.height(), img_out_matrix, mascara, 5, threadPerBlockx, threadPerBlocky);
   	image_out.save("output_blur_GPU.ppm");
   	
    return 0;
}

