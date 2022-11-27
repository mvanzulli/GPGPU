# GPGPU
A repository with files for dydactical propuse to learn CUDAC Grphic Programing for General
 Propuse Units. This work was carried out during 2021 in [GPGPU FING course](https://eva.fing.edu.uy/course/view.php?id=1076)
 framework. Inside each assignment folder folder is a description of the probelm solved and running route writted up.

##  Práctico 1: CPU manejo y acceso a memoria.

### Letra:

<details>
<summary>Ejercicio 1: Acceso coalseced suma de matrices </summary>


- Letra:
    - Defina una matriz N×N como un arreglo estático de tipo double, para N=128, 256 y 512.

        a) Implemente una función double ``suma_est_fil(double A[N][N])`` que suma todos los elementos
de una matriz, recorriéndola por filas.

        b) Implemente una función ``double suma_est_col(double A[N][N])`` que suma todos los elementos
de una matriz, recorriéndola por columnas.

    - Defina una matriz N×N como un arreglo dinámico de tipo double para N=1024, 2048 y 4096

        a) Implemente una función double `suma_din_fil(double * A)` que suma todos los elementos de
una matriz, recorriéndola por filas.

        b) Implemente una función double `suma_din_col(double * A)` que suma todos los elementos de
una matriz, recorriéndola por columnas.

        c) Implemente una función double `suma_din_rand(double * A)` que suma todos los elementos de
una matriz, recorriéndola de forma aleatoria.

    - Mida los tiempos de ejecución de cada variante y analice las diferencias. ¿A qué se deben?


</details>


<details>
<summary>Ejercicio 2: Uso de memoria caché en multupicación de matrices </summary>

- Letra:

    - Construya la versión de la multiplicación de matrices ``int mul_simple(const double * __restrict__ A, const double * __restrict__ B, const double * __restrict__ C, size_t N)`` con el patrón de acceso usual, es decir, computando todas las operaciones correspondientes a una entrada c_ij antes de avanzar a la siguente.

    - Construya otra versión `int mul_fila(const double * __restrict__ A, const double * __restrict__ B, const double * __restrict__ C, size_t N)` que acceda por fila tanto a la matriz A como a la matriz B.

    - Construya dos versiones que multipliquen ambas matrices “por bloques”, donde la recorrida de cada bloque se realice:

    - `int mul_bl_simple(const double * __restrict__ A,const double * __restrict__ B,const double * __restrict__ C, size_t N, size_t BL_SZ `siguiendo el patrón de acceso usual.

    - `int mul_bl_fil(const double * __restrict__ A,const double * __restrict__ B,const double * __restrict__ C, size_t N, size_t BL_SZ)` accediendo a los valores por fila en todos los bloques.

- Obtenga los tamaños de las memorias caché de su plataforma experimental. En linux puede usar el comando `getconf -a | grep CACHE`. Luego mida el tiempo de ejecución y el dsempeño en GFlops (considerando que una multiplicación seguida de una suma es una única operación de punto flotante) para cada variante y los siguientes tamaños de matriz y bloque:

    a) A, B y C N×N tal que las 3 matrices entren en la caché L1.

    b) A, B y C N×N tal que las 3 matrices no entren en la caché L1 pero si en L2.

</details>

### Solución:
- [Scripts](https://github.com/mvanzulli/GPGPU/tree/main/practico-1/Tex)
- [Informe](https://github.com/mvanzulli/GPGPU/tree/main/practico-1/)


##  Práctico 2: Desencriptando con GPUs.

### Letra:

<details>
<summary>Ejercicio 1: desencriptando 100 años de soledad  </summary>

El texto que se encuentra en el archivo secreto.txt ha sido encriptado sustituyendo cada
caracter mediante una función definida como
$$
E(x) = (Ax + B) mod M (1)
$$
donde A y B son las claves del cifrado, mod es el operador de módulo, y A y M son co-primos.

Para este ejercicio el valor de A =15, B =27 y M =256 (la cantidad de caracteres en la tabla
ASCII extendida). La función de desencriptado se define como:
$$
D(x) = A^{−1} (x − B) mod M (2)
$$

Donde $A^{−1}$ es el inverso multiplicativo de A módulo M. En este caso $A^{−1}$ = −17.
Como cada caracter puede ser encriptado y desencriptado de forma independiente podemos
utilizar la GPU para desencriptar el texto en paralelo. Para esto debemos lanzar un kernel
que realice el desencriptado, asignando un thread por caracter. Usando como base `ej1.cu`:

- Parte a)

    1. Implementar el kernel decrypt_kernel utilizando un solo bloque de threads (en la dimensión x).
    A,B y M están definidas en el código como macros.

    2. Reservar memoria en la GPU para el texto a desencriptar.

    3. Copiar el texto a la memoria de la GPU.

    4. Configurar la grilla de threads con 1 bloque de `n` threads (tamaño de bloque a elección) e invocar el
kernel.

    5. Transferir el texto a la memoria de la CPU para desplegarlo en pantalla.


- Parte b)

    1.  Modificar el código del kernel y la definición de la grilla de threads para que utilice varios bloques,
procesando textos de largo arbitrario.

- Parte c)

    Modificar el código del kernel y la definición de la grilla de threads para que utilice una cantidad fija de
bloques (por ejemplo 128), para procesar textos de largo arbitrario.


</details>

<details>
<summary>Ejercicio 2: manejo de memoria shaerd y global </summary>

 - Implemente una función que tome como entrada el texto desencriptado y genere un vector de tamaño
256 con la cantidad de ocurrencias de cada caracter en el texto. El conteo de la cantidad de ocurrencias debe
realizarse en la GPU, utilizando un arreglo en memoria global, y el resultado debe quedar en la memoria de
la CPU. Recuerde que si dos hilos acceden concurrentemente a la misma posición de un vector y al menos
uno de ellos escribe, debe utilizarse algún mecanismo para evitar una condición de carrera.</details>
</details>


### Solución:
- [Scripts](https://github.com/mvanzulli/GPGPU/tree/main/practico-2/Tex)
- [Informe](https://github.com/mvanzulli/GPGPU/tree/main/practico-2/)


##  Práctico 3: Filtrando imágenes con GPUs.

### Letra:

<details>
<summary>Ejercicio 1: ajuste de brillo  </summary>

La función `ajustar_brillo_cpu(...)` recorre la imagen sumando un coeficiente entre -255 y 255a cada píxel, aumentando o reduciendo su brillo.

- Parte a)
Construir la función `ajustar_brillo_gpu(...)` y los kernels correspondientes
    para realizar esta tarea en la GPU. Configure adecuadamente la grilla (bidimensional) de threads para aceptar matrices
    de cualquier tamaño. En el kernel, utilice las variables `blockIdx` y `threadIdx`
    adecuadamente para acceder a la estructura bidimensional. La recorrida de la imagen
    debe realizarse en dos versiones:

    1. En `ajustar_brillo_coalesced_kernel`, threads con índice consecutivo en la dirección x deben
    acceder a pixels de una misma fila de la imagen-. En `ajustar_brillo_no_coalesced_kernel`,
    threads con índice consecutivo en la dirección `x` deben acceder a pixels de una misma
    columna de la imagen.

- Parte b) Registre los tiempos de cada etapa de la función ajustar_brillo_gpu
    (reserva de memoria, transferencia dedatos, ejecución del kernel, etc.) para las dos
    variantes usando los dos mecanismos de medición de tiempo provistos (CUDA events y la función gettimeofday).
    Compare los tiempos de ejecución entre sí y con los que se obtienen mediante el comando
    `nvprof ./blur imagen.pgm.` Explique las diferencias.

- Parte c) Utilizando `nvprof --metrics gld_efficiency ./blur imagen.ppm` obtenga el valor de la
     métrica gld_efficiency para ambos kernels y analice los resultados (
        ver sección 9.2.1 de https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory).

</details>

<details>
<summary>Ejercicio 2: filtro gausiano  </summary>
Construiremos un filtro Gaussiano para reducir el ruido de una imagen en escala de grises.
Consiste en sustituir el valor de intensidad de cada pixel por un promedio ponderado de los
 pixels vecinos. Los pesos por los cuales se pondera cada vecino en el promedio se almacenan
  en una matriz cuadrada (máscara).

- Parte a) Construya el kernel que aplica el filtro en la GPU y la función que invoca dicho kernel y
 transfiere los datos correspondientes hacia y desde la GPU.

- Parte b) Registre los tiempos de cada etapa de la función y compare el tiempo de ejecución las
 variantes de CPU y GPU.

</details>


### Solución:
- [Scripts](https://github.com/mvanzulli/GPGPU/tree/main/practico-3/Tex)
- [Informe](https://github.com/mvanzulli/GPGPU/tree/main/practico-3/)


##  Práctico 4: Manejo de memoria shared en procesamiento de imágenes.

### Letra:

<details>
<summary>Ejercicio 1: rotación de una imágen  </summary>

Este ejercicio pretende ilustrar el uso de la memoria compartida de la GPU para evitar el acceso no
coalesced a memoria global.

El resultado de transponer la matriz de las intensidaes es el resultado de rotar imagen 90 grados
en sentido horario y luego reflejarla horizontalmente (o rotar 90 grados antihorario y reflejar verticalmente).

Una forma sencilla de resolver el problema consiste en leer una fila de la matriz de la memoria,
para luego escribir dicha fila como una columna de la matriz transpuesta.
En el contexto de la GPU, pueden lanzarsetantos threads como elementos haya en la matriz,
y cada thread puede encargarse de leer un elemento de la matriz y escribirlo en la posición
correspondiente de la matriz resultado.

- Parte a)  Construir un programa que obtenga la matriz transpuesta de una matriz de entra-da almacenada en la memoria de la GPU utilizando únicamente la memoria global del dispositivo.

- Parte b) Utilizando nvprof registre el tiempo del kernel y las métricas `gld_efficiency` y
    `gst_efficiency`. explicando el resultado


</details>

<details>

<summary>Ejercicio 2: mejorando el acceso a memoria  </summary>

Para mejorar el patrón de acceso a la memoria podemos utilizar la memoria compartida. Dividiremos
conceptualmente la matriz en tiles bi-dimensionales (cuadrados) de tamaño igual al tamaño de bloque elegido
en la configuración de la grilla y luego se realiza la operación en tres pasos:

1. Cada bloque de threads carga un tile de la matriz a memoria compartida leyendo sus elementos por
fila de forma coalesced.

1. Los threads de cada warp (`threadIdx.x` contiguo) leerán una columna del tile almacenado en memoria
compartida.

1. Luego los threads de cada warp (`threadIdx.x` contiguo) realizan la escritura de dicha columna como
(parte de) una fila de la matriz de salida en memoria global de forma coalesced.

1. Construya el kernel que aplica el filtro en la GPU y la función que invoca dicho kernel y
 transfiere los datos correspondientes hacia y desde la GPU.

1. Registre los tiempos de cada etapa de la función y compare el tiempo de ejecución las
 variantes de CPU y GPU.

- Parte a) Modificar el código de la parte anterior para utilizar la memoria compartida
siguiendo el procedimiento que fue explicado anteriormente.

- Parte b) Utilizando nvprof registre el tiempo del kernel y las métricas `gld_efficiency` y
`gst_efficiency`. Compare con los resultados obtenidos en la parte anterio


</details>


<details>

<summary>Ejercicio 3: solución de conflictos de bancos  </summary>


Para maximizar la eficiencia en el acceso a la memoria compartida, la misma se divide en 32 bancos con
un ancho de 32 bits. Las palabras de 32 bits de un arreglo almacenado en memoria compartida se distribuirán
secuencialmente entre los bancos, de manera que si los 32 threads de un warp acceden a palabras contiguas,
las mismas pueden accederse en paralelo1. Sin embargo, si dos threads de un mismo warp acceden a palabras
distintas correspondientes a un mismo banco, ocurre un conflicto de bancos y dicho acceso se serializa


- Parte a) Modificar el código de la parte a) para utilizar la memoria compartida siguiendo el
procedimiento que fue explicado anteriormente.

- Parte b) Utilizando nvprof registre el tiempo del kernel y las métricas gld_efficiency y
gst_efficiency. Compare con los resultados obtenidos en la parte anterior.

</details>

### Solución:
- [Scripts](https://github.com/mvanzulli/GPGPU/tree/main/practico-4/Tex)
- [Informe](https://github.com/mvanzulli/GPGPU/tree/main/practico-4/)


### Archivos:

```
├── Apuntes
│   ├── Apuntes programación CUDA.aux
│   ├── Apuntes programación CUDA.bbl
│   ├── Apuntes programación CUDA.blg
│   ├── Apuntes programación CUDA.log
│   ├── Apuntes programación CUDA.pdf
│   ├── Apuntes programación CUDA.synctex.gz
│   └── Apuntes programación CUDA.tex
├── practico-1
│   ├── aux.c
│   ├── aux.h
│   ├── bench.h
│   ├── gmon.out
│   ├── main
│   ├── main.c
│   ├── Makefile
│   ├── product.c
│   ├── product.h
│   ├── README.md
│   ├── sum.c
│   ├── sum.h
│   ├── template
│   └── Tex
│       ├── Cache.pdf
│       ├── ciclosperinstructionsYL1cachemisesesVSn.png
│       ├── logo_FING.jpg
│       ├── logo_UdelaR.png
│       ├── main.aux
│       ├── main.log
│       ├── main.out
│       ├── main.pdf
│       ├── main.tex
│       ├── Ncubo.png
│       ├── tiempoYperformance.mult_bl_fila.png
│       ├── tiempoYperformance.mult_bl_simple.png
│       ├── tiempoYperformanceVSn.png
│       ├── t_mul_bl_fila.pdf
│       ├── t_mul_bl_simple.pdf
│       ├── t_mul_fila.pdf
│       └── t_mul_simple.pdf
├── practico-2
│   ├── decrypt
│   ├── decrypt.cu
│   ├── launch_single.sh
│   ├── letra.pdf
│   ├── Makefile
│   ├── output.out
│   ├── output.txt
│   ├── README.md
│   └── secreto.txt
├── practico-3
│   ├── blur
│   ├── blur.cu
│   ├── bright.cu
│   ├── CImg.h
│   ├── img
│   │   ├── fing1.jpg
│   │   ├── fing1.pgm
│   │   ├── fing1_ruido.pgm
│   │   └── lena.pgm
│   ├── launch_single.sh
│   ├── letra.pdf
│   ├── main.cpp
│   ├── Makefile
│   ├── output_blur_CPU.ppm
│   ├── output_blur_GPU.ppm
│   ├── output_brillo.ppm
│   ├── README.md
│   └── util.h
├── practico-4
│   ├── blur_cpu.cpp
│   ├── blur_gpu.cu
│   ├── CImg.h
│   ├── img
│   │   ├── fing1.pgm
│   │   └── lena.pgm
│   ├── launch_single.sh
│   ├── letra.pdf
│   ├── main
│   ├── main.cpp
│   ├── Makefile
│   ├── output_blur_CPU.ppm
│   ├── output_blur_GPU.ppm
│   ├── output_transpose_cpu.ppm
│   ├── output_transpose_gpu.ppm
│   ├── transpose_cpu.cpp
│   ├── transpose_gpu.cu
│   └── util.h
└── README.md
```

## Issues
Si hay problemas siemplente [click y abrir un issue](https://github.com/mvanzulli/GPGPU/issues/new).
