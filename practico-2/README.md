Este programa se ejecuta en clusteruy con el siguiente comando:

```
sbatch launch_single.sh ./decrypt secreto.txt nb parte
```

Donde `launch_single.sh` es el archivo de lanzamiento del gestor SLURM.

`secreto.txt` es el texto a desencriptar, `nb` es el tamaño del bloque, es decir la máxima cantidad de threads por bloque, y `parte` es la parte del ejercicio 1 del práctico 2 a ejecutar (1, 2 o 3).
