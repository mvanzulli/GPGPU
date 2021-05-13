Este programa se ejecuta en clusteruy con el siguiente comando:

```
make
sbatch launch_single.sh ./blur dirImage (booleano coalesced) numeroThredsX numeroThredsY
```

Donde `launch_single.sh` es el archivo de lanzamiento del gestor SLURM.

`dirImage` es la dirección de la imagen a aplicarle el blur e incremento de brillo.
 `booleano coalesced` es 1 o 0 dependiendo si se busca realizar el acceso calesced o no.
 `numeroThredsX` numero de threades por bloque en x 
 `numeroThredsY` numero de threades por bloque en y

Para la ejecución local es análogo pero se debe verificar la arquietctura en el make en la línea 4 _50 o _60. Se ejecuta con:

```
make
./blur dirImage (booleano coalesced) numeroThredsX numeroThredsY
```
