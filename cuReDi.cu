/*************************

   File: ReDi.c
   Compile: nvcc cuReDi.cu ReDiUtils.c -O3 -o cuReDi -lm -fmad=false -arch=compute_20 -code=sm_20,sm_30,sm_35
   Use: ./cuReDi [input file]

   Performs Gray-Scott Reaction-Diffusion on a 2D grid
   Input file format:

   # cycles
   width of grid (including boundary)
   # initial data points
   
   3 integers per data point: i and j indices, data


*************************/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "ReDi.h"
#include "ReDiUtils.c"
#include "cuReDiUtils.cu"

__global__ void cuUpdateGridGlobal(double *u, double *tu, double *v, double *tv, int w);
__global__ void cuUpdateGridShared(double *u, double *tu, double *v, double *tv, int w);

int main(int argc, char **argv) {
  int width;
  int numCycles;
  int i, j, n, ok;
  double *u0, *u1, *v0, *v1;
  double *d_u0, *d_u1, *d_tptr, *d_v0, *d_v1;
  double inTemp;
  int cycle = 0;
  int numInit;

  FILE *fp;

  if (argc != 2) {
    printf("Use: ./cuReDi [input file]");
    exit(1);
  }

  struct timeval tv;
  gettimeofday(&tv, NULL);
  double t0 = tv.tv_sec*1e6 + tv.tv_usec;

  fp = fopen(argv[1], "r");

  ok = fscanf(fp, "%d", &numCycles);
  ok = fscanf(fp, "%d", &width);
  ok = fscanf(fp, "%d", &numInit);
#if !SHARED
  printf("Global - ");
#elif SHARED
  printf("Shared - ");
#endif
  printf("TileWidth %i - ", TILEWIDTH);
  printf("# cycles %d width %d # initializations %d\n", numCycles, width, numInit);

  u0 = (double *) calloc(width * width, sizeof(double));
  u1 = (double *) calloc(width * width, sizeof(double));
  v0 = (double *) calloc(width * width, sizeof(double));
  v1 = (double *) calloc(width * width, sizeof(double));

  cudaMalloc((void **)&d_u0, width * width * sizeof(double));
  cudaMalloc((void **)&d_u1, width * width * sizeof(double));
  cudaMalloc((void **)&d_v0, width * width * sizeof(double));
  cudaMalloc((void **)&d_v1, width * width * sizeof(double));

  initGrid(u0, u1, v0, v1, width);

  for (n=0; n<numInit; n++) {
    ok = fscanf(fp, "%d%d%lf", &i, &j, &inTemp);
    dataAt(u1, i, j, width) = inTemp;
    dataAt(v1, i, j, width) = inTemp;
  }

  cudaMemcpy(d_u0, u1, width * width * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_u1, u1, width * width * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v0, v1, width * width * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v1, v1, width * width * sizeof(double), cudaMemcpyHostToDevice);

  dim3 grid((width-2)/TILEWIDTH, (width-2)/TILEWIDTH, 1);
  dim3 block(TILEWIDTH, TILEWIDTH, 1);

  // printf("%i, %i\n", grid.x, grid.y);
  // printf("%i, %i\n", block.x, block.y);

  for (cycle=0; cycle<numCycles; cycle++) {
#if !SHARED
    cuUpdateGridGlobal<<< grid, block >>>(d_u0, d_u1, d_v0, d_v1, width);
#elif SHARED
    cuUpdateGridShared<<< grid, block >>>(d_u0, d_u1, d_v0, d_v1, width);
#endif
    d_tptr = d_u0;
    d_u0 = d_u1;
    d_u1 = d_tptr;
    d_tptr = d_v0;
    d_v0 = d_v1;
    d_v1 = d_tptr;
    cudaDeviceSynchronize();
  }

  // cudaMemcpy(u0, d_u0, width * width * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(u1, d_u1, width * width * sizeof(double), cudaMemcpyDeviceToHost);
  // cudaMemcpy(v0, d_v0, width * width * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(v1, d_v1, width * width * sizeof(double), cudaMemcpyDeviceToHost);


  gettimeofday(&tv, NULL);
  double t1 = tv.tv_sec*1e6 + tv.tv_usec;

  printf("Elapsed time = %f\n\n", t1-t0);
#if DUMP
  cuDumpGrid(u1, v1, width);
#endif

  cudaFree(d_u0);
  cudaFree(d_u1);
  cudaFree(d_v0);
  cudaFree(d_v1);
  free(u0);
  free(u1);
  free(v0);
  free(v1);
  ok++;
}

