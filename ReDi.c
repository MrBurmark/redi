/*************************

   File: ReDi.c
   Compile: g++ ReDi.c ReDiUtils.c -O3 -o ReDi -lm
   Use: ./ReDi [input file]

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

int main(int arg, char **argv) {
  int width;
  int numCycles;
  int i, j, n;
  double *u0, *u1, *tptr, *v0, *v1;
  double inTemp;
  int cycle = 0;
  int numInit;

  FILE *fp;

  struct timeval tv;
  gettimeofday(&tv, NULL);
  double t0 = tv.tv_sec*1e6 + tv.tv_usec;

  fp = fopen(argv[1], "r");

  fscanf(fp, "%d", &numCycles);
  fscanf(fp, "%d", &width);
  fscanf(fp, "%d", &numInit);
  printf("# cycles %d width %d # initializations %d\n", numCycles, width, numInit);

  u0 = (double *) calloc(width * width, sizeof(double));
  u1 = (double *) calloc(width * width, sizeof(double));
  v0 = (double *) calloc(width * width, sizeof(double));
  v1 = (double *) calloc(width * width, sizeof(double));

  initGrid(u0, u1, v0, v1, width);

  for (n=0; n<numInit; n++) {
    fscanf(fp, "%d%d%lf", &i, &j, &inTemp);
    dataAt(u1, i, j, width) = inTemp;
    dataAt(v1, i, j, width) = inTemp;
  }
  
  for (cycle=0; cycle<numCycles; cycle++) {
    updateGrid(u0, u1, v0, v1, width);
    tptr = u0;
    u0 = u1;
    u1 = tptr;
    tptr = v0;
    v0 = v1;
    v1 = tptr;
  }

  gettimeofday(&tv, NULL);
  double t1 = tv.tv_sec*1e6 + tv.tv_usec;

  printf("Elapsed time = %f\n", t1-t0);
  dumpGrid(u1, v1, width);

}

