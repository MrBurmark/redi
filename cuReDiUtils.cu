#include <stdio.h>
#include "ReDi.h"
#include <math.h>

/****

     tu[] is u input grid
     u[] is u output grid
     tv[] is v input grid
     v[] is v output grid
     w is width of grid

     compute 4-nearest neighbor updates 

****/

__global__ void cuUpdateGridGlobal(double *u, double *tu, double *v, double *tv, int w) {
  
  int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  double u1, v1, uv2;

  // printf("(%i, %i)", i, j);

  uv2 = dataAt(tu, i, j, w) * dataAt(tv, i, j, w) * dataAt(tv, i, j, w);

  // printf("yay");

  u1 = dataAt(tu, i, j, w)
    + 0.2 * (dataAt(tu, i+1, j, w) + dataAt(tu, i-1, j, w)
            + dataAt(tu, i, j+1, w) + dataAt(tu, i, j-1, w)
            - 4 * dataAt(tu, i, j, w))
    - uv2 + 0.025 * (1 - dataAt(tu, i, j, w));

  u1 = fmin(1, u1);
  dataAt(u, i, j, w) = fmax(0, u1);

  
  v1 = dataAt(tv, i, j, w)
    + 0.1 * (dataAt(tv, i+1, j, w) + dataAt(tv, i-1, j, w)
            + dataAt(tv, i, j+1, w) + dataAt(tv, i, j-1, w)
            - 4 * dataAt(tv, i, j, w))
    + uv2 - 0.08 * dataAt(tv, i, j, w);
  v1 = fmin(1, v1);
  dataAt(v, i, j, w) = fmax(0, v1);

  // printf("(%f, %f)", fmax(0, u1), fmax(0, v1));
}

__global__ void cuUpdateGridShared(double *u, double *tu, double *v, double *tv, int w) {
  
  int i = blockIdx.y * blockDim.y;
  int j = blockIdx.x * blockDim.x;
  double u1, v1, uv2;
  __shared__ double su[TILEWIDTH+2][TILEWIDTH+2];
  __shared__ double sv[TILEWIDTH+2][TILEWIDTH+2];

  int linId = threadIdx.y * blockDim.x + threadIdx.x;

  int si = linId / (TILEWIDTH+2);
  int sj = linId % (TILEWIDTH+2);

  su[si][sj] = dataAt(tu, (i+si), (j+sj), w);
  sv[si][sj] = dataAt(tv, (i+si), (j+sj), w);

  // printf("copy ij (%i, %i) to sij (%i, %i) ", i+si, j+sj, si, sj);
  // printf("\n");

  linId += blockDim.x * blockDim.y;

  si = linId / (TILEWIDTH+2);
  sj = linId % (TILEWIDTH+2);

  if (si < TILEWIDTH+2 && sj < TILEWIDTH+2) {
    su[si][sj] = dataAt(tu, (i+si), (j+sj), w);
    sv[si][sj] = dataAt(tv, (i+si), (j+sj), w);
  }

  // printf("copy ij (%i, %i) to sij (%i, %i) ", i+si, j+sj, si, sj);
  // printf("\n");

  __syncthreads();

  // printf("(%i, %i)", i, j);

  i += threadIdx.y + 1;
  j += threadIdx.x + 1;

  si = threadIdx.y + 1;
  sj = threadIdx.x + 1;


  // if (su[si][sj] != dataAt(tu, i, j, w)){
  //   printf("incorrect u copy at ij (%i, %i) %lf sij (%i, %i) %lf ", i, j, dataAt(tu, i, j, w), si, sj, su[si][sj]);
  // }
  // if (sv[si][sj] != dataAt(tv, i, j, w)){
  //   printf("incorrect v copy at ij (%i, %i) %lf sij (%i, %i) %lf ", i, j, dataAt(tv, i, j, w), si, sj, sv[si][sj]);
  // }

  uv2 = su[si][sj] * sv[si][sj] * sv[si][sj];

  // printf("yay");

  u1 = su[si][sj]
    + 0.2 * (su[si+1][sj] + su[si-1][sj]
            + su[si][sj+1] + su[si][sj-1]
            - 4 * su[si][sj])
    - uv2 + 0.025 * (1 - su[si][sj]);

  u1 = fmin(1, u1);
  dataAt(u, i, j, w) = fmax(0, u1);

  
  v1 = sv[si][sj]
    + 0.1 * (sv[si+1][sj] + sv[si-1][sj]
            + sv[si][sj+1] + sv[si][sj-1]
            - 4 * sv[si][sj])
    + uv2 - 0.08 * sv[si][sj];
  v1 = fmin(1, v1);
  dataAt(v, i, j, w) = fmax(0, v1);

  // printf("(%f, %f)", fmax(0, u1), fmax(0, v1));
}

/*
void initGrid(double u0[], double u1[], double v0[], double v1[], int w) {
  int i, j;

  for (i=0; i<w; i++) {
    for (j=0; j<w; j++) {
      dataAt(u0, i, j, w) = 1.0;
      dataAt(u1, i, j, w) = 1.0;
      dataAt(v0, i, j, w) = 0.0;
      dataAt(v1, i, j, w) = 0.0;
    }
  }
}
*/

