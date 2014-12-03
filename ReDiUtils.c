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

void updateGrid(double u[], double tu[], double v[], double tv[], int w) {
  int i, j;
  double uv2;

  for (i=1; i<w-1; i++) {
    for (j=1; j<w-1; j++) {
      uv2 = dataAt(tu, i, j, w) * dataAt(tv, i, j, w) * dataAt(tv, i, j, w);


      double u1 = dataAt(tu, i, j, w)
        + .2 * (dataAt(tu, i+1, j, w) + dataAt(tu, i-1, j, w)
                + dataAt(tu, i, j+1, w) + dataAt(tu, i, j-1, w)
                - 4 * dataAt(tu, i, j, w))
        - uv2 + .025 * (1 - dataAt(tu, i, j, w));

      u1 = fmin(1, u1);
      dataAt(u, i, j, w) = fmax(0, u1);
      
      double v1 = dataAt(tv, i, j, w)
        + .1 * (dataAt(tv, i+1, j, w) + dataAt(tv, i-1, j, w)
                + dataAt(tv, i, j+1, w) + dataAt(tv, i, j-1, w)
                - 4 * dataAt(tv, i, j, w))
        + uv2 - .08 * dataAt(tv, i, j, w);
      v1 = fmin(1, v1);
      dataAt(v, i, j, w) = fmax(0, v1);
    }
  }
}

void printGrid(double g[], int w) {
  int i, j;

  for (i=0; i<w; i++) {
    for (j=0; j<w; j++) {
      printf("%7.3f ", dataAt(g, i, j, w));
    }
    printf("\n");
  }
}

void dumpGrid(double g[], double h[], int w) {
  int i, j;
  FILE *fp;

  fp = fopen("dump.out", "w");
  
  for (i=0; i<w; i++) {
    for (j=0; j<w; j++) {
      fprintf(fp, "%d %d %f %f\n", i, j, dataAt(g, i, j, w), dataAt(h, i, j, w));
    }
  }
  fclose(fp);
}

void initGrid(double u0[], double u1[], double v0[], double v1[], int w) {
  int i, j;

  for (i=0; i<w; i++) {
    for (j=0; j<w; j++) {
      dataAt(u0, i, j, w) = 1.;
      dataAt(u1, i, j, w) = 1.;
      dataAt(v0, i, j, w) = 0.;
      dataAt(v1, i, j, w) = 0.;
    }
  }
}

