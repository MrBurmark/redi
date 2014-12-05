#ifndef _FDIFF_H_
#define _FDIFF_H_

#ifndef DUMP
#define DUMP 1
#endif
#ifndef SHARED
#define SHARED 1
#endif
#ifndef TILEWIDTH
#define TILEWIDTH 16
#endif

#define dataAt(DATA, I, J, W) DATA[(I) * (W) + J]

void updateGrid(double *, double *, double *, double *, int);
void printGrid(double *, int);
void printMid(double g[], int w, int r);
void initGrid(double [], double [], double [], double [], int);
void dumpGrid(double [], double[], int);
void cuDumpGrid(double [], double[], int);

#endif
