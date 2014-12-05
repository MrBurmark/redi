#!/bin/bash

for input in G1026 G2050; do
	for tw in 16 32; do
		for run in 1 2 3 4 5; do
			nvcc -D TILEWIDTH=$tw -D SHARED=0 -D DUMP=0 cuReDi.cu ReDiUtils.c -O3 -o cuReDi -lm -fmad=false -arch=compute_20 -code=sm_20,sm_30,sm_35
			./cuReDi $input
		done
	done
done

for input in G1026 G2050; do
	for tw in 16 32; do
		for run in 1 2 3 4 5; do
			nvcc -D TILEWIDTH=$tw -D SHARED=1 -D DUMP=0 cuReDi.cu ReDiUtils.c -O3 -o cuReDi -lm -fmad=false -arch=compute_20 -code=sm_20,sm_30,sm_35
			./cuReDi $input
		done
	done
done
