#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


#define SIZE (180*1024*1000)
#define ITER 20


__declspec(align(256)) static double a[SIZE], b[SIZE], c[SIZE];

extern double elapsedTime(void);


int main()
{
	double startTime, duration;
	int i, j;

	//#pragma omp parrallel for
	for(i = 0; i < SIZE; i++){
		c[i] = 0.0f;
		b[i] = a[i] = (double) 1.0f;
	}

	//measure c = a*b+c performance
	startTime = elapsedTime();
	for (i = 0; i < ITER; i++){
	//#pragma omp parrallel for
		for(j = 0; j < SIZE; j++){
			c[j] = a[j] * b[j] + c[j];
		}
	}
	duration = elapsedTime() - startTime;

	double GB = SIZE * sizeof(double) / 1e+9;
	double GBps = 4 * ITER * GB / duration;
	//printf("Running %d openmp threads\n," omp_get_max_threads());
	printf("DP ArraySize = %1f MB, GB/s = %1f\n\n", GB * 1000, GBps);
	
	return 0;
}
