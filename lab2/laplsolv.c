
//-----------------------------------------------------------------------
// Serial program for solving the heat conduction problem 
// on a square using the Jacobi method. 
// Written by August Ernstsson 2015-2019
//-----------------------------------------------------------------------

#define _POSIX_C_SOURCE 199309L
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>

double timediff(struct timespec *begin, struct timespec *end)
{
	double sec = 0.0, nsec = 0.0;
	if ((end->tv_nsec - begin->tv_nsec) < 0)
	{
		sec  = (double)(end->tv_sec  - begin->tv_sec  - 1);
		nsec = (double)(end->tv_nsec - begin->tv_nsec + 1000000000);
	} else
	{
		sec  = (double)(end->tv_sec  - begin->tv_sec );
		nsec = (double)(end->tv_nsec - begin->tv_nsec);
	}
	return sec + nsec / 1E9;
}

void printm(int n, double *M)
{
	for (int i = 0; i < n; i ++)
	{
		for (int j = 0; j < n; j ++)
		{
			printf("%f\t", *(M + n * i + j));
		}
		printf("\n");
	}
	printf("\n");
}


void arrcpy(double *dst, double *src, int len)
{
	for (int it = 0; it < len; it++)
		dst[it] = src[it];
}


void laplsolv(int n, int maxiter, double tol)
{
	double T[n+2][n+2];
	double last_row[n], end_row[n], cur_row[n];
	int k;
	
	struct timespec starttime, endtime;
	
	// Set boundary conditions and initial values for the unknowns
	for (int i = 0; i <= n+1; ++i)
	{
		for (int j = 0; j <= n+1; ++j)
		{
			if      (i == n+1)           T[i][j] = 2;
			else if (j == 0 || j == n+1) T[i][j] = 1;
			else                         T[i][j] = 0;
		}
	}
	
	clock_gettime(CLOCK_MONOTONIC, &starttime);
	
	// Solve the linear system of equations using the Jacobi method
	for (k = 0; k < maxiter; ++k)
	{
		double error = -INFINITY;

#pragma omp parallel private(last_row, end_row, cur_row)
		{
            int cur_num = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
            int rows_per = n / num_threads;
            double my_error = -INFINITY;


            int start = cur_num * rows_per;
            int end = start + rows_per;
            if (cur_num == num_threads - 1)
            {
                end = n;
            }

            arrcpy(last_row, &T[start][1], n);
            arrcpy(end_row, &T[end + 1][1], n);

#pragma omp barrier
            // Loop for each of this thread's rows
            for (int i = start + 1; i <= end - 1; ++i)
            {
                arrcpy(cur_row, &T[i][1], n);
                
                // Apply the Jacobi algorithm to each element in this row
                for (int j = 1; j <= n; ++j)
                {
                    cur_row[j-1] = (T[i][j-1] + T[i][j+1] + T[i+1][j] + last_row[j-1]) / 4.0;
                    my_error = fmax(my_error, fabs(cur_row[j-1] - T[i][j]));
                }
                
                arrcpy(last_row, &T[i][1], n);
                arrcpy(&T[i][1], cur_row, n);
            }

            for (int j = 1; j <= n; ++j)
            {
                cur_row[j-1] = (T[end][j-1] + T[end][j+1] + last_row[j-1] + end_row[j-1]) / 4.0;
                my_error = fmax(my_error, fabs(cur_row[j-1] - T[end][j]));
            }
            arrcpy(&T[end][1], cur_row, n);

#pragma omp critical
            {
                if (my_error > error)
                    error = my_error;
            }
		}
		if (error < tol)
			break;
	}
	
	clock_gettime(CLOCK_MONOTONIC, &endtime);
	
	printf("Time: %f\n", timediff(&starttime, &endtime));
	printf("Number of iterations: %d\n", k);
	printf("Temperature of element T(1,1): %.17f\n", T[1][1]);
}


int main(int argc, char* argv[])
{
	if (argc != 4)
	{
		printf("Usage: %s [size] [maxiter] [tolerance] \n", argv[0]);
		exit(1);
	}
	
	int size = atoi(argv[1]);
	int maxiter = atoi(argv[2]);
	double tol = atof(argv[3]);
	
	printf("Size %d, max iter %d and tolerance %f.\n", size, maxiter, tol);
	laplsolv(size, maxiter, tol);
	return 0;
}