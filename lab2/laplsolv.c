//-----------------------------------------------------------------------
// Serial program for solving the heat conduction problem 
// on a square using the Jacobi method. 
// Modified for O(N) extra memory version with OpenMP.
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
    double prev[n+2], curr[n+2], next[n+2], new_curr[n+2];
    int iter = 0;
    double max_diff;

    struct timespec starttime, endtime;

    // Set boundary conditions and initial values
    for (int i = 0; i <= n+1; ++i)
    {
        for (int j = 0; j <= n+1; ++j)
        {
            if      (i == n+1)           T[i][j] = 2.0; // bottom
            else if (j == 0 || j == n+1) T[i][j] = 1.0; // left and right
            else                         T[i][j] = 0.0; // inside
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &starttime);

    // Jacobi iteration
    do {
        max_diff = 0.0;

        for (int i = 1; i <= n; ++i) {
            // Copy rows
            for (int j = 0; j <= n+1; ++j) {
                prev[j] = T[i-1][j];
                curr[j] = T[i][j];
                next[j] = T[i+1][j];
            }

            // Compute new current row
            #pragma omp parallel for reduction(max:max_diff)
            for (int j = 1; j <= n; ++j) {
                new_curr[j] = 0.25 * (prev[j] + next[j] + curr[j-1] + curr[j+1]);
                double diff = fabs(new_curr[j] - curr[j]);
                if (diff > max_diff)
                    max_diff = diff;
            }

            // Write back updated values
            for (int j = 1; j <= n; ++j) {
                T[i][j] = new_curr[j];
            }
        }

        iter++;
    } while (iter < maxiter && max_diff > tol);

    clock_gettime(CLOCK_MONOTONIC, &endtime);

    printf("Time: %f\n", timediff(&starttime, &endtime));
    printf("Number of iterations: %d\n", iter);
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
