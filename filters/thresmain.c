#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include "ppmio.h"
#include "thresfilter.h"

#define MAX_RAD 1000
#define TAG 0

int main (int argc, char ** argv)
{
	int xsize, ysize, colmax;
	pixel *src = (pixel*) malloc(sizeof(pixel) * MAX_PIXELS);
	pixel *local_src = NULL;
	struct timespec stime, etime;
	uint i, nump, sum, local_sum = 0;

	int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	/* Take care of the arguments */
	if (argc != 3)
	{
		if(rank == 0) fprintf(stderr, "Usage: %s infile outfile\n", argv[0]);
		MPI_Finalize();
		exit(1);
	}
	
	/* Read file */
	if(rank == 0) {
		if(read_ppm (argv[1], &xsize, &ysize, &colmax, (char *) src) != 0) {
			MPI_Abort(MPI_COMM_WORLD, 1);
		}

		if (colmax > 255) {
			fprintf(stderr, "Too large maximum color-component value\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}
	
	printf("Has read the image, calling filter\n");

	MPI_Bcast(&xsize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ysize, 1, MPI_INT, 0, MPI_COMM_WORLD);

	int rows_per_proc = ysize / size;
    int extra_rows = ysize % size;
    int start_row = rank * rows_per_proc + (rank < extra_rows ? rank : extra_rows);
    int local_rows = rows_per_proc + (rank < extra_rows ? 1 : 0);

	local_src = (pixel *)malloc(sizeof(pixel) * xsize * local_rows);

	//thresfilter(xsize, ysize, src);
	if(rank == 0) {
		printf("pic size: %d * %d = %d\n",xsize, ysize, xsize * ysize);
		for (int i = 1; i < size; i++) {
			int s_row = i * rows_per_proc + (i < extra_rows ? i : extra_rows);
			int l_rows = rows_per_proc + (i < extra_rows ? 1 : 0);
			MPI_Send(&src[s_row * xsize], xsize * l_rows * sizeof(pixel), MPI_BYTE, i, TAG, MPI_COMM_WORLD);
		}
		memcpy(local_src, &src[start_row * xsize], xsize * local_rows * sizeof(pixel));
	}
	else {
		MPI_Recv(local_src, xsize * local_rows * sizeof(pixel), MPI_BYTE, 0, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	if (rank == 0) clock_gettime(CLOCK_REALTIME, &stime);

	for (i = 0; i < xsize * local_rows; i++) {
		local_sum += (uint)local_src[i].r + (uint)local_src[i].g + (uint)local_src[i].b;
	}

	MPI_Reduce(&local_sum, &sum, 1, MPI_UNSIGNED, MPI_SUM, 0, MPI_COMM_WORLD);//add every process's local sum to main sum

	if (rank == 0) {
		nump = xsize * ysize; 
		sum /= nump;
	}
	MPI_Bcast(&sum, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

	thresfilter(xsize, local_rows, 0, local_rows, local_src, sum);

	if(rank != 0) {
        MPI_Send(local_src, xsize * local_rows * sizeof(pixel), MPI_BYTE, 0, TAG, MPI_COMM_WORLD);
    }

	if (rank == 0) {
        memcpy(&src[start_row * xsize], local_src, xsize * local_rows * sizeof(pixel));
        for (int i = 1; i < size; i++) {
            int s_row = i * rows_per_proc + (i < extra_rows ? i : extra_rows);
            int l_rows = rows_per_proc + (i < extra_rows ? 1 : 0);
            MPI_Recv(&src[s_row * xsize], xsize * l_rows * sizeof(pixel), MPI_BYTE, i, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

		clock_gettime(CLOCK_REALTIME, &etime);
		printf("Filtering took: %g secs\n", (etime.tv_sec  - stime.tv_sec) + 1e-9*(etime.tv_nsec  - stime.tv_nsec)) ;
	
		if (write_ppm(argv[2], xsize, ysize, (char *)src) != 0) {
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		printf("Writing output file\n");

		free(src);
	}

	free(local_src);
    MPI_Finalize();
	return 0;
}
