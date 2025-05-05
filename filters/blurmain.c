#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include "ppmio.h"
#include "blurfilter.h"
#include "gaussw.h"

#define MAX_RAD 1000
#define TAG 0

int main(int argc, char **argv) {
    int radius, xsize, ysize, colmax;
    pixel *src = NULL;
    pixel *local_src = NULL;
    double w[MAX_RAD];

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    struct timespec stime, etime;

    if (argc != 4) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s radius infile outfile\n", argv[0]);
        MPI_Finalize();
        exit(1);
    }

    radius = atoi(argv[1]);
    if ((radius > MAX_RAD) || (radius < 1)) {
        if (rank == 0)
            fprintf(stderr, "Radius must be between 1 and %d\n", MAX_RAD);
        MPI_Finalize();
        exit(1);
    }

    if (rank == 0) {
        src = (pixel *)malloc(sizeof(pixel) * MAX_PIXELS);
        if (read_ppm(argv[2], &xsize, &ysize, &colmax, (char *)src) != 0) {
            fprintf(stderr, "Failed to read file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);//stop all the other processes, exit code 1
        }
        if (colmax > 255) {
            fprintf(stderr, "Too large maximum color-component value\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        get_gauss_weights(radius, w);
    }

    MPI_Bcast(&xsize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ysize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(w, MAX_RAD, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //printf("Rank %d reached after bcast\n", rank);

    int rows_per_proc = ysize / size;
    int extra_rows = ysize % size;
    int start_row = rank * rows_per_proc + (rank < extra_rows ? rank : extra_rows);
    int local_rows = rows_per_proc + (rank < extra_rows ? 1 : 0);

    local_src = (pixel *)malloc(sizeof(pixel) * xsize * local_rows);
    pixel *local_dst = (pixel*) malloc(sizeof(pixel) * MAX_PIXELS);

    if (rank == 0) {
        printf("pic size: %d * %d = %d\n",xsize, ysize, xsize * ysize);
        for (int i = 1; i < size; i++) {
            int s_row = i * rows_per_proc + (i < extra_rows ? i : extra_rows);
            int l_rows = rows_per_proc + (i < extra_rows ? 1 : 0);
            MPI_Send(&src[s_row * xsize], xsize * l_rows * sizeof(pixel), MPI_BYTE, i, TAG, MPI_COMM_WORLD);
            //printf("Rank %d reached after send\n", rank);
        }
        memcpy(local_src, &src[start_row * xsize], xsize * local_rows * sizeof(pixel));
    } else {
        MPI_Recv(local_src, xsize * local_rows * sizeof(pixel), MPI_BYTE, 0, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //printf("Rank %d reached after recv\n", rank);
    }

    if (rank == 0)
        clock_gettime(CLOCK_REALTIME, &stime);

    blurfilter1(xsize, local_rows, local_src, local_dst, 0, local_rows, radius, w);
    blurfilter2(xsize, local_rows, local_src, local_dst, 0, local_rows, radius, w);
    //printf("Rank %d reached after blurfilter\n", rank);

    if(rank != 0) {
        MPI_Send(local_src, xsize * local_rows * sizeof(pixel), MPI_BYTE, 0, TAG, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        memcpy(&src[start_row * xsize], local_src, xsize * local_rows * sizeof(pixel));
        for (int i = 1; i < size; i++) {
            int s_row = i * rows_per_proc + (i < extra_rows ? i : extra_rows);
            int l_rows = rows_per_proc + (i < extra_rows ? 1 : 0);
            MPI_Recv(&src[s_row * xsize], xsize * l_rows * sizeof(pixel), MPI_BYTE, i, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //printf("Rank %d reached after mixing up data\n", rank);
        }

        clock_gettime(CLOCK_REALTIME, &etime);
        printf("Filtering took: %g secs\n", 
            (etime.tv_sec  - stime.tv_sec) + 1e-9*(etime.tv_nsec  - stime.tv_nsec));
        if (write_ppm(argv[3], xsize, ysize, (char *)src) != 0) {
            fprintf(stderr, "Write failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        free(src);
    }

    free(local_src);
    MPI_Finalize();
    return 0;
}
