#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include "ppmio.h"
#include "blurfilter.h"
#include "gaussw.h"

#define MAX_RAD 1000
#define NUM_THREADS 4

struct thread_data {
	int thread_id;
	int xsize;
	int ysize;
	pixel* src;
	pixel* dst;
	int startRow; 
	int endRow; 
	int radius;
	double *w;
};
struct thread_data thread_data_array[NUM_THREADS];

void *thread_blur_filter(void *arg) {
	struct thread_data *data = (struct thread_data *) arg;
	blurfilter1(data -> xsize, data -> ysize, data -> src, data -> dst, data -> startRow, data -> endRow, data -> radius, data -> w);
	pthread_exit(NULL);
}

void *thread_blur_filter2(void *arg) {
	struct thread_data *data = (struct thread_data *) arg;
	blurfilter2(data -> xsize, data -> ysize, data -> src, data -> dst, data -> startRow, data -> endRow, data -> radius, data -> w);
	pthread_exit(NULL);
}

int main (int argc, char ** argv)
{
	int radius, xsize, ysize, colmax;
	pixel *src = (pixel*) malloc(sizeof(pixel) * MAX_PIXELS);
	pixel *dst = (pixel*) malloc(sizeof(pixel) * MAX_PIXELS);
	struct timespec stime, etime;
	double w[MAX_RAD];
	pthread_t threads[NUM_THREADS];
	
	/* Take care of the arguments */
	if (argc != 4)
	{
		fprintf(stderr, "Usage: %s radius infile outfile\n", argv[0]);
		exit(1);
	}
	
	radius = atoi(argv[1]);
	if ((radius > MAX_RAD) || (radius < 1))
	{
		fprintf(stderr, "Radius (%d) must be greater than zero and less then %d\n", radius, MAX_RAD);
		exit(1);
	}
	
	/* Read file */
	if (read_ppm (argv[2], &xsize, &ysize, &colmax, (char *) src) != 0)
		exit(1);
	
	if (colmax > 255)
	{
		fprintf(stderr, "Too large maximum color-component value\n");
		exit(1);
	}
	
	printf("Has read the image, generating coefficients\n");
	
	/* filter */
	get_gauss_weights(radius, w);
	
	printf("Calling filter\n");
	printf("%d %d\n", xsize, ysize);
	
	clock_gettime(CLOCK_REALTIME, &stime);
	int rows_per_thread = ysize / NUM_THREADS;
	
	for (int t = 0; t < NUM_THREADS; t++) {
        thread_data_array[t].thread_id = t;
        thread_data_array[t].xsize = xsize;
        thread_data_array[t].ysize = ysize;
        thread_data_array[t].radius = radius;
        thread_data_array[t].w = w;
        thread_data_array[t].src = src;
		thread_data_array[t].dst = dst;
        thread_data_array[t].startRow = t * rows_per_thread;
        thread_data_array[t].endRow = (t == NUM_THREADS - 1) ? ysize : (t + 1) * rows_per_thread;
		printf("first run\n");
		printf("%d: st:%d ed:%d\n", t, thread_data_array[t].startRow, thread_data_array[t].endRow);

        pthread_create(&threads[t], NULL, thread_blur_filter, (void *) &thread_data_array[t]);
    }

	for (int t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }

	for (int t = 0; t < NUM_THREADS; t++) {
        thread_data_array[t].thread_id = t;
        thread_data_array[t].xsize = xsize;
        thread_data_array[t].ysize = ysize;
        thread_data_array[t].radius = radius;
        thread_data_array[t].w = w;
        thread_data_array[t].src = src;
		thread_data_array[t].dst = dst;
        thread_data_array[t].startRow = t * rows_per_thread;
        thread_data_array[t].endRow = (t == NUM_THREADS - 1) ? ysize : (t + 1) * rows_per_thread;
		printf("second run\n");
		printf("%d: st:%d ed:%d\n", t, thread_data_array[t].startRow, thread_data_array[t].endRow);

        pthread_create(&threads[t], NULL, thread_blur_filter2, (void *) &thread_data_array[t]);
    }

	for (int t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }
	clock_gettime(CLOCK_REALTIME, &etime);
	
	printf("Filtering took: %g secs\n", (etime.tv_sec  - stime.tv_sec) +
	1e-9*(etime.tv_nsec  - stime.tv_nsec)) ;
	
	/* Write result */
	printf("Writing output file\n");
	
	if(write_ppm (argv[3], xsize, ysize, (char *)src) != 0)
		exit(1);
}
