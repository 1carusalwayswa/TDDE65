#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include "ppmio.h"
#include "thresfilter.h"

#define NUM_THREADS 64 

struct thread_data {
	// calculation function arguments
	int startIdx;
    int endIdx;
    uint partialSum;
	// filter function arguments
	int thread_id;
	int xsize;
	int ysize;
	int startRow;
	int endRow;
	uint sum;
	pixel* src;
};
struct thread_data thread_data_array[NUM_THREADS];

void *compute_sum(void *arg) {
    struct thread_data *data = (struct thread_data *) arg;
    uint local_sum = 0;
	pixel* local_src = data->src;
    int start = data->startIdx;
    int end = data->endIdx;
    
    for (int i = start; i < end; i++) {
        local_sum += (uint)local_src[i].r + (uint)local_src[i].g + (uint)local_src[i].b;
    }
    
    data->partialSum = local_sum;
    pthread_exit(NULL);
}

void *thread_filter(void *arg) {
	struct thread_data *data = (struct thread_data *) arg;
	thresfilter(data -> xsize, data -> ysize, data -> startRow, data -> endRow, data -> src, data->sum);
	pthread_exit(NULL);
}

int main (int argc, char ** argv)
{
	int xsize, ysize, colmax;
	pixel *src = (pixel*) malloc(sizeof(pixel) * MAX_PIXELS);
	struct timespec stime, etime;
	pthread_t threads[NUM_THREADS];
	uint nump,sum,i;
	
	/* Take care of the arguments */
	if (argc != 3)
	{
		fprintf(stderr, "Usage: %s infile outfile\n", argv[0]);
		exit(1);
	}
	
	/* Read file */
	if(read_ppm (argv[1], &xsize, &ysize, &colmax, (char *) src) != 0)
		exit(1);
	
	if (colmax > 255)
	{
		fprintf(stderr, "Too large maximum color-component value\n");
		exit(1);
	}
	
	printf("Has read the image, calling filter\n");

	clock_gettime(CLOCK_REALTIME, &stime);
	nump = xsize * ysize;
    
	int pixels_per_thread = nump / NUM_THREADS;
    
    for (int t = 0; t < NUM_THREADS; t++) {
        thread_data_array[t].thread_id = t;
        thread_data_array[t].src = src;
        thread_data_array[t].startIdx = t * pixels_per_thread;
        thread_data_array[t].endIdx = (t == NUM_THREADS - 1) ? nump : (t + 1) * pixels_per_thread;
        thread_data_array[t].partialSum = 0;
        
        pthread_create(&threads[t], NULL, compute_sum, (void *) &thread_data_array[t]);
    }
    
    sum = 0;
    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
        sum += thread_data_array[t].partialSum;
    }
	sum = sum / nump;

	int rows_per_thread = ysize / NUM_THREADS;
	for (int t = 0; t < NUM_THREADS; t++) {
        thread_data_array[t].thread_id = t;
        thread_data_array[t].xsize = xsize;
        thread_data_array[t].ysize = ysize;
        thread_data_array[t].src = src;
        thread_data_array[t].startRow = t * rows_per_thread;
        thread_data_array[t].endRow = (t == NUM_THREADS - 1) ? ysize : (t + 1) * rows_per_thread;
		thread_data_array[t].sum = sum;
		// printf("first run\n");
		// printf("%d: st:%d ed:%d\n", t, thread_data_array[t].startRow, thread_data_array[t].endRow);

        pthread_create(&threads[t], NULL, thread_filter, (void *) &thread_data_array[t]);
    }

	for (int t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }
	clock_gettime(CLOCK_REALTIME, &etime);
	
	printf("Filtering took: %g secs\n", (etime.tv_sec  - stime.tv_sec) + 1e-9*(etime.tv_nsec  - stime.tv_nsec)) ;
	
	/* Write result */
	printf("Writing output file\n");
	
	if (write_ppm(argv[2], xsize, ysize, (char *)src) != 0)
		exit(1);
}
