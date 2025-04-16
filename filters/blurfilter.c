/*
  File: blurfilter.c
  Implementation of blurfilter function.
 */
 
#include <stdio.h>
#include <stdlib.h>
#include "blurfilter.h"
#include "ppmio.h"

const int ThreadCount = 1024 * 1024;

pixel* pix(pixel* image, const int xx, const int yy, const int xsize)
{
	int off = xsize*yy + xx;
	return (image + off);
}

void pix_handle_thread(int x, int y, const int xsize, const int ysize, pixel* src, pixel* dst, double *w, int mode) {
	r = w[0] * pix(src, x, y, xsize)->r;
	g = w[0] * pix(src, x, y, xsize)->g;
	b = w[0] * pix(src, x, y, xsize)->b;
	n = w[0];
	for ( wi=1; wi <= radius; wi++)
	{
		wc = w[wi];
		if (mode == 0) x2 = x - wi;
		else y2 = y - wi;
		
		if (x2 >= 0 && y2 >= 0)
		{
			r += wc * pix(src, x2, y2, xsize)->r;
			g += wc * pix(src, x2, y2, xsize)->g;
			b += wc * pix(src, x2, y2, xsize)->b;
			n += wc;
		}
		if (mode == 0) x2 = x + wi;
		else y2 = y + wi;
		if (x2 < xsize && y2 < ysize)
		{
			r += wc * pix(src, x2, y2, xsize)->r;
			g += wc * pix(src, x2, y2, xsize)->g;
			b += wc * pix(src, x2, y2, xsize)->b;
			n += wc;
		}
	}
	pix(dst,x,y, xsize)->r = r/n;
	pix(dst,x,y, xsize)->g = g/n;
	pix(dst,x,y, xsize)->b = b/n;

}

void blurfilter(const int xsize, const int ysize, pixel* src, int startRow, int endRow, const int radius, const double *w)
{
	int x, y, x2, y2, wi;
	double r, g, b, n, wc;
	pixel *dst = (pixel*) malloc(sizeof(pixel) * MAX_PIXELS);

	pthread_t threads[ThreadCount];
	int threadId = 0;
	
	for (y=0; y<ysize; y++)
	{
		for (x=0; x<xsize; x++)
		{
			pthread_create(&threads[threadId], NULL, pix_handle_thread, (void*)pix(x, y, xsize, ysize, src, dst, w, 0));
			threadId++;
		}
	}

	for (int t = 0; t < threadId; t++) {
        pthread_join(threads[t], NULL);
    }
	threadId = 0;
	
	for (y=0; y<ysize; y++)
	{
		for (x=0; x<xsize; x++)
		{
			pthread_create(&threads[threadId], NULL, pix_handle_thread, (void*)pix(x, y, xsize, ysize, src, dst, w, 0));
			threadId++;	
		}
	}
	for (int t = 0; t < threadId; t++) {
        pthread_join(threads[t], NULL);
    }
	free(dst);
}
