#include "thresfilter.h"

void thresfilter(const int xsize, const int ysize, int startRow, int endRow, pixel* src, uint sum)
{
	uint i, psum, nump;
	/*nump = xsize * ysize;
	
	for (i = 0, sum = 0; i < nump; i++)
	{
		sum += (uint)src[i].r + (uint)src[i].g + (uint)src[i].b;
	}
	
	sum /= nump;*/
	
	for (int y = startRow; y < endRow; y++) {

		for (int x = 0; x < xsize; x++) {

			int idx = y * xsize + x;
			uint psum = src[idx].r + src[idx].g + src[idx].b;

			if (psum < sum) {
				src[idx].r = src[idx].g = src[idx].b = 0;
			} else {
				src[idx].r = src[idx].g = src[idx].b = 255;
			}
			
		}

	}
}
