/*
  File: blurfilter.h
  Declaration of pixel structure and blurfilter function.
 */

#ifndef _BLURFILTER_H_
#define _BLURFILTER_H_

/* NOTE: This structure must not be padded! */
typedef struct _pixel {
	unsigned char r,g,b;
} pixel;

void blurfilter1(const int xsize, const int ysize, pixel* src, pixel* dst, int startRow, int endRow, const int radius, const double *w);
void blurfilter2(const int xsize, const int ysize, pixel* src, pixel* dst, int startRow, int endRow, const int radius, const double *w);

#endif
