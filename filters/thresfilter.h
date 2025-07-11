/*
  File: thresfilter.h
  Declaration of pixel structure and thresfilter function.
 */

#ifndef _THRESFILTER_H_
#define _THRESFILTER_H_

/* NOTE: This structure must not be padded! */
typedef struct _pixel {
	unsigned char r,g,b;
} pixel;

#define uint unsigned int 
void thresfilter(const int xsize, const int ysize, int startRow, int endRow, pixel* src, uint psum);

#endif
