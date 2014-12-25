#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include "helpers.h"
#define DIM 2000
#include <jpeglib.h>
#include <time.h>

int julia(int x, int y) {
	const double scale = 1.5;
	double jx = scale * (DIM / 2 - x)/(DIM/2);
	double jy = scale * (DIM / 2 - y)/(DIM/2);
	double complex c = -0.4 + 0.6 * I;
	double complex a = jx + jy * I;
	int i = 0;
	for (i = 0; i < 200; i++) {
		a = a * a + c;
		if (cabs(a) * cabs(a) > 1000)
			return 0;
	}
	return 1;
}

int main(void) {

	int width = DIM;
	int height = DIM;
	int quality = 75;
	char *scr=(char*)malloc(width * height * 3);
	time_start();
	char *default_name = "out_cpu.jpg";
	char buf[80];
	snprintf(buf, 80, "%s%d", default_name, (int)time(NULL));
	FILE* outfile = fopen(buf, "wb");
	int i = 0;
	int j = 0;
	for (i = 0 ; i < width; i++)
		for (j = 0; j < height; j++) {
			char *p = scr + j * width * 3 + i * 3;
			int result = julia(i, j);
			*p = result * 255;
			p++;
			*p = 0;//result % 255;
			p++;
			*p = 0;//result % 255;
		}
	printf("It takes %ld for calculation on cpu\n", time_stop());
	/* Compress to JPEG */
	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr jerr;
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_compress(&cinfo);

	jpeg_stdio_dest(&cinfo, outfile);

	cinfo.image_width=width;
	cinfo.image_height=height;
	cinfo.input_components=3;
	cinfo.in_color_space=JCS_RGB;

	jpeg_set_defaults(&cinfo);
	jpeg_set_quality(&cinfo, quality, 1);
	jpeg_start_compress(&cinfo, TRUE);

	JSAMPROW row_pointer[1];
	int row_stride;

	row_stride = cinfo.image_width*3;

	while (cinfo.next_scanline < cinfo.image_height) {
		row_pointer[0]=(JSAMPLE *)(scr+cinfo.next_scanline*row_stride);
		jpeg_write_scanlines(&cinfo, row_pointer, 1);
	}
	jpeg_finish_compress(&cinfo);
	jpeg_destroy_compress(&cinfo);

	free(scr);
	return 0;
}
