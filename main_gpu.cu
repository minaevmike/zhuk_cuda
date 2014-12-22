#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <cuda.h>
#include "helpers.h"
#define DIM 2000
#include <jpeglib.h>

struct cuComplex{
	double i,r;
	__device__ cuComplex(double a, double b) : r(a), i(b){};
	__device__ double abs(void) {
		return i * i + r * r;
	}
	__device__ cuComplex operator*(const cuComplex& a){
		return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
	}
	__device__ cuComplex operator+(const cuComplex& a){
		return cuComplex(r + a.r, i + a.i);
	}
};

__device__ int julia(int x, int y) {
	const double scale = 1.5;
	double jx = scale * (DIM / 2 - x)/(DIM/2);
	double jy = scale * (DIM / 2 - y)/(DIM/2);
	cuComplex c(-0.8, 0.156);
	cuComplex a(jx, jy);
	int i = 0;
	for (i = 0; i < 200; i++) {
		a = a * a + c;
		if (a.abs() > 1000)
			return 0;
	}
	return 1;
}

__global__ void kernel(unsigned char *src) {
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y * gridDim.x;
	src[3 * offset + 0] =  julia(x, y) * 255;
	src[3 * offset + 1] =  0;
	src[3 * offset + 2] =  0;
}

int main(void) {

	int width = DIM;
	int height = DIM;
	int quality = 75;
	int size = width * height * 3;
	unsigned char *scr=(unsigned char*)malloc(size);
	unsigned char *dev_src;
	time_start();
	cudaMalloc((void **)&dev_src, size);
	dim3 grid(DIM, DIM);
	kernel <<<grid, 1>>>(dev_src);
	cudaMemcpy(scr, dev_src, size, cudaMemcpyDeviceToHost);
	printf("It takes %ld for calculation on cpu\n", time_stop());

	FILE* outfile = fopen("out_gpu.jpg", "wb");
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
	jpeg_set_quality(&cinfo, quality, true);
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
