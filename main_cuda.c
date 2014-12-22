#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#define DIM 2000
extern "C"
{
#include <jpeglib.h>
}

__device__ int julia(int x, int y) {
	const double scale = 1.5;
	double jx = scale * (DIM / 2 - x)/(DIM/2);
	double jy = scale * (DIM / 2 - y)/(DIM/2);
	double complex c = -0.8 + 0.156 * I;
	double complex a = jx + jy * I;
	int i = 0;
	for (i = 0; i < 200; i++) {
		a = a * a + c;
		if (cabs(a) * cabs(a) > 1000)
			return 0;
	}
	return 1;
}

__global__ void kernel(unsigned char *src) {
	int x = blockIdx.x;
	int y = blockIdx.y;
	int width = DIM;
	char *p = scr + x * width * 3 + y * 3;
	*p = julia(i, j) * 255;
	p++;
	*p = 0;
	p++;
	*p = 0;
}

int main(void) {

	int width = DIM;
	int height = DIM;
	int quality = 75;
	int size = width * height * 3;
	char *scr=(char*)malloc(size);
	char *dev_src;
	cudaMalloc((void **)&dev_src, size);
	dim3 grid(DIM, DIM);
	kernel <<<grid, 1>>>(dev_src);
	cudaMemcpy(src, dev_src, size, cudaMemcpyDeviceToHost);
	FILE* outfile = fopen("out.jpg", "wb");
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
