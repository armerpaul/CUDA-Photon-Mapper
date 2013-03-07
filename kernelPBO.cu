//kernelPBO.cu (Rob Farber)
 
#include <stdio.h>
#include "cudaPhotonMapper.h"
 
void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 
 
//Simple kernel writes changing colors to a uchar4 array
__global__ void kernel(uchar4* pos, unsigned int width, unsigned int height, 
               float time)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int x = index%width;
  unsigned int y = index/width;
 
  if(index < width*height) {
    unsigned char r = (x + (int) time)&0xff;
    unsigned char g = (y + (int) time)&0xff;
    unsigned char b = ((x+y) + (int) time)&0xff;
     
    // Each thread writes one pixel location in the texture (textel)
    pos[index].w = 0;
    pos[index].x = r;
    pos[index].y = g;
    pos[index].z = b;
  }
}
 
// Wrapper for the __global__ call that sets up the kernel call
extern "C" void launch_kernel(uchar4* pos, unsigned int WINDOW_WIDTH, 
                  unsigned int WINDOW_HEIGHT, float time)
{
  // execute the kernel
  int nThreads=256;
  int totalThreads = WINDOW_HEIGHT * WINDOW_WIDTH;
  int nBlocks = totalThreads/nThreads; 
  nBlocks += ((totalThreads%nThreads)>0)?1:0;
 
  kernel<<< nBlocks, nThreads>>>(pos, WINDOW_WIDTH, WINDOW_HEIGHT, time);
   
  // make certain the kernel has completed 
  cudaThreadSynchronize();
 
  checkCUDAError("kernel failed!");
}
