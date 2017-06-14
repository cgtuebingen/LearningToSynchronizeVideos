// Author: Patrick Wieschollek <mail@patwie.com>

#include <stdio.h>
#include "p2dist_op.cu.hh"


template<typename T>
__global__ void P2distOpForwardCudaKernel(T* C, const T* A, const T* B,
                                          const int BB, const int M, const int N, const int D) {
  const int num_threads = 32;

  float cValue = 0; // 0

  int r = blockIdx.y * num_threads + threadIdx.y;
  int c = blockIdx.x * num_threads + threadIdx.x;
  int b = blockIdx.z;

  // if(threadIdx.x==0){
  //     printf("arows: %i acols: %i, brows: %i bcols: %i, crows: %i ccols: %i\n",
  //       M, D, D, N, M, N);
  //     printf("num_threads: %i, blockdim: %i\n", num_threads, blockDim.y);
  //     // printf("M: %i, N: %i D: %i\n", M,N,D);
  // }
  __shared__ float As[num_threads][num_threads];
  __shared__ float Bs[num_threads][num_threads];

  for (int d = 0; d < (num_threads + D - 1) / num_threads; d++) {
      if (d * num_threads + threadIdx.x < D && r < M)
        As[threadIdx.y][threadIdx.x] = A[r * D + d * num_threads + threadIdx.x + M*D*b];
      else
        As[threadIdx.y][threadIdx.x] = 0.0;

      if (d * num_threads + threadIdx.y < D && c < N)
        Bs[threadIdx.y][threadIdx.x] = B[(d * 32 + threadIdx.y) * N + c + N*D*b];
      else
        Bs[threadIdx.y][threadIdx.x] = 0.0;
      __syncthreads();

      #pragma unroll
      for (int n = 0; n < num_threads; ++n){
        cValue += (As[threadIdx.y][n] - Bs[n][threadIdx.x])*(As[threadIdx.y][n] - Bs[n][threadIdx.x]);
        //printf(\"%f - %f\\n\",As[threadIdx.y][n] , Bs[n][threadIdx.x]);
      }
      __syncthreads();
      //if(threadIdx.x==0)
      //printf(\"%f\\n\",cValue);
  }

  const int x = (blockIdx.y * blockDim.y + threadIdx.y);
  const int y = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (r < M && c < N){
    C[(x * N) + y + M*N*b] = sqrt(cValue);
  }


}

template<typename T>
void P2distOpForwardCudaKernelLauncher(T* top,
                                          const T* matrixA, const T* matrixB, 
                                          const int B, const int M, const int N, const int D) {
  
  const int num_threads = 32;
  dim3 threads(num_threads, num_threads);
  dim3 grid((M-1)/threads.x+1, (N-1)/threads.y+1, B);
  P2distOpForwardCudaKernel<T> <<<grid, threads>>>(top, 
                                             matrixA, matrixB, 
                                             B, M, N, D);
  cudaDeviceSynchronize();
}

#define REGISTER_MATRIX_ADD_FORWARD(T) \
  template void P2distOpForwardCudaKernelLauncher<T>(T* top,  \
                                                        const T* matrixA, const T* matrixB, \
                                                        const int B, const int M, const int N, const int D);

REGISTER_MATRIX_ADD_FORWARD(int);
REGISTER_MATRIX_ADD_FORWARD(float);
REGISTER_MATRIX_ADD_FORWARD(double);
