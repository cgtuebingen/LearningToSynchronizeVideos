#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <cuda.h>
// #include <helper_cuda.h>

#include "p2dist_op.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace {
inline int up2(int len, int th) { return (len - 1) / th + 1; }

template <typename Dtype>
__global__ void forward(Dtype* Cmat, const Dtype* Amat, const Dtype* Bmat,
                        const int Arows, const int Brows, const int D) {
  const int num_threads = 32;

  float distance = 0;

  // row ids
  int a_id = blockIdx.y * num_threads + threadIdx.y;
  int b_id = blockIdx.x * num_threads + threadIdx.x;
  int batch = blockIdx.z;

  // in-chip memory
  __shared__ float As[num_threads][num_threads];
  __shared__ float Bs[num_threads][num_threads];

  // split dimension into chunks of size block.dim
  for (int d = 0; d < (num_threads + D - 1) / num_threads; d++) {
    const int da = d * num_threads + threadIdx.x;
    const int db = d * num_threads + threadIdx.y;

    // load matrices into shared memory (when ptr within range)
    if (da < D && a_id < Arows)
      As[threadIdx.y][threadIdx.x] =
          Amat[batch * (Arows * D) + a_id * (D) + da];
    else
      As[threadIdx.y][threadIdx.x] = 0.0;
    if (db < D && b_id < Brows)
      Bs[threadIdx.y][threadIdx.x] =
          Bmat[batch * (Brows * D) + b_id * (D) + db];
    else
      Bs[threadIdx.y][threadIdx.x] = 0.0;
    __syncthreads();

    // distance computation
#pragma unroll
    for (int n = 0; n < num_threads; ++n) {
      distance += (As[threadIdx.y][n] - Bs[n][threadIdx.x]) *
                  (As[threadIdx.y][n] - Bs[n][threadIdx.x]);
    }
    __syncthreads();
  }

  // return the result
  if (a_id < Arows && b_id < Brows) {
    Cmat[batch * (Arows * Brows) + (a_id * Brows) + b_id] = sqrt(distance);
  }
}

}  // namespace

namespace tensorflow {
namespace functor {

template <typename Dtype>
struct P2distFunctor<GPUDevice, Dtype> {
  void operator()(::tensorflow::OpKernelContext* ctx, const Tensor& matrix_a,
                  const Tensor& matrix_b, Tensor* output_) {
    // printf("GPU::P2distFunctor:operator()\n");

    // get dimensions
    const int B = matrix_a.dim_size(0);
    const int Arows = matrix_a.dim_size(1);
    const int Brows = matrix_b.dim_size(1);
    const int D = matrix_a.dim_size(2);

    const int threads = 32;
    dim3 block(threads, threads, 1);
    dim3 grid(up2(Brows, threads), up2(Arows, threads), B);

    float* output_data = output_->flat<Dtype>().data();
    const float* matrix_a_data = matrix_a.flat<Dtype>().data();
    const float* matrix_b_data = matrix_b.flat<Dtype>().data();

    forward<Dtype><<<grid, block>>>(output_data, matrix_a_data, matrix_b_data,
                                    Arows, Brows, D);
    // cudaDeviceSynchronize();
    // getLastCudaError("P2distFunctor::forward::kernel execution failed");
    // checkCudaErrors(cudaDeviceSynchronize());
  }
};

template struct P2distFunctor<GPUDevice, float>;

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
