#ifndef P2DIST_KERNELS_P2DIST_OP_H_
#define P2DIST_KERNELS_P2DIST_OP_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
class OpKernelContext;
class Tensor;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;
}  // namespace tensorflow

namespace tensorflow {
namespace functor {

template <typename Device, typename Dtype>
struct P2distFunctor {
  void operator()(::tensorflow::OpKernelContext* ctx, const Tensor& matrix_a,
                  const Tensor& matrix_b, Tensor* output_);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // P2DIST_KERNELS_P2DIST_OP_H_
