// Author: Patrick Wieschollek <mail@patwie.com>
#include "p2dist_op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

// Forward-Pass (CPU)
// --------------------------------------------------
template <typename Device, typename Dtype>
class P2distOp : public OpKernel {
 public:
  explicit P2distOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& matrix_a = ctx->input(0);
    const Tensor& matrix_b = ctx->input(1);

    // get dimensions
    const int B = matrix_a.dim_size(0);
    const int Arows = matrix_a.dim_size(1);
    const int Brows = matrix_b.dim_size(1);
    const int D = matrix_a.dim_size(2);

    // construct output
    Tensor* output = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({B, Arows, Brows}), &output));

    ::tensorflow::functor::P2distFunctor<Device, Dtype>()(ctx, matrix_a,
                                                          matrix_b, output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(P2distOp);
};

#define OPNAME(NAME) NAME##Op
#define REGISTER(NAME, Dtype)                                    \
  REGISTER_KERNEL_BUILDER(                                       \
      Name(#NAME).Device(DEVICE_CPU).TypeConstraint<Dtype>("T"), \
      OPNAME(NAME) < CPUDevice, Dtype >);                        \
  REGISTER_KERNEL_BUILDER(                                       \
      Name(#NAME).Device(DEVICE_GPU).TypeConstraint<Dtype>("T"), \
      OPNAME(NAME) < GPUDevice, Dtype >);

REGISTER(P2dist, float);

}  // namespace tensorflow
