// Author: Patrick Wieschollek <mail@patwie.com>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <stdio.h>
#include "tensorflow/core/framework/shape_inference.h"
#define EIGEN_USE_GPU

#include "p2dist_op.cu.hh"

namespace tensorflow {

namespace shape_inference {

Status UnchangedShape(shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  }
}

REGISTER_OP("P2dist")
.Attr("T: realnumbertype")
.Input("matrix_a: T")
.Input("matrix_b: T")
.Output("output: T")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c)
    {
      // we require the input to have 3 axes [B, ?, D]
      ::tensorflow::shape_inference::ShapeHandle unused_shape_hnd;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &unused_shape_hnd));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &unused_shape_hnd));

      // specify output-shape
      auto B = c->Dim(c->input(0), 0);
      auto B_ = c->Dim(c->input(1), 0);
      auto M = c->Dim(c->input(0), 1);
      auto N = c->Dim(c->input(1), 1);
      auto D = c->Dim(c->input(0), 2);
      auto D_ = c->Dim(c->input(1), 1);

      // TODO aasert D = D_
      shape_inference::DimensionHandle merged;
      TF_RETURN_IF_ERROR(c->Merge(B, B_, &merged));
      TF_RETURN_IF_ERROR(c->Merge(D, D_, &merged));


      ::tensorflow::shape_inference::ShapeHandle matrix_a_shape = c->input(0);
      ::tensorflow::shape_inference::ShapeHandle matrix_b_shape = c->input(1);

      c->set_output(0, c->MakeShape({B,M,N}));

      return Status::OK();
    })
.Doc(R"doc(
All pairwise distances.

This computes C_ij = squared_eucl_dist(A_i, B_j). 

matrix_a: A batch of matrices [B, M, D].
matrix_b: A batch of matrices [B, D, N].
output: A batch of matrices [B, M, N] containing the result.
)doc");


typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// Forward-Pass (CPU)
// --------------------------------------------------
template<typename Device, typename Dtype>
class P2distOp: public OpKernel {
public:
  explicit P2distOp(OpKernelConstruction* context) :
      OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {
    // printf("--> Compute CPU Version <--\n");
    // access incoming tensors (const)
    const Tensor& matrix_a = context->input(0);
    const auto matrix_a_tensor = matrix_a.tensor<Dtype, 3>(); // b,m,d
    const Tensor& matrix_b = context->input(1);
    const auto matrix_b_tensor = matrix_b.tensor<Dtype, 3>(); // b.d,n

    // get dimensions
    const int B = matrix_a.shape().dim_size(0);
    const int M = matrix_a.shape().dim_size(1);
    const int N = matrix_b.shape().dim_size(2);
    const int D = matrix_a.shape().dim_size(2);


    TensorShape output_shape;
    output_shape.AddDim(B);
    output_shape.AddDim(M);
    output_shape.AddDim(N);
    // construct output
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,context->allocate_output(0, output_shape, &output));
    auto out_tensor = output->tensor<Dtype, 3>();

    for (int b = 0; b < B; ++b){
      for (int m = 0; m < M; ++m){
        for (int n = 0; n < N; ++n){
          Dtype d = 0;
          for (int d = 0; d < D; ++d){
            const Dtype aa = matrix_a_tensor(b, m, d);
            const Dtype bb = matrix_b_tensor(b, d, n);
            d += (aa - bb) * (aa - bb);
          }
          out_tensor(b, m, n) = sqrt(d);
        }
      }
    }
  }

private:
//  TF_DISALLOW_COPY_AND_ASSIGN(P2distOp);
};

// Forward-Pass (GPU)
// --------------------------------------------------
template<typename Dtype>
class P2distOp<GPUDevice, Dtype>: public OpKernel {
public:
  explicit P2distOp(OpKernelConstruction* context) :
      OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {
    // printf("--> Compute GPU Version <--\n");
    const Tensor& matrix_a = context->input(0);
    const Tensor& matrix_b = context->input(1);

    // access the elements
    auto matrix_a_flat = matrix_a.flat<Dtype>();
    auto matrix_b_flat = matrix_b.flat<Dtype>();

    const int B = matrix_a.shape().dim_size(0);
    const int M = matrix_a.shape().dim_size(1);
    const int N = matrix_b.shape().dim_size(2);
    const int D = matrix_a.shape().dim_size(2);

    // printf("B: %i, M: %i, N: %i, D: %i\n", B, M, N, D);
      

    TensorShape output_shape;
    output_shape.AddDim(B);
    output_shape.AddDim(M);
    output_shape.AddDim(N);
    // construct output
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,context->allocate_output(0, output_shape, &output));
    auto out_flat = output->flat<Dtype>();

    Dtype* _top = out_flat.data();
    const Dtype* _inA = matrix_a_flat.data();
    const Dtype* _inB = matrix_b_flat.data();


    P2distOpForwardCudaKernelLauncher<Dtype>(_top, _inA, _inB, B, M, N, D);
    
  }

private:
//  TF_DISALLOW_COPY_AND_ASSIGN(P2distOp);
};


#define REGISTER_MYCOPY_KERNELS(type)                                          \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("P2dist").Device(DEVICE_CPU).TypeConstraint<type>("T"),          \
      P2distOp<CPUDevice, type>);                                           \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("P2dist").Device(DEVICE_GPU).TypeConstraint<type>("T"),          \
      P2distOp<GPUDevice, type>);


REGISTER_MYCOPY_KERNELS(float);
REGISTER_MYCOPY_KERNELS(double);

}
