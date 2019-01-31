// ComputerGraphics Tuebingen, 2017

#include "p2dist_op.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

namespace functor {

template <typename Dtype>
struct P2distFunctor<CPUDevice, Dtype> {
  void operator()(::tensorflow::OpKernelContext* ctx, const Tensor& matrix_a,
                  const Tensor& matrix_b, Tensor* output_) {
    // printf("CPU::P2distFunctor:operator()\n");

    const auto Amat = matrix_a.tensor<Dtype, 3>();
    const auto Bmat = matrix_b.tensor<Dtype, 3>();

    auto output = output_->tensor<Dtype, 3>();

    // get dimensions
    const int B = matrix_a.dim_size(0);
    const int Arows = matrix_a.dim_size(1);
    const int Brows = matrix_b.dim_size(1);
    const int D = matrix_a.dim_size(2);

    for (int b = 0; b < B; ++b) {
      for (int arow = 0; arow < Arows; ++arow) {
        for (int brow = 0; brow < Brows; ++brow) {
          Dtype d = 0;
          for (int d = 0; d < D; ++d) {
            const Dtype aa = Amat(b, arow, d);
            const Dtype bb = Bmat(b, d, brow);
            d += (aa - bb) * (aa - bb);
          }
          output(b, arow, brow) = sqrt(d);
        }
      }
    }
  }
};

template struct P2distFunctor<CPUDevice, float>;
}  // namespace functor
}  // namespace tensorflow
