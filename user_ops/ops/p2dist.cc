// ComputerGraphics Tuebingen, 2018
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

namespace shape_inference {

Status UnchangedShape(shape_inference::InferenceContext* c) {
  c->set_output(0, c->input(0));
  return Status::OK();
}
}  // namespace shape_inference

REGISTER_OP("P2dist")
    .Attr("T: realnumbertype")
    .Input("matrix_a: T")
    .Input("matrix_b: T")
    .Output("output: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      // we require the input to have 3 axes [B, ?, D]
      ::tensorflow::shape_inference::ShapeHandle unused_shape_hnd;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &unused_shape_hnd));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &unused_shape_hnd));

      // specify output-shape
      auto B = c->Dim(c->input(0), 0);
      auto Arows = c->Dim(c->input(0), 1);
      auto D = c->Dim(c->input(0), 2);

      auto B_ = c->Dim(c->input(1), 0);
      auto Brows = c->Dim(c->input(1), 1);
      auto D_ = c->Dim(c->input(1), 2);

      shape_inference::DimensionHandle merged;
      TF_RETURN_IF_ERROR(c->Merge(B, B_, &merged));
      TF_RETURN_IF_ERROR(c->Merge(D, D_, &merged));

      ::tensorflow::shape_inference::ShapeHandle matrix_a_shape = c->input(0);
      ::tensorflow::shape_inference::ShapeHandle matrix_b_shape = c->input(1);

      c->set_output(0, c->MakeShape({B, Arows, Brows}));

      return Status::OK();
    })
    .Doc(R"doc(
All pairwise distances.

This computes C_ij = squared_eucl_dist(A_i, B_j).

matrix_a: A batch of matrices [B, Arows, D].
matrix_b: A batch of matrices [B, D, Brows].
output: A batch of matrices [B, Arows, Brows] containing the result.
)doc");

}  // namespace tensorflow
