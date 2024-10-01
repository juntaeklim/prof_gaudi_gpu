#include <torch/extension.h>

#include <vector>
// CUDA forward declarations

torch::Tensor scatter_v_i32_indices_cuda(
    torch::Tensor inputs,
    torch::Tensor indices,
    torch::Tensor outputs);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DATATYPE_INDEX(x) TORCH_CHECK(x.dtype() == torch::kInt32, #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_INDEX(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_DATATYPE_INDEX(x)

torch::Tensor scatter_v_i32_indices(
  torch::Tensor inputs,
  torch::Tensor indices,
  torch::Tensor outputs) {
  CHECK_INPUT(inputs);
  CHECK_INDEX(indices);
  CHECK_INPUT(outputs);

  TORCH_CHECK(inputs.size(0) == indices.size(0))
  TORCH_CHECK(inputs.size(1) == outputs.size(1))

  return scatter_v_i32_indices_cuda(inputs, indices, outputs);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scatter_v_i32_indices", &scatter_v_i32_indices, "Embedding scatter (CUDA)");
}