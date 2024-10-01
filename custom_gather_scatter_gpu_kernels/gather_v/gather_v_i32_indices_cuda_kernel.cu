#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__global__ void gather_v_i32_indices_cuda_kernel_unroll8(
    const scalar_t* __restrict__ inputs,
    const int32_t* __restrict__ indices,
    scalar_t* __restrict__ outputs,
    const int embedding_dim,
    const int num_indices) {

  const int cur_thread_0 = 8 * blockIdx.x * blockDim.x + 8 * threadIdx.x;
  const int cur_thread_1 = 8 * blockIdx.x * blockDim.x + 8 * threadIdx.x + 1;
  const int cur_thread_2 = 8 * blockIdx.x * blockDim.x + 8 * threadIdx.x + 2;
  const int cur_thread_3 = 8 * blockIdx.x * blockDim.x + 8 * threadIdx.x + 3;
  const int cur_thread_4 = 8 * blockIdx.x * blockDim.x + 8 * threadIdx.x + 4;
  const int cur_thread_5 = 8 * blockIdx.x * blockDim.x + 8 * threadIdx.x + 5;
  const int cur_thread_6 = 8 * blockIdx.x * blockDim.x + 8 * threadIdx.x + 6;
  const int cur_thread_7 = 8 * blockIdx.x * blockDim.x + 8 * threadIdx.x + 7;

  if (cur_thread_7 < num_indices * embedding_dim){
    const int out_row_0 = cur_thread_0 / embedding_dim;
    const int out_row_1 = cur_thread_1 / embedding_dim;
    const int out_row_2 = cur_thread_2 / embedding_dim;
    const int out_row_3 = cur_thread_3 / embedding_dim;
    const int out_row_4 = cur_thread_4 / embedding_dim;
    const int out_row_5 = cur_thread_5 / embedding_dim;
    const int out_row_6 = cur_thread_6 / embedding_dim;
    const int out_row_7 = cur_thread_7 / embedding_dim;

    const int out_col_0 = cur_thread_0 % embedding_dim;
    const int out_col_1 = cur_thread_1 % embedding_dim;
    const int out_col_2 = cur_thread_2 % embedding_dim;
    const int out_col_3 = cur_thread_3 % embedding_dim;
    const int out_col_4 = cur_thread_4 % embedding_dim;
    const int out_col_5 = cur_thread_5 % embedding_dim;
    const int out_col_6 = cur_thread_6 % embedding_dim;
    const int out_col_7 = cur_thread_7 % embedding_dim;

    const int index_0 = indices[out_row_0];
    const int index_1 = indices[out_row_1];
    const int index_2 = indices[out_row_2];
    const int index_3 = indices[out_row_3];
    const int index_4 = indices[out_row_4];
    const int index_5 = indices[out_row_5];
    const int index_6 = indices[out_row_6];
    const int index_7 = indices[out_row_7];

    const int tmp_0 = inputs[index_0 * embedding_dim + out_col_0];
    const int tmp_1 = inputs[index_1 * embedding_dim + out_col_1];
    const int tmp_2 = inputs[index_2 * embedding_dim + out_col_2];
    const int tmp_3 = inputs[index_3 * embedding_dim + out_col_3];
    const int tmp_4 = inputs[index_4 * embedding_dim + out_col_4];
    const int tmp_5 = inputs[index_5 * embedding_dim + out_col_5];
    const int tmp_6 = inputs[index_6 * embedding_dim + out_col_6];
    const int tmp_7 = inputs[index_7 * embedding_dim + out_col_7];

    outputs[out_row_0 * embedding_dim + out_col_0] = tmp_0;
    outputs[out_row_1 * embedding_dim + out_col_1] = tmp_1;
    outputs[out_row_2 * embedding_dim + out_col_2] = tmp_2;
    outputs[out_row_3 * embedding_dim + out_col_3] = tmp_3;
    outputs[out_row_4 * embedding_dim + out_col_4] = tmp_4;
    outputs[out_row_5 * embedding_dim + out_col_5] = tmp_5;
    outputs[out_row_6 * embedding_dim + out_col_6] = tmp_6;
    outputs[out_row_7 * embedding_dim + out_col_7] = tmp_7;

  }else if (cur_thread_6 < num_indices * embedding_dim){
    const int out_row_0 = cur_thread_0 / embedding_dim;
    const int out_row_1 = cur_thread_1 / embedding_dim;
    const int out_row_2 = cur_thread_2 / embedding_dim;
    const int out_row_3 = cur_thread_3 / embedding_dim;
    const int out_row_4 = cur_thread_4 / embedding_dim;
    const int out_row_5 = cur_thread_5 / embedding_dim;
    const int out_row_6 = cur_thread_6 / embedding_dim;

    const int out_col_0 = cur_thread_0 % embedding_dim;
    const int out_col_1 = cur_thread_1 % embedding_dim;
    const int out_col_2 = cur_thread_2 % embedding_dim;
    const int out_col_3 = cur_thread_3 % embedding_dim;
    const int out_col_4 = cur_thread_4 % embedding_dim;
    const int out_col_5 = cur_thread_5 % embedding_dim;
    const int out_col_6 = cur_thread_6 % embedding_dim;

    const int index_0 = indices[out_row_0];
    const int index_1 = indices[out_row_1];
    const int index_2 = indices[out_row_2];
    const int index_3 = indices[out_row_3];
    const int index_4 = indices[out_row_4];
    const int index_5 = indices[out_row_5];
    const int index_6 = indices[out_row_6];

    const int tmp_0 = inputs[index_0 * embedding_dim + out_col_0];
    const int tmp_1 = inputs[index_1 * embedding_dim + out_col_1];
    const int tmp_2 = inputs[index_2 * embedding_dim + out_col_2];
    const int tmp_3 = inputs[index_3 * embedding_dim + out_col_3];
    const int tmp_4 = inputs[index_4 * embedding_dim + out_col_4];
    const int tmp_5 = inputs[index_5 * embedding_dim + out_col_5];
    const int tmp_6 = inputs[index_6 * embedding_dim + out_col_6];

    outputs[out_row_0 * embedding_dim + out_col_0] = tmp_0;
    outputs[out_row_1 * embedding_dim + out_col_1] = tmp_1;
    outputs[out_row_2 * embedding_dim + out_col_2] = tmp_2;
    outputs[out_row_3 * embedding_dim + out_col_3] = tmp_3;
    outputs[out_row_4 * embedding_dim + out_col_4] = tmp_4;
    outputs[out_row_5 * embedding_dim + out_col_5] = tmp_5;
    outputs[out_row_6 * embedding_dim + out_col_6] = tmp_6;

  }else if (cur_thread_5 < num_indices * embedding_dim){
    const int out_row_0 = cur_thread_0 / embedding_dim;
    const int out_row_1 = cur_thread_1 / embedding_dim;
    const int out_row_2 = cur_thread_2 / embedding_dim;
    const int out_row_3 = cur_thread_3 / embedding_dim;
    const int out_row_4 = cur_thread_4 / embedding_dim;
    const int out_row_5 = cur_thread_5 / embedding_dim;

    const int out_col_0 = cur_thread_0 % embedding_dim;
    const int out_col_1 = cur_thread_1 % embedding_dim;
    const int out_col_2 = cur_thread_2 % embedding_dim;
    const int out_col_3 = cur_thread_3 % embedding_dim;
    const int out_col_4 = cur_thread_4 % embedding_dim;
    const int out_col_5 = cur_thread_5 % embedding_dim;

    const int index_0 = indices[out_row_0];
    const int index_1 = indices[out_row_1];
    const int index_2 = indices[out_row_2];
    const int index_3 = indices[out_row_3];
    const int index_4 = indices[out_row_4];
    const int index_5 = indices[out_row_5];

    const int tmp_0 = inputs[index_0 * embedding_dim + out_col_0];
    const int tmp_1 = inputs[index_1 * embedding_dim + out_col_1];
    const int tmp_2 = inputs[index_2 * embedding_dim + out_col_2];
    const int tmp_3 = inputs[index_3 * embedding_dim + out_col_3];
    const int tmp_4 = inputs[index_4 * embedding_dim + out_col_4];
    const int tmp_5 = inputs[index_5 * embedding_dim + out_col_5];

    outputs[out_row_0 * embedding_dim + out_col_0] = tmp_0;
    outputs[out_row_1 * embedding_dim + out_col_1] = tmp_1;
    outputs[out_row_2 * embedding_dim + out_col_2] = tmp_2;
    outputs[out_row_3 * embedding_dim + out_col_3] = tmp_3;
    outputs[out_row_4 * embedding_dim + out_col_4] = tmp_4;
    outputs[out_row_5 * embedding_dim + out_col_5] = tmp_5;

  }else if (cur_thread_4 < num_indices * embedding_dim){
    const int out_row_0 = cur_thread_0 / embedding_dim;
    const int out_row_1 = cur_thread_1 / embedding_dim;
    const int out_row_2 = cur_thread_2 / embedding_dim;
    const int out_row_3 = cur_thread_3 / embedding_dim;
    const int out_row_4 = cur_thread_4 / embedding_dim;

    const int out_col_0 = cur_thread_0 % embedding_dim;
    const int out_col_1 = cur_thread_1 % embedding_dim;
    const int out_col_2 = cur_thread_2 % embedding_dim;
    const int out_col_3 = cur_thread_3 % embedding_dim;
    const int out_col_4 = cur_thread_4 % embedding_dim;

    const int index_0 = indices[out_row_0];
    const int index_1 = indices[out_row_1];
    const int index_2 = indices[out_row_2];
    const int index_3 = indices[out_row_3];
    const int index_4 = indices[out_row_4];

    const int tmp_0 = inputs[index_0 * embedding_dim + out_col_0];
    const int tmp_1 = inputs[index_1 * embedding_dim + out_col_1];
    const int tmp_2 = inputs[index_2 * embedding_dim + out_col_2];
    const int tmp_3 = inputs[index_3 * embedding_dim + out_col_3];
    const int tmp_4 = inputs[index_4 * embedding_dim + out_col_4];

    outputs[out_row_0 * embedding_dim + out_col_0] = tmp_0;
    outputs[out_row_1 * embedding_dim + out_col_1] = tmp_1;
    outputs[out_row_2 * embedding_dim + out_col_2] = tmp_2;
    outputs[out_row_3 * embedding_dim + out_col_3] = tmp_3;
    outputs[out_row_4 * embedding_dim + out_col_4] = tmp_4;

  }else if (cur_thread_3 < num_indices * embedding_dim){
    const int out_row_0 = cur_thread_0 / embedding_dim;
    const int out_row_1 = cur_thread_1 / embedding_dim;
    const int out_row_2 = cur_thread_2 / embedding_dim;
    const int out_row_3 = cur_thread_3 / embedding_dim;

    const int out_col_0 = cur_thread_0 % embedding_dim;
    const int out_col_1 = cur_thread_1 % embedding_dim;
    const int out_col_2 = cur_thread_2 % embedding_dim;
    const int out_col_3 = cur_thread_3 % embedding_dim;

    const int index_0 = indices[out_row_0];
    const int index_1 = indices[out_row_1];
    const int index_2 = indices[out_row_2];
    const int index_3 = indices[out_row_3];

    const int tmp_0 = inputs[index_0 * embedding_dim + out_col_0];
    const int tmp_1 = inputs[index_1 * embedding_dim + out_col_1];
    const int tmp_2 = inputs[index_2 * embedding_dim + out_col_2];
    const int tmp_3 = inputs[index_3 * embedding_dim + out_col_3];

    outputs[out_row_0 * embedding_dim + out_col_0] = tmp_0;
    outputs[out_row_1 * embedding_dim + out_col_1] = tmp_1;
    outputs[out_row_2 * embedding_dim + out_col_2] = tmp_2;
    outputs[out_row_3 * embedding_dim + out_col_3] = tmp_3;

  }else if (cur_thread_2 < num_indices * embedding_dim){
    const int out_row_0 = cur_thread_0 / embedding_dim;
    const int out_row_1 = cur_thread_1 / embedding_dim;
    const int out_row_2 = cur_thread_2 / embedding_dim;

    const int out_col_0 = cur_thread_0 % embedding_dim;
    const int out_col_1 = cur_thread_1 % embedding_dim;
    const int out_col_2 = cur_thread_2 % embedding_dim;

    const int index_0 = indices[out_row_0];
    const int index_1 = indices[out_row_1];
    const int index_2 = indices[out_row_2];

    const int tmp_0 = inputs[index_0 * embedding_dim + out_col_0];
    const int tmp_1 = inputs[index_1 * embedding_dim + out_col_1];
    const int tmp_2 = inputs[index_2 * embedding_dim + out_col_2];

    outputs[out_row_0 * embedding_dim + out_col_0] = tmp_0;
    outputs[out_row_1 * embedding_dim + out_col_1] = tmp_1;
    outputs[out_row_2 * embedding_dim + out_col_2] = tmp_2;

  }else if (cur_thread_1 < num_indices * embedding_dim){
    const int out_row_0 = cur_thread_0 / embedding_dim;
    const int out_row_1 = cur_thread_1 / embedding_dim;

    const int out_col_0 = cur_thread_0 % embedding_dim;
    const int out_col_1 = cur_thread_1 % embedding_dim;

    const int index_0 = indices[out_row_0];
    const int index_1 = indices[out_row_1];

    const int tmp_0 = inputs[index_0 * embedding_dim + out_col_0];
    const int tmp_1 = inputs[index_1 * embedding_dim + out_col_1];

    outputs[out_row_0 * embedding_dim + out_col_0] = tmp_0;
    outputs[out_row_1 * embedding_dim + out_col_1] = tmp_1;

  }else if (cur_thread_0 < num_indices * embedding_dim){
    const int out_row_0 = cur_thread_0 / embedding_dim;

    const int out_col_0 = cur_thread_0 % embedding_dim;

    const int index_0 = indices[out_row_0];

    outputs[out_row_0 * embedding_dim + out_col_0] = inputs[index_0 * embedding_dim + out_col_0];

  }
}

template <typename scalar_t>
__global__ void gather_v_i32_indices_cuda_kernel_unroll4(
    const scalar_t* __restrict__ inputs,
    const int32_t* __restrict__ indices,
    scalar_t* __restrict__ outputs,
    const int embedding_dim,
    const int num_indices) {

  const int cur_thread_0 = 4 * blockIdx.x * blockDim.x + 4 * threadIdx.x;
  const int cur_thread_1 = 4 * blockIdx.x * blockDim.x + 4 * threadIdx.x + 1;
  const int cur_thread_2 = 4 * blockIdx.x * blockDim.x + 4 * threadIdx.x + 2;
  const int cur_thread_3 = 4 * blockIdx.x * blockDim.x + 4 * threadIdx.x + 3;

  if (cur_thread_3 < num_indices * embedding_dim){
    const int out_row_0 = cur_thread_0 / embedding_dim;
    const int out_row_1 = cur_thread_1 / embedding_dim;
    const int out_row_2 = cur_thread_2 / embedding_dim;
    const int out_row_3 = cur_thread_3 / embedding_dim;

    const int out_col_0 = cur_thread_0 % embedding_dim;
    const int out_col_1 = cur_thread_1 % embedding_dim;
    const int out_col_2 = cur_thread_2 % embedding_dim;
    const int out_col_3 = cur_thread_3 % embedding_dim;

    const int index_0 = indices[out_row_0];
    const int index_1 = indices[out_row_1];
    const int index_2 = indices[out_row_2];
    const int index_3 = indices[out_row_3];

    const int tmp_0 = inputs[index_0 * embedding_dim + out_col_0];
    const int tmp_1 = inputs[index_1 * embedding_dim + out_col_1];
    const int tmp_2 = inputs[index_2 * embedding_dim + out_col_2];
    const int tmp_3 = inputs[index_3 * embedding_dim + out_col_3];

    outputs[out_row_0 * embedding_dim + out_col_0] = tmp_0;
    outputs[out_row_1 * embedding_dim + out_col_1] = tmp_1;
    outputs[out_row_2 * embedding_dim + out_col_2] = tmp_2;
    outputs[out_row_3 * embedding_dim + out_col_3] = tmp_3;

  }else if (cur_thread_2 < num_indices * embedding_dim){
    const int out_row_0 = cur_thread_0 / embedding_dim;
    const int out_row_1 = cur_thread_1 / embedding_dim;
    const int out_row_2 = cur_thread_2 / embedding_dim;

    const int out_col_0 = cur_thread_0 % embedding_dim;
    const int out_col_1 = cur_thread_1 % embedding_dim;
    const int out_col_2 = cur_thread_2 % embedding_dim;

    const int index_0 = indices[out_row_0];
    const int index_1 = indices[out_row_1];
    const int index_2 = indices[out_row_2];

    const int tmp_0 = inputs[index_0 * embedding_dim + out_col_0];
    const int tmp_1 = inputs[index_1 * embedding_dim + out_col_1];
    const int tmp_2 = inputs[index_2 * embedding_dim + out_col_2];

    outputs[out_row_0 * embedding_dim + out_col_0] = tmp_0;
    outputs[out_row_1 * embedding_dim + out_col_1] = tmp_1;
    outputs[out_row_2 * embedding_dim + out_col_2] = tmp_2;

  }else if (cur_thread_1 < num_indices * embedding_dim){
    const int out_row_0 = cur_thread_0 / embedding_dim;
    const int out_row_1 = cur_thread_1 / embedding_dim;

    const int out_col_0 = cur_thread_0 % embedding_dim;
    const int out_col_1 = cur_thread_1 % embedding_dim;

    const int index_0 = indices[out_row_0];
    const int index_1 = indices[out_row_1];

    const int tmp_0 = inputs[index_0 * embedding_dim + out_col_0];
    const int tmp_1 = inputs[index_1 * embedding_dim + out_col_1];

    outputs[out_row_0 * embedding_dim + out_col_0] = tmp_0;
    outputs[out_row_1 * embedding_dim + out_col_1] = tmp_1;

  }else if (cur_thread_0 < num_indices * embedding_dim){
    const int out_row_0 = cur_thread_0 / embedding_dim;

    const int out_col_0 = cur_thread_0 % embedding_dim;

    const int index_0 = indices[out_row_0];

    outputs[out_row_0 * embedding_dim + out_col_0] = inputs[index_0 * embedding_dim + out_col_0];
  }
}

template <typename scalar_t>
__global__ void gather_v_i32_indices_cuda_kernel_unroll2(
    const scalar_t* __restrict__ inputs,
    const int32_t* __restrict__ indices,
    scalar_t* __restrict__ outputs,
    const int embedding_dim,
    const int num_indices) {

  const int cur_thread_0 = 2 * blockIdx.x * blockDim.x + 2 * threadIdx.x;
  const int cur_thread_1 = 2 * blockIdx.x * blockDim.x + 2 * threadIdx.x + 1;
  if (cur_thread_1 < num_indices * embedding_dim){
    const int out_row_0 = cur_thread_0 / embedding_dim;
    const int out_row_1 = cur_thread_1 / embedding_dim;

    const int out_col_0 = cur_thread_0 % embedding_dim;
    const int out_col_1 = cur_thread_1 % embedding_dim;

    const int index_0 = indices[out_row_0];
    const int index_1 = indices[out_row_1];

    const int tmp_0 = inputs[index_0 * embedding_dim + out_col_0];
    const int tmp_1 = inputs[index_1 * embedding_dim + out_col_1];

    outputs[out_row_0 * embedding_dim + out_col_0] = tmp_0;
    outputs[out_row_1 * embedding_dim + out_col_1] = tmp_1;

  }else if (cur_thread_0 < num_indices * embedding_dim){
    const int out_row_0 = cur_thread_0 / embedding_dim;

    const int out_col_0 = cur_thread_0 % embedding_dim;

    const int index_0 = indices[out_row_0];

    outputs[out_row_0 * embedding_dim + out_col_0] = inputs[index_0 * embedding_dim + out_col_0];
  }
}

template <typename scalar_t>
__global__ void gather_v_i32_indices_cuda_kernel(
    const scalar_t* __restrict__ inputs,
    const int32_t* __restrict__ indices,
    scalar_t* __restrict__ outputs,
    const int embedding_dim,
    const int num_indices) {

  const int cur_thread = blockIdx.x * blockDim.x + threadIdx.x;
  if (cur_thread < num_indices * embedding_dim){
    const int out_row = cur_thread / embedding_dim;
    const int out_col = cur_thread % embedding_dim;
    const int index = indices[out_row];
    outputs[out_row * embedding_dim + out_col] = inputs[index * embedding_dim + out_col];
  }
}

torch::Tensor gather_v_i32_indices_cuda(torch::Tensor inputs,torch::Tensor indices) {
  const auto embedding_dim = inputs.size(1);
  const auto num_indices = indices.size(0);

  auto outputs = torch::empty({num_indices, embedding_dim}, inputs.options());

  cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 32);

  const int threads = 128;
  const int blocks = (num_indices * embedding_dim + threads - 1) / threads / 4;
  AT_DISPATCH_FLOATING_TYPES(inputs.scalar_type(), "gather_v_i32_indices_cuda", ([&] {
    gather_v_i32_indices_cuda_kernel_unroll4<scalar_t><<<blocks, threads>>>(
        inputs.data_ptr<scalar_t>(),
        indices.data_ptr<int32_t>(),
        outputs.data_ptr<scalar_t>(),
        embedding_dim,
        num_indices
        );
  }));

  return outputs;
}