#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

// template <typename scalar_t>
// __global__ void scatter_v_i32_indices_cuda_kernel_unroll8(
//     const scalar_t* __restrict__ inputs,
//     const int32_t* __restrict__ indices,
//     scalar_t* __restrict__ outputs,
//     const int embedding_dim,
//     const int num_indices) {

//   const int cur_thread_0 = 8 * blockIdx.x * blockDim.x + 8 * threadIdx.x;
//   const int cur_thread_1 = 8 * blockIdx.x * blockDim.x + 8 * threadIdx.x + 1;
//   const int cur_thread_2 = 8 * blockIdx.x * blockDim.x + 8 * threadIdx.x + 2;
//   const int cur_thread_3 = 8 * blockIdx.x * blockDim.x + 8 * threadIdx.x + 3;
//   const int cur_thread_0 = 8 * blockIdx.x * blockDim.x + 8 * threadIdx.x + 4;
//   const int cur_thread_1 = 8 * blockIdx.x * blockDim.x + 8 * threadIdx.x + 5;
//   const int cur_thread_2 = 8 * blockIdx.x * blockDim.x + 8 * threadIdx.x + 6;
//   const int cur_thread_3 = 8 * blockIdx.x * blockDim.x + 8 * threadIdx.x + 7;

//   if (cur_thread_3 < num_indices * embedding_dim){
//     const int in_row_0 = cur_thread_0 / embedding_dim;
//     const int in_row_1 = cur_thread_1 / embedding_dim;
//     const int in_row_2 = cur_thread_2 / embedding_dim;
//     const int in_row_3 = cur_thread_3 / embedding_dim;
//     const int in_row_4 = cur_thread_4 / embedding_dim;
//     const int in_row_5 = cur_thread_5 / embedding_dim;
//     const int in_row_6 = cur_thread_6 / embedding_dim;
//     const int in_row_7 = cur_thread_7 / embedding_dim;

//     const int in_col_0 = cur_thread_0 % embedding_dim;
//     const int in_col_1 = cur_thread_1 % embedding_dim;
//     const int in_col_2 = cur_thread_2 % embedding_dim;
//     const int in_col_3 = cur_thread_3 % embedding_dim;
//     const int in_col_4 = cur_thread_4 % embedding_dim;
//     const int in_col_5 = cur_thread_5 % embedding_dim;
//     const int in_col_6 = cur_thread_6 % embedding_dim;
//     const int in_col_7 = cur_thread_7 % embedding_dim;

//     const int out_row_0 = indices[in_row_0];
//     const int out_row_1 = indices[in_row_1];
//     const int out_row_2 = indices[in_row_2];
//     const int out_row_3 = indices[in_row_3];
//     const int out_row_4 = indices[in_row_4];
//     const int out_row_5 = indices[in_row_5];
//     const int out_row_6 = indices[in_row_6];
//     const int out_row_7 = indices[in_row_7];

//     outputs[out_row_0 * embedding_dim + in_col_0] = inputs[in_row_0 * embedding_dim + in_col_0];
//     outputs[out_row_1 * embedding_dim + in_col_1] = inputs[in_row_1 * embedding_dim + in_col_1];
//     outputs[out_row_2 * embedding_dim + in_col_2] = inputs[in_row_2 * embedding_dim + in_col_2];
//     outputs[out_row_3 * embedding_dim + in_col_3] = inputs[in_row_3 * embedding_dim + in_col_3];
//     outputs[out_row_4 * embedding_dim + in_col_4] = inputs[in_row_4 * embedding_dim + in_col_4];
//     outputs[out_row_5 * embedding_dim + in_col_5] = inputs[in_row_5 * embedding_dim + in_col_5];
//     outputs[out_row_6 * embedding_dim + in_col_6] = inputs[in_row_6 * embedding_dim + in_col_6];
//     outputs[out_row_7 * embedding_dim + in_col_7] = inputs[in_row_7 * embedding_dim + in_col_7];

//   }else if (cur_thread_2 < num_indices * embedding_dim){
//     const int in_row_0 = cur_thread_0 / embedding_dim;
//     const int in_row_1 = cur_thread_1 / embedding_dim;
//     const int in_row_2 = cur_thread_2 / embedding_dim;

//     const int in_col_0 = cur_thread_0 % embedding_dim;
//     const int in_col_1 = cur_thread_1 % embedding_dim;
//     const int in_col_2 = cur_thread_2 % embedding_dim;

//     const int out_row_0 = indices[in_row_0];
//     const int out_row_1 = indices[in_row_1];
//     const int out_row_2 = indices[in_row_2];

//     outputs[out_row_0 * embedding_dim + in_col_0] = inputs[in_row_0 * embedding_dim + in_col_0];
//     outputs[out_row_1 * embedding_dim + in_col_1] = inputs[in_row_1 * embedding_dim + in_col_1];
//     outputs[out_row_2 * embedding_dim + in_col_2] = inputs[in_row_2 * embedding_dim + in_col_2];

//   }else if (cur_thread_1 < num_indices * embedding_dim){
//     const int in_row_0 = cur_thread_0 / embedding_dim;
//     const int in_row_1 = cur_thread_1 / embedding_dim;

//     const int in_col_0 = cur_thread_0 % embedding_dim;
//     const int in_col_1 = cur_thread_1 % embedding_dim;

//     const int out_row_0 = indices[in_row_0];
//     const int out_row_1 = indices[in_row_1];

//     outputs[out_row_0 * embedding_dim + in_col_0] = inputs[in_row_0 * embedding_dim + in_col_0];
//     outputs[out_row_1 * embedding_dim + in_col_1] = inputs[in_row_1 * embedding_dim + in_col_1];

//   }else if (cur_thread_0 < num_indices * embedding_dim){
//     const int in_row_0 = cur_thread_0 / embedding_dim;

//     const int in_col_0 = cur_thread_0 % embedding_dim;

//     const int out_row_0 = indices[in_row_0];

//     outputs[out_row_0 * embedding_dim + in_col_0] = inputs[in_row_0 * embedding_dim + in_col_0];

//   }
// }

template <typename scalar_t>
__global__ void scatter_v_i32_indices_cuda_kernel_unroll4(
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
    const int in_row_0 = cur_thread_0 / embedding_dim;
    const int in_row_1 = cur_thread_1 / embedding_dim;
    const int in_row_2 = cur_thread_2 / embedding_dim;
    const int in_row_3 = cur_thread_3 / embedding_dim;

    const int in_col_0 = cur_thread_0 % embedding_dim;
    const int in_col_1 = cur_thread_1 % embedding_dim;
    const int in_col_2 = cur_thread_2 % embedding_dim;
    const int in_col_3 = cur_thread_3 % embedding_dim;

    const int out_row_0 = indices[in_row_0];
    const int out_row_1 = indices[in_row_1];
    const int out_row_2 = indices[in_row_2];
    const int out_row_3 = indices[in_row_3];

    const int tmp_0 = inputs[in_row_0 * embedding_dim + in_col_0];
    const int tmp_1 = inputs[in_row_1 * embedding_dim + in_col_1];
    const int tmp_2 = inputs[in_row_2 * embedding_dim + in_col_2];
    const int tmp_3 = inputs[in_row_3 * embedding_dim + in_col_3];

    outputs[out_row_0 * embedding_dim + in_col_0] = tmp_0;
    outputs[out_row_1 * embedding_dim + in_col_1] = tmp_1;
    outputs[out_row_2 * embedding_dim + in_col_2] = tmp_2;
    outputs[out_row_3 * embedding_dim + in_col_3] = tmp_3;

  }else if (cur_thread_2 < num_indices * embedding_dim){
    const int in_row_0 = cur_thread_0 / embedding_dim;
    const int in_row_1 = cur_thread_1 / embedding_dim;
    const int in_row_2 = cur_thread_2 / embedding_dim;

    const int in_col_0 = cur_thread_0 % embedding_dim;
    const int in_col_1 = cur_thread_1 % embedding_dim;
    const int in_col_2 = cur_thread_2 % embedding_dim;

    const int out_row_0 = indices[in_row_0];
    const int out_row_1 = indices[in_row_1];
    const int out_row_2 = indices[in_row_2];

    const int tmp_0 = inputs[in_row_0 * embedding_dim + in_col_0];
    const int tmp_1 = inputs[in_row_1 * embedding_dim + in_col_1];
    const int tmp_2 = inputs[in_row_2 * embedding_dim + in_col_2];

    outputs[out_row_0 * embedding_dim + in_col_0] = tmp_0;
    outputs[out_row_1 * embedding_dim + in_col_1] = tmp_1;
    outputs[out_row_2 * embedding_dim + in_col_2] = tmp_2;

  }else if (cur_thread_1 < num_indices * embedding_dim){
    const int in_row_0 = cur_thread_0 / embedding_dim;
    const int in_row_1 = cur_thread_1 / embedding_dim;

    const int in_col_0 = cur_thread_0 % embedding_dim;
    const int in_col_1 = cur_thread_1 % embedding_dim;

    const int out_row_0 = indices[in_row_0];
    const int out_row_1 = indices[in_row_1];

    const int tmp_0 = inputs[in_row_0 * embedding_dim + in_col_0];
    const int tmp_1 = inputs[in_row_1 * embedding_dim + in_col_1];

    outputs[out_row_0 * embedding_dim + in_col_0] = tmp_0;
    outputs[out_row_1 * embedding_dim + in_col_1] = tmp_1;

  }else if (cur_thread_0 < num_indices * embedding_dim){
    const int in_row_0 = cur_thread_0 / embedding_dim;

    const int in_col_0 = cur_thread_0 % embedding_dim;

    const int out_row_0 = indices[in_row_0];

    outputs[out_row_0 * embedding_dim + in_col_0] = inputs[in_row_0 * embedding_dim + in_col_0];

  }
}

template <typename scalar_t>
__global__ void scatter_v_i32_indices_cuda_kernel_unroll2(
    const scalar_t* __restrict__ inputs,
    const int32_t* __restrict__ indices,
    scalar_t* __restrict__ outputs,
    const int embedding_dim,
    const int num_indices) {

  const int cur_thread_0 = 2 * blockIdx.x * blockDim.x + 2 * threadIdx.x;
  const int cur_thread_1 = 2 * blockIdx.x * blockDim.x + 2 * threadIdx.x + 1;

  if (cur_thread_1 < num_indices * embedding_dim){
    const int in_row_0 = cur_thread_0 / embedding_dim;
    const int in_row_1 = cur_thread_1 / embedding_dim;

    const int in_col_0 = cur_thread_0 % embedding_dim;
    const int in_col_1 = cur_thread_1 % embedding_dim;

    const int out_row_0 = indices[in_row_0];
    const int out_row_1 = indices[in_row_1];

    const int tmp_0 = inputs[in_row_0 * embedding_dim + in_col_0];
    const int tmp_1 = inputs[in_row_1 * embedding_dim + in_col_1];

    outputs[out_row_0 * embedding_dim + in_col_0] = tmp_0;
    outputs[out_row_1 * embedding_dim + in_col_1] = tmp_1;

  }else if (cur_thread_0 < num_indices * embedding_dim){
    const int in_row_0 = cur_thread_0 / embedding_dim;

    const int in_col_0 = cur_thread_0 % embedding_dim;

    const int out_row_0 = indices[in_row_0];

    outputs[out_row_0 * embedding_dim + in_col_0] = inputs[in_row_0 * embedding_dim + in_col_0];

  }
}

template <typename scalar_t>
__global__ void scatter_v_i32_indices_cuda_kernel(
    const scalar_t* __restrict__ inputs,
    const int32_t* __restrict__ indices,
    scalar_t* __restrict__ outputs,
    const int embedding_dim,
    const int num_indices) {

  const int cur_thread = blockIdx.x * blockDim.x + threadIdx.x;
  if (cur_thread < num_indices * embedding_dim){
    const int in_row = cur_thread / embedding_dim;
    const int in_col = cur_thread % embedding_dim;
    const int out_row = indices[in_row];
    outputs[out_row * embedding_dim + in_col] = inputs[in_row * embedding_dim + in_col];
  }
}

torch::Tensor scatter_v_i32_indices_cuda(torch::Tensor inputs, torch::Tensor indices, torch::Tensor outputs) {
  const auto embedding_dim = inputs.size(1);
  const auto num_indices = indices.size(0);

  // cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 32);

  const int threads = 128;
  const int blocks = (num_indices * embedding_dim + threads/4 - 1) / threads/4;
  AT_DISPATCH_FLOATING_TYPES(inputs.scalar_type(), "scatter_v_i32_indices_cuda", ([&] {
    scatter_v_i32_indices_cuda_kernel_unroll4<scalar_t><<<blocks, threads>>>(
        inputs.data_ptr<scalar_t>(),
        indices.data_ptr<int32_t>(),
        outputs.data_ptr<scalar_t>(),
        embedding_dim,
        num_indices
        );
  }));

  return outputs;
}