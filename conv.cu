#include <math.h>
#include <stdio.h>
#include <algorithm>    // std::max
#include<cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#define WEIGHT_SIZE 5
class Layer
{
public:
    int M, N, O; // O: output 6x24x24x, N: #feature 6, M: #params_per_feature 5x5
    float* pre_output, * output;
    float* weight, * bias;
    float* w_h, * b_h;
    float* output_h, * pre_output_h;

    Layer(int M, int N, int O) {

        this->M = M;
        this->N = N;
        this->O = O;

        cudaMalloc((void**)&this->weight, M * N * sizeof(float));
        cudaMalloc((void**)&this->bias, N * sizeof(float));
        cudaMalloc((void**)&this->pre_output, O * sizeof(float));
        cudaMalloc((void**)&this->output, O * sizeof(float));

        this->b_h = (float*)malloc(N * sizeof(float));
        this->w_h = (float*)malloc(M * N * sizeof(float));
        this->output_h = (float*)malloc(O * sizeof(float));
        this->pre_output_h = (float*)malloc(O * sizeof(float));

        for (int i = 0; i < M * N; i++) {
            if (i < N) this->b_h[i] = -1.0f;
            this->w_h[i] = -1.0f;
        }


        cudaMemcpy(this->weight, this->w_h, M * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(this->bias, this->b_h, N * sizeof(float), cudaMemcpyHostToDevice);
       
    }

    void cpyDeviceToHost(float* o_h, float* o_d) {
        cudaMemcpy(o_h, o_d, O * sizeof(float), cudaMemcpyDeviceToHost);
    }


    void print(float* out) {
        for (int i = 0; i < 6; i++) {
            printf("z = %d\n", i);
            for (int j = 0; j < 24; j++) {
                for (int k = 0; k < 24; k++) {
                    printf("%f ", out[k + j * 24 + i * 24 * 24]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }

    ~Layer() {
        cudaFree(weight);
        cudaFree(bias);
        cudaFree(pre_output);
        cudaFree(output);
        free(w_h);
        free(b_h);
        free(output_h);
        free(pre_output_h);
    }

};


__global__ void kernel_conv_filter(float* input, float* pre_output, float* weight)
{
    int row = threadIdx.y;
    int col = threadIdx.x;

    float temp = 0;

    for (int i = 0; i < WEIGHT_SIZE; i++)
    {
        for (int j = 0; j < WEIGHT_SIZE; j++)
        {
            int input_idx = (row + i) * 28 + (col + j);
            int block_idx = blockIdx.x * 25 + i * WEIGHT_SIZE + j;
            temp += input[input_idx] * weight[block_idx];
        }
    }
    int idx = blockIdx.x * 576 + row * 24 + col;
    pre_output[idx] = temp;
}

__global__ void kernel_conv_bias(float* pre_output, float* bias)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int idz = threadIdx.z + blockIdx.z * blockDim.z;

    pre_output[idx + idy * 24 + idz * 24 * 24] = pre_output[idx + idy * 24 + idz * 24 * 24] + bias[idz];

}

__global__ void kernel_conv_sigmoid(float* pre_output, float* output)
{
    int idx = blockIdx.x * 576 + 24 * threadIdx.x + threadIdx.y;

    output[idx] = 1.0 / (1.0 + exp(-pre_output[idx]));
}

// Inputs are: arr - array, val - asseted value, size will be dimx * dimy * dimz
void maxError(float* arr, float val, float size)
{
    float maxErr = 0;
    for (int i = 0; i < size; i++)
    {
        if (arr[i] != val)
        {
            maxErr = std::max(maxErr, abs(arr[i] - val));
        }
    }
    printf("maxEror = %f (asserted with %f)\n", maxErr, val);
}

void forward_pass(float* data, Layer* layer)
{
    float* data_d;
    cudaMalloc((void**)&data_d, 28 * 28 * sizeof(28));
    cudaMemcpy(data_d, data, 28 * 28 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blocks(1, 1, 6);
    dim3 threads(24, 24, 1);
    kernel_conv_filter << <blocks, threads >> > (data_d, layer->pre_output, layer->weight);
    layer->cpyDeviceToHost(layer->pre_output_h, layer->pre_output);
    maxError(layer->pre_output_h, 25.0f, 24*24*6);
    
    kernel_conv_bias <<<blocks, threads >>> (layer->pre_output, layer->bias);
    layer->cpyDeviceToHost(layer->pre_output_h, layer->pre_output);
    maxError(layer->pre_output_h, 24.0f, 24 * 24 * 6);
    
    kernel_conv_sigmoid << < blocks, threads >> > (layer->pre_output, layer->output);
    layer->cpyDeviceToHost(layer->output_h, layer->output);
    maxError(layer->output_h, 1.0f, 24 * 24 * 6);

    cudaFree(data_d);
}

int main()
{
    float data[28 * 28];
    for (int i = 0; i < 28 * 28; i++) data[i] = -1.0f;

    Layer layer(5 * 5, 6, 6 * 24 * 24);
    forward_pass(data, &layer);

    return 0;
}