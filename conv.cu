#include <math.h>
#include <stdio.h>

#define WEIGHT_SIZE 5
class Layer
{
public:
    int M, N, O; // O: output, N: #feature, M: #params_per_feature
    float *pre_output, *output;
    float *weight, *bias;
    Layer(int M, int N, int O);
    ~Layer();
};

Layer::~Layer()
{
}

__global__ void kernel_conv_filter(float *input, float *pre_output, float *weight)
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

__global__ void kernel_conv_bias(float *pre_output, float *bias)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int idz = threadIdx.z + blockIdx.z * blockDim.z;

    pre_output[idx + idy * 24 + idz * 24 * 24] = pre_output[idx + idy * 24 + idz * 24 * 24] + bias[idz];

}

__global__ void kernel_conv_sigmoid(float *pre_output, float *output)
{
    int idx = blockIdx.x * 576 + 24 * threadIdx.x + threadIdx.y;

    output[idx] = 1.0 / (1.0 + exp(-pre_output[idx]));
}

// Inputs are: arr - array, val - asseted value, size will be dimx * dimy * dimz
void maxError(float *arr, float val, float size)
{
    float maxErr = 0;
    for (int i = 0; i < size; i++)
    {
        if (arr[i] != val)
        {
            maxErr = max(maxErr, abs(arr[i] - val));
        }
    }
    printf("maxEror = %f (asserted with %f)\n", maxErr, val);
}

void forward_pass(double data[28][28])
{
    int THREADS = 24;
    int BLOCKS = 6;
    dim3 numBlocks(BLOCKS, BLOCKS);
    dim3 threadPerBlock(THREADS, THREADS);
    kernel_conv_filter<<<numBlocks, threadPerBlock>>>(d_input, d_output, d_weight);

    kernel_conv_sigmoid<<<6, dim3(24, 24)>>>()
}

int main()
{
    double data[28][28];
    forward_pass(data);
}