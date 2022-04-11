#include <math.h>
#include <stdio.h>

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
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0;
    float value = 0.0;
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

    printf("%f \n", output[idx]);
}

void forward_pass(double data[28][28])
{
}

int main()
{
    double data[28][28];
    forward_pass(data);
}