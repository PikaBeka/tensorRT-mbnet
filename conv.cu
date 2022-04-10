class Layer
{
public:
    int M, N, O; // O: output, N: #feature, M: #params_per_feature
    float *pre_output, *output;
    float *weight, *bias;
    Layer(int M, int N, int O);
    ~Layer();
};

__global__ void kernel_conv_filter(float *input, float *pre_output, float *weight)
{
}

__global__ void kernel_conv_bias(float *pre_output, float *bias)
{
}

__global__ void kernel_conv_sigmoid(float *pre_output, float *output)
{
}

void forward_pass(double data[28][28])
{
}

int main()
{
}