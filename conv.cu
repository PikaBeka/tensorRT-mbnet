#include <math.h>
#include <stdio.h>
#include <algorithm> // std::max
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "slenet_params.h"

#define WEIGHT_SIZE 5
#define POOL_SIZE 4
#define INSIZE 28
/*Paths to images*/
#define IMAGE "./data/t10k-images.idx3-ubyte"
#define LABEL "./data/t10k-labels.idx1-ubyte"
/*Number of images*/
#define SIZE 10000
/*How many images to print, Change it to test*/
#define TEST_NUMER 4

typedef struct mnist_data
{
    double data[INSIZE][INSIZE];
    unsigned int label;
} mnist_data;
class Layer
{
public:
    int M, N, O;
    float *pre_output, *output;
    float *weight, *bias;
    float *w_h, *b_h;
    float *output_h, *pre_output_h;

    Layer(int M, int N, int O)
    {

        this->M = M;
        this->N = N;
        this->O = O;

        cudaMalloc((void **)&this->weight, M * N * sizeof(float));
        cudaMalloc((void **)&this->bias, N * sizeof(float));
        cudaMalloc((void **)&this->pre_output, O * sizeof(float));
        cudaMalloc((void **)&this->output, O * sizeof(float));

        this->b_h = (float *)malloc(N * sizeof(float));
        this->w_h = (float *)malloc(M * N * sizeof(float));
        this->output_h = (float *)malloc(O * sizeof(float));
        this->pre_output_h = (float *)malloc(O * sizeof(float));

        initWeightBias(1.0f, 1.0f);
    }

    void initWeightBias(float w, float b)
    {
        for (int i = 0; i < M * N; i++)
        {
            if (i < N)
                this->b_h[i] = b;
            this->w_h[i] = w;
        }

        cudaMemcpy(this->weight, this->w_h, M * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(this->bias, this->b_h, N * sizeof(float), cudaMemcpyHostToDevice);
    }

    void cpyDeviceToHost(float *o_h, float *o_d, size_t size)
    {
        cudaMemcpy(o_h, o_d, size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    void print(float *out, size_t size)
    {
        for (int i = 0; i < size; i++)
        {
            printf("%d) %f\n", i, out[i]);
        }
    }

    ~Layer()
    {
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

static unsigned int mnist_bin_to_int(char *tmp)
{
    unsigned int res;

    // 4 bytes, so each is 8 bits. total 32;
    for (int i = 0; i < 4; i++)
    {
        res <<= 8;
        res = res | (unsigned char)tmp[i];
    }
    return res;
}

static int mnist_load(const char *image_filename, const char *label_filename,
                      mnist_data **data_set, unsigned int *count)
{
    /*Open image file*/
    FILE *image_file = fopen(image_filename, "rb");
    if (image_file == NULL)
    {
        printf("Image file was not found. Please check the location\n");
        fputs("File error", stderr);
        exit(1);
    }

    /*Open label file*/
    FILE *label_file = fopen(label_filename, "rb");
    if (label_file == NULL)
    {
        printf("Label file was not found. Please check the location\n");
        fputs("File error", stderr);
        exit(1);
    }

    int image_data[4], label_data[2];

    /*Loads essential data from image file*/
    for (int i = 0; i < 4; i++)
    {
        char toRead[4];
        fread(toRead, 1, 4, image_file);
        image_data[i] = mnist_bin_to_int(toRead);
    }

    /*Loads essential data from label file*/
    for (int i = 0; i < 2; i++)
    {
        char toRead[4];
        fread(toRead, 1, 4, label_file);
        label_data[i] = mnist_bin_to_int(toRead);
    }

    /*Prints checked values*/
    printf("image magic number = %d (should be 2051)\n", image_data[0]);
    printf("label magic number = %d (should be 2049)\n", label_data[0]);
    printf("image total number = %d (should be 10000)\n", image_data[1]);
    printf("label total number = %d (should be 10000)\n", label_data[1]);
    printf("rows = %d, cols = %d (both should be 28)\n", image_data[2], image_data[3]);

    /*Initializes the buffers with the count of images*/

    int cnt = 0;
    unsigned char labels[10000];
    unsigned char image[10000 * 28 * 28];

    /*Read all labels and iamges*/
    fread(labels, 1, label_data[1], label_file);
    fread(image, 1, image_data[1] * image_data[2] * image_data[3], image_file);

    /*iterate all images and labels*/
    for (int i = 0; i < image_data[1]; i++)
    {
        mnist_data *d = &(*data_set)[i]; // pointer to the memory of the first element
        d->label = labels[i];            // store label
        for (int j = 0; j < INSIZE; j++)
        {
            for (int k = 0; k < INSIZE; k++)
            {
                d->data[j][k] = image[i * INSIZE * INSIZE + j * INSIZE + k] / 255.0;
            }
        }
        cnt++;
    }
    *count = cnt;

    /*Close files*/
    fclose(label_file);
    fclose(image_file);

    return 0;
}

// Printing MNIST data set examples
void printExamples(mnist_data **data_set, int count)
{
    for (int i = 0; i < count; i++)
    {
        printf("\nImage:\n");

        for (int j = 0; j < INSIZE; j++)
        {
            for (int k = 0; k < INSIZE; k++)
            {
                printf("%.3f ", (*data_set)[i].data[j][k]);
            }
            printf("\n");
        }

        printf("Label: %d\n", (*data_set)[i].label);
    }
}

// end of TEST

// CONV LAYER
__global__ void kernel_conv_filter(float *input, float *pre_output, float *weight)
{
    int row = threadIdx.y;
    int col = threadIdx.x;
    float temp = 0.0;
    __shared__ float shm[28][28];
    __shared__ float shw[5][5];
    shm[row][col] = input[row * 28 + col];

    if (row < 5 && col < 5)
    {
        shw[row][col] = weight[blockIdx.z * 25 + row * 5 + col];
    }

    __syncthreads();

    if (row < 24 && col < 24)
    {
        for (int i = 0; i < WEIGHT_SIZE; i++)
        {
            for (int j = 0; j < WEIGHT_SIZE; j++)
            {
                // int input_idx = (row + i) * 28 + (col + j);
                // int block_idx = blockIdx.z * 25 + i * WEIGHT_SIZE + j;
                temp += shm[row + i][col + j] * shw[i][j];
            }
        }
        int idx = blockIdx.z * 576 + row * 24 + col;
        pre_output[idx] = temp;
    }
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
    int idx = blockIdx.z * 576 + 24 * threadIdx.y + threadIdx.x;

    output[idx] = 1 / (1 + exp(-pre_output[idx]));
}

// SUBSAMPLING LAYER
__global__ void kernel_ss1_filter(float *input, float *pre_output, float *weight)
{
    int row = threadIdx.y;
    int col = threadIdx.x;

    float tmp = 0;

    int inp_r = row * 4;
    int inp_c = col * 4;
    for (int i = 0; i < POOL_SIZE; i++)
    {
        for (int j = 0; j < POOL_SIZE; j++)
        {
            int input_idx = (inp_r + i) * 24 + inp_c + j;
            int weight_idx = i * POOL_SIZE + j;
            tmp += input[blockIdx.z * 24 * 24 + input_idx] * weight[weight_idx];
        }
    }
    int idx = blockIdx.z * 36 + row * 6 + col;
    pre_output[idx] = tmp;
}

__global__ void kernel_ss1_bias(float *pre_output, float *bias)
{
    pre_output[threadIdx.x + threadIdx.y * 6 + blockIdx.z * 36] += bias[0];
}

__global__ void kernel_ss1_sigmoid(float *pre_output, float *output)
{
    int idx = blockIdx.z * 36 + 6 * threadIdx.y + threadIdx.x;

    output[idx] = 1 / (1 + exp(-pre_output[idx]));
}

// FULLY CONNECTED LAYER
__global__ void kernel_fc1(float *input, float *pre_output, float *weight)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int c = idx % 10;
    float tempC = 0.0f;

    __shared__ float shm[28][28];
    __shared__ float shw[5][5];

    for (int i = 0; i < 6; i++)
    {
        for (int j = 0; j < 6; j++)
        {
            for (int k = 0; k < 6; k++)
            {
                tempC += weight[c * 6 * 6 * 6 + i * 6 * 6 + j * 6 + k] * input[i * 6 * 6 + j * 6 + k];
            }
        }
    }

    pre_output[c] = tempC;
}

__global__ void kernel_fc1_bias(float *pre_output, float *bias)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x < 10)
        pre_output[x] += bias[x];
}

__global__ void kernel_fc1_sigmoid(float *pre_output, float *output)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x < 10)
        output[x] = 1 / (1 + exp(-pre_output[x]));
}

// Inputs are: arr - array, val - asseted value, size will be dimx * dimy * dimz
void maxError(float *arr, float val, float size)
{
    float maxErr = 0;
    for (int i = 0; i < size; i++)
    {
        printf("%f \n", arr[i]);
    }
    // printf("maxEror = %f (asserted with %f)\n", maxErr, val);
}

Layer conv_layer(5 * 5, 6, 6 * 24 * 24);
Layer pool_layer(4 * 4, 1, 6 * 6 * 6);
Layer fl_layer(6 * 6 * 6, 10, 10);

void cpyTrainedValues()
{
    cudaMemcpy(conv_layer.weight, c1_weight, sizeof(float) * 6 * 5 * 5, cudaMemcpyHostToDevice);
    cudaMemcpy(conv_layer.bias, c1_bias, sizeof(float) * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(pool_layer.weight, s2_weight, sizeof(float) * 4 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(pool_layer.bias, s2_bias, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(fl_layer.weight, f3_weight, sizeof(float) * 10 * 6 * 6 * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(fl_layer.bias, f3_bias, sizeof(float) * 10, cudaMemcpyHostToDevice);
}

static double forward_pass(double data[28][28])
{

    float input[28 * 28];

    for (int i = 0; i < 28; i++)
    {
        for (int j = 0; j < 28; j++)
            input[i * 28 + j] = data[i][j];
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    float *data_d;
    cudaMalloc((void **)&data_d, 28 * 28 * sizeof(float));
    cudaMemcpy(data_d, input, 28 * 28 * sizeof(float), cudaMemcpyHostToDevice);

    /*28x28 * 6@5x5 = 6@24x24*/
    dim3 conv_blocks(1, 1, 6);
    dim3 conv_threads(28, 28, 1);
    kernel_conv_filter<<<conv_blocks, conv_threads>>>(data_d, conv_layer.pre_output, conv_layer.weight);

    // conv_layer.cpyDeviceToHost(conv_layer.pre_output_h, conv_layer.pre_output, 6 * 24 * 24);
    // printf("Verifying Convolutional filtering layer: ");
    // maxError(conv_layer.pre_output_h, 25.0f, 24 * 24 * 6);
    dim3 blocks(1, 1, 6);
    dim3 threads(24, 24, 1);
    kernel_conv_bias<<<blocks, threads>>>(conv_layer.pre_output, conv_layer.bias);
    // conv_layer.cpyDeviceToHost(conv_layer.pre_output_h, conv_layer.pre_output, 24 * 24 * 6);
    // printf("Verifying Convolutional bias layer: ");
    // maxError(conv_layer.pre_output_h, 26.0f, 24 * 24 * 6);

    kernel_conv_sigmoid<<<blocks, threads>>>(conv_layer.pre_output, conv_layer.output);
    // conv_layer.cpyDeviceToHost(conv_layer.output_h, conv_layer.output, 24 * 24 * 6);
    // printf("Verifying Convolutional sigmoid layer: ");
    // maxError(conv_layer.output_h, 1.0f, 24 * 24 * 6);

    /*6@24x24 * 1@4x4 = 6@6x6*/
    dim3 pool_blocks(1, 1, 6);
    dim3 pool_threads(6, 6, 1);
    kernel_ss1_filter<<<pool_blocks, pool_threads>>>(conv_layer.output, pool_layer.pre_output, pool_layer.weight);

    // kernel_ss1_filter << <pool_blocks, pool_threads >> > (conv_layer.output, pool_layer.pre_output, pool_layer.weight);
    // pool_layer.cpyDeviceToHost(pool_layer.pre_output_h, pool_layer.pre_output, 6 * 6 * 6);
    // printf("Verifying Subsampling filtering layer: ");
    // maxError(pool_layer.pre_output_h, 16.0f, 6 * 6 * 6);

    kernel_ss1_bias<<<pool_blocks, pool_threads>>>(pool_layer.pre_output, pool_layer.bias);
    // pool_layer.cpyDeviceToHost(pool_layer.pre_output_h, pool_layer.pre_output, 6 * 6 * 6);
    // printf("Verifying Subsampling bias layer: ");
    // maxError(pool_layer.pre_output_h, 17.0f, 6 * 6 * 6);

    kernel_ss1_sigmoid<<<pool_blocks, pool_threads>>>(pool_layer.pre_output, pool_layer.output);
    // pool_layer.cpyDeviceToHost(pool_layer.output_h, pool_layer.output, 6 * 6 * 6);
    // printf("Verifying Subsampling sigmoid layer: ");
    // maxError(pool_layer.output_h, 1.0f, 6 * 6 * 6);

    /*6@6x6 * 10@6x6x6 = 10@1x1*/
    kernel_fc1<<<10 * 216 / 16, 16>>>(pool_layer.output, fl_layer.pre_output, fl_layer.weight);
    // kernel_fc1 << <dim3(10, 1, 1), dim3(6, 6, 6) >> > (pool_layer.output, fl_layer.pre_output, fl_layer.weight);
    // fl_layer.cpyDeviceToHost(fl_layer.pre_output_h, fl_layer.pre_output, 10);
    // printf("Verifying Fully-Connected layer: ");
    // maxError(fl_layer.pre_output_h, 216.0f, 10);

    kernel_fc1_bias<<<1, 10>>>(fl_layer.pre_output, fl_layer.bias);
    // fl_layer.cpyDeviceToHost(fl_layer.pre_output_h, fl_layer.pre_output, 10);
    // printf("Verifying Fully-Connected bias layer: ");
    // maxError(fl_layer.pre_output_h, 217.0f, 10);

    kernel_fc1_sigmoid<<<1, 10>>>(fl_layer.pre_output, fl_layer.output);
    // fl_layer.cpyDeviceToHost(fl_layer.output_h, fl_layer.output, 10);
    // printf("Verifying Fully-Connected sigmoid layer: ");
    // maxError(fl_layer.output_h, 1.0f, 10);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(data_d);

    return elapsedTime;
}

int main()
{
    // double data[28][28];
    // for (int i = 0; i < 28; i++)
    //     for (int j = 0; j < 28; j++)
    //         data[i][j] = 1.0f;

    // forward_pass(data);

    // return 0;

    double time_taken = 0.0;
    int ret;
    int i;
    mnist_data *test_set = (mnist_data *)malloc(SIZE * sizeof(mnist_data));
    static unsigned int test_cnt = 1; // load data
    if (ret = mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", &test_set,
                         &test_cnt) != 0)
        printf("An error occurred: %d \n", ret);
    else
        printf("test_cnt = %d \n", test_cnt);

    cpyTrainedValues();
    // forward_pass(test_set[0].data);
    // return 0;
    unsigned int error = 0;
    unsigned int max = 0;
    float res[10];
    for (i = 0; i < test_cnt; i++)
    {
        time_taken += forward_pass(test_set[i].data);
        cudaMemcpy(res, fl_layer.output, sizeof(float) * 10, cudaMemcpyDeviceToHost);
        for (int j = 0; j < 10; j++)
        {
            if (res[max] < res[j])
                max = j;
        }
        if (max != test_set[i].label)
            ++error; // error must have the number of incorrect predictions.
    }

    printf("Error Rate = %f%% (%d out of 10,000)\n", double(error) / double(test_cnt) * 100.0, error);
    printf("Accuracy = %.3f%% (%d out of 10,000)\n",
           100.0 - double(error) / double(test_cnt) * 100.0, test_cnt - error);
    printf("Ex time = %f (ms) \n", time_taken); // NOTE: cudaMemcpy operations also should be added into the time_taken

    // return 0;
}