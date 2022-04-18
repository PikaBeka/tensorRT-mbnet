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
/*How many images to print, Chnage it to test*/
#define TEST_NUMER 4

typedef struct mnist_data
{
    double data[INSIZE][INSIZE];
    unsigned int label;
} mnist_data;
class Layer
{
public:
    int M, N, O; // O: output 6x24x24x, N: #feature 6, M: #params_per_feature 5x5
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

        for (int i = 0; i < M * N; i++)
        {
            if (i < N)
                this->b_h[i] = -1.0f;
            this->w_h[i] = -1.0f;
        }

        cudaMemcpy(this->weight, this->w_h, M * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(this->bias, this->b_h, N * sizeof(float), cudaMemcpyHostToDevice);
    }

    void cpyDeviceToHost(float *o_h, float *o_d)
    {
        cudaMemcpy(o_h, o_d, O * sizeof(float), cudaMemcpyDeviceToHost);
    }

    void print(float *out)
    {
        for (int i = 0; i < 6; i++)
        {
            printf("z = %d\n", i);
            for (int j = 0; j < 24; j++)
            {
                for (int k = 0; k < 24; k++)
                {
                    printf("%f ", out[k + j * 24 + i * 24 * 24]);
                }
                printf("\n");
            }
            printf("\n");
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
    // printf("image magic number = %d (should be 2051)\n", image_data[0]);
    // printf("label magic number = %d (should be 2049)\n", label_data[0]);
    // printf("image total number = %d (should be 10000)\n", image_data[1]);
    // printf("label total number = %d (should be 10000)\n", label_data[1]);
    // printf("rows = %d, cols = %d (both should be 28)\n", image_data[2], image_data[3]);

    /*Initializes the buffers with the count of images*/
    int cnt = 0;
    char labels[label_data[1]];
    char image[image_data[1] * image_data[2] * image_data[3]];

    /*Read all labels and iamges*/
    fread(labels, 1, label_data[1], label_file);
    fread(image, 1, image_data[1] * image_data[2] * image_data[3], image_file);

    int start = 0, limit = 28 * 28;

    /*iterate all images and labels*/
    for (int i = 0; i < image_data[1]; i++)
    {
        mnist_data *d = &(*data_set)[i]; // pointer to the memory of the first element
        d->label = labels[i];            // store label
        int k = 0;
        for (int j = start; j < limit; j++) // stores the double value of images
        {
            d->data[k / 28][k % 28] = abs(image[j] / 255.0);
            k++;
        }
        /*Require to control the image iteration*/
        start = limit;
        limit += 28 * 28;
        cnt++;
    }
    *count = cnt;

    /*Close files*/
    fclose(label_file);
    fclose(image_file);

    return 0;
}

__global__ void kernel_conv_filter(float *input, float *pre_output, float *weight)
{
    int row = threadIdx.y;
    int col = threadIdx.x;

    float temp = 0;

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
            maxErr = std::max(maxErr, abs(arr[i] - val));
        }
    }
    printf("maxEror = %f (asserted with %f)\n", maxErr, val);
}

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
            tmp += input[input_idx] * weight[weight_idx];
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

    output[idx] = 1.0 / (1.0 + exp(-pre_output[idx]));
}

void forward_pass(float *data, Layer *layer)
{
    float *data_d;
    cudaMalloc((void **)&data_d, 28 * 28 * sizeof(28));
    cudaMemcpy(data_d, data, 28 * 28 * sizeof(float), cudaMemcpyHostToDevice);

    /*28x28 * 6@5x5 = 6@24x24*/
    dim3 conv_blocks(1, 1, 6);
    dim3 conv_threads(28, 28, 1);
    kernel_conv_filter<<<conv_blocks, conv_threads>>>(data_d, layer->pre_output, layer->weight);
    layer->cpyDeviceToHost(layer->pre_output_h, layer->pre_output);
    // maxError(layer->pre_output_h, 25.0f, 24 * 24 * 6);

    // /*6@24x24 * 6@1x1 = 6@24x24*/
    // dim3 blocks(1, 1, 6);
    // dim3 threads(24, 24, 1);
    // kernel_conv_bias<<<blocks, threads>>>(layer->pre_output, layer->bias);
    // layer->cpyDeviceToHost(layer->pre_output_h, layer->pre_output);
    // maxError(layer->pre_output_h, 24.0f, 24 * 24 * 6);

    // /*6@24x24 = 6@24x24*/
    // kernel_conv_sigmoid<<<blocks, threads>>>(layer->pre_output, layer->output);
    // layer->cpyDeviceToHost(layer->output_h, layer->output);
    // maxError(layer->output_h, 1.0f, 24 * 24 * 6);

    // Layer pool_layer(4 * 4, 1, 6 * 6 * 6);

    // /*6@24x24 * 1@4x4 = 6@6x6*/
    // dim3 pool_blocks(1, 1, 6);
    // dim3 pool_threads(6, 6, 1);
    // kernel_ss1_filter<<<pool_blocks, pool_threads>>>(layer->output, pool_layer.pre_output, pool_layer.weight);
    // pool_layer.cpyDeviceToHost(pool_layer.pre_output_h, pool_layer.pre_output);
    // maxError(pool_layer.pre_output_h, -16.0f, 6 * 6 * 6);

    // kernel_ss1_bias<<<pool_blocks, pool_threads>>>(pool_layer.pre_output, pool_layer.bias);
    // pool_layer.cpyDeviceToHost(pool_layer.pre_output_h, pool_layer.pre_output);
    // maxError(pool_layer.pre_output_h, -17.0f, 6 * 6 * 6);

    // kernel_ss1_sigmoid<<<pool_blocks, pool_threads>>>(pool_layer.pre_output, pool_layer.output);
    // pool_layer.cpyDeviceToHost(pool_layer.output_h, pool_layer.output);
    // maxError(pool_layer.output_h, 0.0f, 6 * 6 * 6);

    cudaFree(data_d);
}

int main()
{

    int ret;
    int i;
    mnist_data *test_set = (mnist_data *)malloc(SIZE * sizeof(mnist_data));
    static unsigned int test_cnt = 0; // load data
    if (ret = mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", &test_set,
                         &test_cnt) != 0)
        printf("An error occurred: %d \n", ret);
    else
        printf("test_cnt = %d \n", test_cnt);

    float data[28 * 28];
    for (int i = 0; i < 28 * 28; i++)
        data[i] = -1.0f;

    for (int i = 0; i < 10000; i++)
    {
        Layer layer(5 * 5, 6, 6 * 24 * 24);
        forward_pass(data, &layer);
    }

    return 0;
}