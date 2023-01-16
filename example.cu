#include "slenet_params.h"
#include <stdio.h>
#include <stdlib.h>

#define CHANNEL 6
#define CONV_OUTPUT_SIZE 24
#define FILTER_SIZE 5
#define INSIZE 28
#define NUM_CLASSES 10
#define SS_CHANNELS 1
#define SS_OUTPUT_SIZE 6
#define SS_SIZE 4
#define STRIDE 4

#define N1 CHANNEL *CONV_OUTPUT_SIZE *CONV_OUTPUT_SIZE
#define K1 128

#define N2 CHANNEL *SS_OUTPUT_SIZE *SS_OUTPUT_SIZE
#define K2 8

#define N3 NUM_CLASSES *SS_OUTPUT_SIZE *SS_OUTPUT_SIZE *SS_OUTPUT_SIZE
#define K3 16

// Defining the mnist_data struct
typedef struct mnist_data
{
    double data[INSIZE][INSIZE];
    unsigned int label;
} mnist_data;

// Defining the Layer class
class Layer
{
public:
    int M, N, O;
    float *pre_output, *output;
    float *weight, *bias;

    Layer(int M, int N, int O);
    ~Layer();
};

Layer::Layer(int M, int N, int O)
{
    this->M = M;
    this->N = N;
    this->O = O;

    float *temp_weight, *temp_bias;

    // Initializing weights and biases
    temp_weight = (float *)malloc(sizeof(float) * M * N);
    temp_bias = (float *)malloc(sizeof(float) * N);

    for (int i = 0; i < M * N; i++)
        temp_weight[i] = 1.0f;

    for (int i = 0; i < N; i++)
        temp_bias[i] = 1.0f;

    // Allocating space for CUDA variables
    cudaMalloc(&pre_output, sizeof(float) * O);
    cudaMalloc(&output, sizeof(float) * O);
    cudaMalloc(&weight, sizeof(float) * M * N);
    cudaMalloc(&bias, sizeof(float) * N);

    // Copying weights and biases to CUDA variables
    cudaMemcpy(weight, temp_weight, sizeof(float) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(bias, temp_bias, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Freeing temporary weights and biases
    free(temp_weight);
    free(temp_bias);
}

Layer::~Layer()
{
    // Freeing all CUDA varibles of a layer
    cudaFree(pre_output);
    cudaFree(output);
    cudaFree(weight);
    cudaFree(bias);
}

// Initializing a convolutional layer
Layer conv_layer(FILTER_SIZE *FILTER_SIZE, CHANNEL, CHANNEL *CONV_OUTPUT_SIZE *CONV_OUTPUT_SIZE);
Layer ss_layer(SS_SIZE *SS_SIZE, SS_CHANNELS, CHANNEL *SS_OUTPUT_SIZE *SS_OUTPUT_SIZE);
Layer fc_layer(CHANNEL *SS_OUTPUT_SIZE *SS_OUTPUT_SIZE, NUM_CLASSES, NUM_CLASSES);

double time_taken = 0.0;

static unsigned int mnist_bin_to_int(unsigned char *tmp)
{
    // Converting the binary char value to the integer value
    unsigned int result = 0;
    short charSize = 4;
    short multiplier = 256;

    for (int i = 0; i < charSize; i++)
    {
        unsigned int temp = tmp[i];

        for (int j = 0; j < charSize - i - 1; j++)
            temp *= multiplier;

        result += temp;
    }

    // Returning the integer value
    return result;
}

static int mnist_load(const char *image_filename, const char *label_filename, mnist_data **data_set, unsigned int *count)
{
    // Initializing necessary variables
    FILE *images;
    FILE *labels;

    unsigned char *imagesBuffer;
    unsigned char *labelsBuffer;

    long imagesFileSize;
    long labelsFileSize;

    short unsignedIntSize = 4;
    short unsignedByteSize = 1;

    unsigned int imageMagicNumber;
    unsigned int labelMagicNumber;
    unsigned int imageTotalNumber;
    unsigned int labelTotalNumber;
    unsigned int rows, cols;

    // Opening image and label files of the test
    images = fopen("data/t10k-images.idx3-ubyte", "rb");

    if (images == NULL)
    {
        printf("Error! Images file cannot be read!\n");
        return 1;
    }

    labels = fopen("data/t10k-labels.idx1-ubyte", "rb");

    if (images == NULL)
    {
        printf("Error! Labels file cannot be read!\n");
        return 1;
    }

    fseek(images, 0, SEEK_END);
    fseek(labels, 0, SEEK_END);

    imagesFileSize = ftell(images);
    labelsFileSize = ftell(labels);

    fseek(images, 0, SEEK_SET);
    fseek(labels, 0, SEEK_SET);

    imagesBuffer = (unsigned char *)malloc(sizeof(unsigned char) * imagesFileSize);

    if (imagesBuffer == NULL)
    {
        printf("Error! Memory error has occured!\n");
        return 2;
    }

    labelsBuffer = (unsigned char *)malloc(sizeof(unsigned char) * labelsFileSize);

    if (labelsBuffer == NULL)
    {
        printf("Error! Memory error has occured!\n");
        return 2;
    }

    // Reading a magic number
    fread(imagesBuffer, unsignedIntSize, 1, images);
    fread(labelsBuffer, unsignedIntSize, 1, labels);
    imageMagicNumber = mnist_bin_to_int(imagesBuffer);
    labelMagicNumber = mnist_bin_to_int(labelsBuffer);
    printf("Image magic number: %d\n", imageMagicNumber);
    printf("Label magic number: %d\n", labelMagicNumber);

    // Reading a number of images and label files
    fread(imagesBuffer, unsignedIntSize, 1, images);
    fread(labelsBuffer, unsignedIntSize, 1, labels);
    imageTotalNumber = mnist_bin_to_int(imagesBuffer);
    labelTotalNumber = mnist_bin_to_int(labelsBuffer);
    printf("Number of images: %d\n", imageTotalNumber);
    printf("Number of labels: %d\n", labelTotalNumber);

    // Check whether the number of images and label files is the same
    if (imageTotalNumber != labelTotalNumber)
    {
        printf("Error! The number of images and the number of labels are different!\n");
        return 3;
    }
    else
    {
        printf("The number of images and the number of labels are the same!\n");
    }

    // Check the number of rows and columns
    fread(imagesBuffer, unsignedIntSize, 1, images);
    rows = mnist_bin_to_int(imagesBuffer);
    fread(imagesBuffer, unsignedIntSize, 1, images);
    cols = mnist_bin_to_int(imagesBuffer);
    printf("Rows: %d\n", rows);
    printf("Cols: %d\n", cols);

    *data_set = (mnist_data *)malloc(sizeof(mnist_data) * imageTotalNumber);

    // Load image data as double type
    for (int i = 0; i < imageTotalNumber; i++)
    {
        fread(imagesBuffer, rows * cols, 1, images);
        fread(labelsBuffer, unsignedByteSize, 1, labels);

        for (int j = 0; j < INSIZE; j++)
        {
            for (int k = 0; k < INSIZE; k++)
            {
                (*data_set)[i].data[j][k] = imagesBuffer[j * INSIZE + k] / 255.0;
            }
        }

        (*data_set)[i].label = labelsBuffer[0];
    }

    // Closing opened files
    fclose(images);
    fclose(labels);
    free(imagesBuffer);
    free(labelsBuffer);
    *count = imageTotalNumber;
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
                if ((*data_set)[i].data[j][k] > 0)
                {
                    printf("1");
                }
                else
                {
                    printf("0");
                }
            }
            printf("\n");
        }

        printf("Label: %d\n", (*data_set)[i].label);
    }
}

__global__ void kernel_conv_filter(float input[INSIZE][INSIZE], float pre_output[CHANNEL][CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE], float weight[CHANNEL][FILTER_SIZE][FILTER_SIZE])
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int channel = idx % CHANNEL;
    int output_x = (idx / CHANNEL) % CONV_OUTPUT_SIZE;
    int output_y = (idx / CHANNEL / CONV_OUTPUT_SIZE) % CONV_OUTPUT_SIZE;
    float tempC = 0.0f;

    for (int i = 0; i < FILTER_SIZE; i++)
    {
        for (int j = 0; j < FILTER_SIZE; j++)
        {
            tempC += weight[channel][i][j] * input[i + output_x][j + output_y];
            // printf("%f \n", weight[channel][i][j]);
        }
    }

    pre_output[channel][output_x][output_y] = tempC;
}

__global__ void kernel_conv_bias(float pre_output[CHANNEL][CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE], float bias[CHANNEL])
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int channel = idx % CHANNEL;
    int output_x = (idx / CHANNEL) % CONV_OUTPUT_SIZE;
    int output_y = (idx / CHANNEL / CONV_OUTPUT_SIZE) % CONV_OUTPUT_SIZE;
    pre_output[channel][output_x][output_y] += bias[channel];
}

__global__ void kernel_conv_sigmoid(float preact[CHANNEL][CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE], float output[CHANNEL][CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE])
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int channel = idx % CHANNEL;
    int output_x = (idx / CHANNEL) % CONV_OUTPUT_SIZE;
    int output_y = (idx / CHANNEL / CONV_OUTPUT_SIZE) % CONV_OUTPUT_SIZE;
    output[channel][output_x][output_y] = 1 / (1 + exp(-preact[channel][output_x][output_y]));
}

__global__ void kernel_ss1_filter(float input[CHANNEL][CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE], float pre_output[CHANNEL][SS_OUTPUT_SIZE][SS_OUTPUT_SIZE], float weight[SS_CHANNELS][SS_SIZE][SS_SIZE])
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int channel = idx % CHANNEL;
    int output_x = (idx / CHANNEL) % SS_OUTPUT_SIZE;
    int output_y = (idx / CHANNEL / SS_OUTPUT_SIZE) % SS_OUTPUT_SIZE;
    float tempC = 0.0f;

    for (int i = 0; i < SS_SIZE; i++)
    {
        for (int j = 0; j < SS_SIZE; j++)
        {
            tempC += weight[0][i][j] * input[channel][i + output_x * STRIDE][j + output_y * STRIDE];
        }
    }

    pre_output[channel][output_x][output_y] = tempC;
}

__global__ void kernel_ss1_bias(float pre_output[CHANNEL][SS_OUTPUT_SIZE][SS_OUTPUT_SIZE], float bias[SS_CHANNELS])
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int channel = idx % CHANNEL;
    int output_x = (idx / CHANNEL) % SS_OUTPUT_SIZE;
    int output_y = (idx / CHANNEL / SS_OUTPUT_SIZE) % SS_OUTPUT_SIZE;
    pre_output[channel][output_x][output_y] += bias[0];
}

__global__ void kernel_ss1_sigmoid(float pre_output[CHANNEL][SS_OUTPUT_SIZE][SS_OUTPUT_SIZE], float output[CHANNEL][SS_OUTPUT_SIZE][SS_OUTPUT_SIZE])
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int channel = idx % CHANNEL;
    int output_x = (idx / CHANNEL) % SS_OUTPUT_SIZE;
    int output_y = (idx / CHANNEL / SS_OUTPUT_SIZE) % SS_OUTPUT_SIZE;
    output[channel][output_x][output_y] = 1 / (1 + exp(-pre_output[channel][output_x][output_y]));
}

__global__ void kernel_fc1(float input[6][6][6], float pre_output[10], float weight[10][6][6][6])
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int channel = idx % NUM_CLASSES;
    float tempC = 0.0f;

    for (int i = 0; i < SS_OUTPUT_SIZE; i++)
    {
        for (int j = 0; j < SS_OUTPUT_SIZE; j++)
        {
            for (int k = 0; k < SS_OUTPUT_SIZE; k++)
            {
                tempC += weight[channel][i][j][k] * input[i][j][k];
            }
        }
    }

    pre_output[channel] = tempC;
}

__global__ void kernel_fc1_bias(float pre_output[10], float bias[10])
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int channel = idx % NUM_CLASSES;
    pre_output[channel] += bias[channel];
}

__global__ void kernel_fc1_sigmoid(float pre_output[10], float output[10])
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int channel = idx % NUM_CLASSES;
    output[channel] = 1 / (1 + exp(-pre_output[channel]));
}

void verifyConv(float *A, float val)
{
    float maxError = 0.0f;

    for (int i = 0; i < CHANNEL * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE; i++)
        printf("%f \n", A[i]);

    printf("maxError = %f\n", maxError);
}

void verifySS(float *A, float val)
{
    float maxError = 0.0f;

    for (int i = 0; i < CHANNEL * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE; i++)
        printf("%f \n", A[i]);

    // printf("maxError = %f\n", maxError);
}

void verifyFC(float *A, float val)
{
    float maxError = 0.0f;

    for (int i = 0; i < NUM_CLASSES; i++)
        maxError = max(abs(A[i] - val), maxError);

    // printf("maxError = %f\n", maxError);
}

// Performing a forward pass using a single image
static double forward_pass(double data[INSIZE][INSIZE], bool verify)
{
    // Copying a double data to a float data
    float input[INSIZE][INSIZE];
    float *verification;

    for (int i = 0; i < INSIZE; i++)
    {
        for (int j = 0; j < INSIZE; j++)
            input[i][j] = data[i][j];
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    float(*d_input)[INSIZE];
    cudaMalloc(&d_input, sizeof(float) * INSIZE * INSIZE);
    cudaMemcpy(d_input, input, sizeof(float) * INSIZE * INSIZE, cudaMemcpyHostToDevice);

    // Performing Convolutional filtering
    kernel_conv_filter<<<N1 / K1, K1>>>(d_input, (float(*)[CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE])conv_layer.pre_output, (float(*)[FILTER_SIZE][FILTER_SIZE])conv_layer.weight);

    // if (verify)
    // {
    //     printf("Verifying Convolutional filtering operation: ");
    //     verification = (float *)malloc(sizeof(float) * CHANNEL * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE);
    //     cudaMemcpy(verification, conv_layer.pre_output, sizeof(float) * CHANNEL * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE, cudaMemcpyDeviceToHost);
    //     verifyConv(verification, 25.0f);
    //     free(verification);
    // }

    // Performing Convolutional bias addition
    kernel_conv_bias<<<N1 / K1, K1>>>((float(*)[CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE])conv_layer.pre_output, conv_layer.bias);

    // Verifying Convolutional bias operation
    // if (verify)
    // {
    //     printf("Verifying Convolutional bias operation: ");
    //     verification = (float *)malloc(sizeof(float) * CHANNEL * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE);
    //     cudaMemcpy(verification, conv_layer.pre_output, sizeof(float) * CHANNEL * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE, cudaMemcpyDeviceToHost);
    //     verifyConv(verification, 26.0f);
    //     free(verification);
    // }

    // Performing Convolutional sigmoid operation
    kernel_conv_sigmoid<<<N1 / K1, K1>>>((float(*)[CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE])conv_layer.pre_output, (float(*)[CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE])conv_layer.output);

    // Verifying Convolutional sigmoid operation
    // if (verify)
    // {
    //     printf("Verifying Convolutional sigmoid operation: ");
    //     verification = (float *)malloc(sizeof(float) * CHANNEL * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE);
    //     cudaMemcpy(verification, conv_layer.output, sizeof(float) * CHANNEL * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE, cudaMemcpyDeviceToHost);
    //     verifyConv(verification, 1.0f);
    //     free(verification);
    // }

    // Performing Subsampling filtering
    kernel_ss1_filter<<<N2 / K2, K2>>>((float(*)[CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE])conv_layer.output, (float(*)[SS_OUTPUT_SIZE][SS_OUTPUT_SIZE])ss_layer.pre_output, (float(*)[SS_SIZE][SS_SIZE])ss_layer.weight);

    // Verifying Subsampling filtering operation
    // if (verify)
    // {
    //     printf("Verifying Subsampling filtering operation: ");
    //     verification = (float *)malloc(sizeof(float) * CHANNEL * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE);
    //     cudaMemcpy(verification, ss_layer.pre_output, sizeof(float) * CHANNEL * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE, cudaMemcpyDeviceToHost);
    //     verifySS(verification, 16.0f);
    //     free(verification);
    // }

    // Performing Subsampling bias addition
    kernel_ss1_bias<<<N2 / K2, K2>>>((float(*)[SS_OUTPUT_SIZE][SS_OUTPUT_SIZE])ss_layer.pre_output, ss_layer.bias);

    // Verifying Subsampling bias operation
    // if (verify) {
    // 	printf("Verifying Subsampling bias operation: ");
    // 	verification = (float*)malloc(sizeof(float) * CHANNEL * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE);
    // 	cudaMemcpy(verification, ss_layer.pre_output, sizeof(float) * CHANNEL * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE, cudaMemcpyDeviceToHost);
    // 	verifySS(verification, 17.0f);
    // 	free(verification);
    // }

    // // Performing Subsampling sigmoid operation
    kernel_ss1_sigmoid<<<N2 / K2, K2>>>((float(*)[SS_OUTPUT_SIZE][SS_OUTPUT_SIZE])ss_layer.pre_output, (float(*)[SS_OUTPUT_SIZE][SS_OUTPUT_SIZE])ss_layer.output);

    // Verifying Subsampling sigmoid operation
    // if (verify) {
    // 	printf("Verifying Subsampling sigmoid operation: ");
    // 	verification = (float*)malloc(sizeof(float) * CHANNEL * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE);
    // 	cudaMemcpy(verification, ss_layer.output, sizeof(float) * CHANNEL * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE, cudaMemcpyDeviceToHost);
    // 	verifySS(verification, 1.0f);
    // 	free(verification);
    // }

    // Performing Fully-Connected Computation
    kernel_fc1<<<N3 / K3, K3>>>((float(*)[SS_OUTPUT_SIZE][SS_OUTPUT_SIZE])ss_layer.output, (float(*))fc_layer.pre_output, (float(*)[SS_OUTPUT_SIZE][SS_OUTPUT_SIZE][SS_OUTPUT_SIZE])fc_layer.weight);

    // Verifying Fully-Connected Computation
    // if (verify) {
    // 	printf("Verifying Fully-Connected Computation: ");
    // 	verification = (float*)malloc(sizeof(float) * NUM_CLASSES);
    // 	cudaMemcpy(verification, fc_layer.pre_output, sizeof(float) * NUM_CLASSES, cudaMemcpyDeviceToHost);
    // 	verifyFC(verification, 216.0f);
    // 	free(verification);
    // }

    // Performing Fully-Connected bias operation
    kernel_fc1_bias<<<1, K3>>>((float(*))fc_layer.pre_output, fc_layer.bias);

    // Verifying Fully-Connected bias operation
    // if (verify) {
    // 	printf("Verifying Fully-Connected bias operation: ");
    // 	verification = (float*)malloc(sizeof(float) * NUM_CLASSES);
    // 	cudaMemcpy(verification, fc_layer.pre_output, sizeof(float) * NUM_CLASSES, cudaMemcpyDeviceToHost);
    // 	verifyFC(verification, 217.0f);
    // 	free(verification);
    // }

    // Performing Fully-Connected sigmoid operation
    kernel_fc1_sigmoid<<<1, K3>>>((float(*))fc_layer.pre_output, (float(*))fc_layer.output);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Verifying Fully-Connected sigmoid operation
    // if (verify)
    // {
    //     printf("Verifying Fully-Connected sigmoid operation: ");
    //     verification = (float *)malloc(sizeof(float) * NUM_CLASSES);
    //     cudaMemcpy(verification, fc_layer.output, sizeof(float) * NUM_CLASSES, cudaMemcpyDeviceToHost);
    //     verifyFC(verification, 1.0f);
    //     free(verification);
    // }

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    return elapsedTime;
}

void copy_trained_parameters()
{
    cudaMemcpy(conv_layer.weight, c1_weight, sizeof(float) * CHANNEL * FILTER_SIZE * FILTER_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(conv_layer.bias, c1_bias, sizeof(float) * CHANNEL, cudaMemcpyHostToDevice);
    cudaMemcpy(ss_layer.weight, s2_weight, sizeof(float) * SS_SIZE * SS_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(ss_layer.bias, s2_bias, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(fc_layer.weight, f3_weight, sizeof(float) * NUM_CLASSES * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(fc_layer.bias, f3_bias, sizeof(float) * NUM_CLASSES, cudaMemcpyHostToDevice);
}

float s_LeNet(mnist_data *test_set)
{
    unsigned int error = 0;
    unsigned int max = 0;
    float res[10];
    float time_taken = 0.0f;
    int test_cnt = 10;

    forward_pass(test_set[0].data, true);
    for (int i = 0; i < test_cnt; i++)
    {
        time_taken += forward_pass(test_set[i].data, false);
        cudaMemcpy(res, fc_layer.output, sizeof(float) * NUM_CLASSES, cudaMemcpyDeviceToHost);

        for (int j = 0; j < NUM_CLASSES; j++)
        {
            if (res[max] < res[j])
                max = j;
        }

        if (max != test_set[i].label)
            error++;
    }

    printf("Execution time = %f (ms) \n", time_taken);
    printf("Error Rate = %f%% (%d out of 10000)\n", double(error) / double(test_cnt) * 100.0, error);
    printf("Accuracy = %.3f%% (%d out of 10000)\n", 100.0 - double(error) / double(test_cnt) * 100.0, test_cnt - error);
    return time_taken;
}

int main()
{
    int ret;
    mnist_data *test_set;
    static unsigned int test_cnt;

    // Calling the mnist_load() function
    if (ret = mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", &test_set, &test_cnt) != 0)
    {
        printf("An error occured: %d\n", ret);
    }
    else
    {
        printf("test_cnt = %d\n", test_cnt);
    }

    // Verifying the image and label data of the specified number of examples
    // printExamples(&test_set, 1);

    // Verifying the convolutional layer
    // double data[INSIZE][INSIZE];

    // for (i = 0; i < INSIZE; i++) {
    // 	for (int j = 0; j < INSIZE; j++)
    // 		data[i][j] = 1.0f;
    // }

    // forward_pass(data, true);

    copy_trained_parameters();

    // Performing forward pass
    int count = 1;
    float averageTime = 0;

    averageTime += s_LeNet(test_set);
    for (int i = 0; i < count; i++)

        averageTime /= count;
    printf("[GPU] (s-LeNet) Average Elapsed Time = %f ms\n", averageTime);

    free(test_set);
    return 0;
}