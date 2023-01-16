#include "slenet_params.h"

#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <map>
#include <assert.h>
#include <vector>
#include <numeric>
#include <string>
#include <functional>
#include <memory>

#include "NvInfer.h"

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

using namespace nvinfer1;
std::shared_ptr<nvinfer1::ICudaEngine> mEngine;

struct InferDeleter
{
    template <typename T>
    void operator()(T *obj) const
    {
        delete obj;
    }
};

// Defining the mnist_data struct
typedef struct mnist_data
{
    double data[INSIZE][INSIZE];
    unsigned int label;
} mnist_data;

class Logger : public ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

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

std::map<std::string, Weights> loadWeights()
{
    std::map<std::string, nvinfer1::Weights> weightMap;

    Weights wt_c1_bias{DataType::kFLOAT, nullptr, 0};
    wt_c1_bias.values = (const void *)c1_bias;
    wt_c1_bias.count = 6;
    weightMap["c1_bias"] = wt_c1_bias;

    Weights wt_c1_weight{DataType::kFLOAT, nullptr, 0};
    wt_c1_weight.values = (const void *)c1_weight;
    wt_c1_weight.count = 6 * 25;
    weightMap["c1_weight"] = wt_c1_weight;

    Weights wt_s2_bias{DataType::kFLOAT, nullptr, 0};
    wt_s2_bias.values = (const void *)s2_bias;
    wt_s2_bias.count = 6;
    weightMap["s2_bias"] = wt_s2_bias;

    Weights wt_s2_weight{DataType::kFLOAT, nullptr, 0};
    wt_s2_weight.values = (const void *)s2_weight;
    wt_s2_weight.count = 36 * 16;
    weightMap["s2_weight"] = wt_s2_weight;

    Weights wt_f3_bias{DataType::kFLOAT, nullptr, 0};
    wt_f3_bias.values = (const void *)f3_bias;
    wt_f3_bias.count = 10;
    weightMap["f3_bias"] = wt_f3_bias;

    Weights wt_f3_weight{DataType::kFLOAT, nullptr, 0};
    wt_f3_weight.values = (const void *)f3_weight;
    wt_f3_weight.count = 10 * 216;
    weightMap["f3_weight"] = wt_f3_weight;

    return weightMap;
}

float s_LeNet(mnist_data *test_set)
{
    unsigned int error = 0;
    unsigned int max = 0;
    float res[10];
    float time_taken = 0.0f;
    int test_cnt = 10;

    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights();

    IBuilder *builder = createInferBuilder(logger);
    const auto explicitBatchFlag = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(explicitBatchFlag);
    auto config = builder->createBuilderConfig();
    if (!config)
    {
        std::cout << "No configuration" << std::endl;
        return false;
    }

    //==================================================================================================================================================

    auto data = network->addInput("data", DataType::kFLOAT, Dims4{1, 1, INSIZE, INSIZE});
    assert(data);

    auto conv1 = network->addConvolutionNd(
        *data, 6, Dims{2, {5, 5}}, weightMap["c1_weight"], weightMap["c1_bias"]);
    assert(conv1);
    conv1->setStrideNd(DimsHW{1, 1});

    // auto *sigmoid1 = network->addActivation(*conv1->getOutput(0), ActivationType::kSIGMOID);
    // assert(sigmoid1);

    // auto *conv2 = network->addConvolutionNd(
    //     *sigmoid1->getOutput(0), 6, Dims{2, {4, 4}}, weightMap["s2_weight"], weightMap["s2_bias"]);
    // assert(conv2);
    // conv2->setStrideNd(DimsHW{4, 4});

    // auto *sigmoid2 = network->addActivation(*conv2->getOutput(0), ActivationType::kSIGMOID);
    // assert(sigmoid2);

    // // Utility for use MatMul as FC
    // auto addMatMulasFCLayer = [&network](ITensor *input, int32_t const outputs, Weights &filterWeights, Weights &biasWeights) -> ILayer *
    // {
    //     Dims inputDims = input->getDimensions();
    //     int32_t const m = inputDims.d[0];
    //     int32_t const k = 216; // std::accumulate(inputDims.d + 1, inputDims.d + inputDims.nbDims, 1, std::multiplies<int32_t>());
    //     int32_t const n = 10;  // static_cast<int32_t>(filterWeights.count / static_cast<int64_t>(k));
    //     // assert(static_cast<int64_t>(n) * static_cast<int64_t>(k) == filterWeights.count);
    //     // std::cout << m << " " << k << " " << n << std::endl;
    //     // std::cout << filterWeights.count << " " << biasWeights.count << " " << n << std::endl;
    //     // assert(static_cast<int64_t>(n) == biasWeights.count);
    //     // assert(n == outputs);

    //     IShuffleLayer *inputReshape = network->addShuffle(*input);
    //     assert(inputReshape);
    //     inputReshape->setReshapeDimensions(Dims{2, {m, k}});

    //     IConstantLayer *filterConst = network->addConstant(Dims{2, {n, k}}, filterWeights);
    //     assert(filterConst);
    //     IMatrixMultiplyLayer *mm = network->addMatrixMultiply(*inputReshape->getOutput(0), MatrixOperation::kNONE,
    //                                                           *filterConst->getOutput(0), MatrixOperation::kTRANSPOSE);
    //     assert(mm);

    //     IConstantLayer *biasConst = network->addConstant(Dims{2, {1, n}}, biasWeights);
    //     assert(biasConst);
    //     IElementWiseLayer *biasAdd = network->addElementWise(*mm->getOutput(0), *biasConst->getOutput(0), ElementWiseOperation::kSUM);
    //     assert(biasAdd);

    //     IShuffleLayer *outputReshape = network->addShuffle(*biasAdd->getOutput(0));
    //     assert(outputReshape);
    //     outputReshape->setReshapeDimensions(Dims{4, {m, n, 1, 1}});

    //     return outputReshape;
    // };

    // // Add second fully connected layer with 10 outputs.
    // ILayer *ip1 = addMatMulasFCLayer(sigmoid2->getOutput(0), 10, weightMap["f3_weight"], weightMap["f3_bias"]);
    // assert(ip1);

    // auto *sigmoid3 = network->addActivation(*ip1->getOutput(0), ActivationType::kSIGMOID);
    // assert(sigmoid3);

    // sigmoid3->getOutput(0)->setName("prob");
    network->markOutput(*conv1->getOutput(0));
    if (!network)
    {
        std::cout << "Network failed" << std::endl;
        return false;
    }

    config->setFlag(BuilderFlag::kFP16);
    IHostMemory *plan = builder->buildSerializedNetwork(*network, *config);
    if (!plan)
    {
        std::cout << "Plan failed" << std::endl;
        return false;
    }

    IRuntime *runtime = createInferRuntime(logger);
    if (!runtime)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()), InferDeleter());

    if (!mEngine)
    {
        return false;
    }
    //==================================================================================================================================================

    float(*d_input)[INSIZE];
    float(*d_output)[10];

    for (int i = 0; i < test_cnt; i++)
    {
        cudaMalloc(&d_input, sizeof(float) * INSIZE * INSIZE);
        cudaMalloc(&d_output, sizeof(float) * 10);
        // Copying a double data to a float data
        float input[INSIZE][INSIZE];

        for (int k = 0; k < INSIZE; k++)
        {
            for (int j = 0; j < INSIZE; j++)
                input[k][j] = test_set[i].data[k][j];
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        cudaMemcpy(d_input, input, sizeof(float) * INSIZE * INSIZE, cudaMemcpyHostToDevice);

        auto context = mEngine->createExecutionContext();
        if (!context)
        {
            return false;
        }

        void *bindings[2];
        bindings[0] = d_input;
        bindings[1] = d_output;

        bool status = context->executeV2(bindings);

        cudaMemcpy(res, d_output, sizeof(float) * NUM_CLASSES, cudaMemcpyDeviceToHost);

        for (int j = 0; j < NUM_CLASSES; j++)
        {
            if (res[max] < res[j])
                max = j;
        }

        if (max != test_set[i].label)
            error++;
        cudaFree(d_input);
        cudaFree(d_output);
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