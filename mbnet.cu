#include "trt_dependencies/argsParser.h"
#include "trt_dependencies/buffers.h"
#include "trt_dependencies/common.h"
#include "trt_dependencies/logger.h"
#include "mbnet.h"

// #include "NvCaffeParser.h"
#include "trt_dependencies/NvInfer.h"
#include <cuda_runtime_api.h>
#include "trt_dependencies/slenet_params.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/param.h>
#include <cmath>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>

#include "cublas_utils.h"

#define CHECK_CUDNN(expression)                                    \
    {                                                              \
        cudnnStatus_t status = (expression);                       \
        if (status != CUDNN_STATUS_SUCCESS)                        \
        {                                                          \
            std::cerr << "Error on line " << __LINE__ << ": "      \
                      << cudnnGetErrorString(status) << std::endl; \
            std::exit(EXIT_FAILURE);                               \
        }                                                          \
    }

using samplesCommon::SampleUniquePtr;

const std::string gSampleName = "TensorRT.sample_mnist_api";

float *input = (float *)malloc(sizeof(float) * input_channels * HW * HW);
float *weight = (float *)malloc(sizeof(float) * RS * RS * K * input_channels);
float *bias = (float *)malloc(sizeof(float) * K);
float *output = (float *)malloc(sizeof(float) * K * PQ * PQ);

int debug = 1;

// double buffManager = 0;
// double process = 0;
// double ItD = 0;
// double OtH = 0;
// double exec = 0;

double im2col_time = 0;

clock_t start, end;

#if TRT
std::string const gCacheFileName = "AlgorithmCache.txt";

//!
//! \brief Writes the default algorithm choices made by TensorRT into a file.
//!
class AlgorithmCacheWriter : public IAlgorithmSelector
{
public:
    //!
    //! \brief Return value in [0, nbChoices] for a valid algorithm.
    //!
    //! \details Lets TRT use its default tactic selection method.
    //! Writes all the possible choices to the selection buffer and returns the length of it.
    //! If BuilderFlag::kSTRICT_TYPES is not set, just returning 0 forces default tactic selection.
    //!
    int32_t selectAlgorithms(const nvinfer1::IAlgorithmContext &context, const nvinfer1::IAlgorithm *const *choices,
                             int32_t nbChoices, int32_t *selection) noexcept override
    {
        // TensorRT always provides more than zero number of algorithms in selectAlgorithms.
        ASSERT(nbChoices > 0);

        std::cout << nbChoices << "\n";

        for (int i = 0; i < nbChoices; ++i)
        {
            std::cout << "Algorithm " << i << ": Implementation = " << choices[i]->getAlgorithmVariant().getImplementation() << std::endl;
        }

        // std::iota(selection, selection + nbChoices, 0);
        return 0;
    }

    //!
    //! \brief called by TensorRT to report choices it made.
    //!
    //! \details Writes the TensorRT algorithm choices into a file.
    //!
    void reportAlgorithms(const nvinfer1::IAlgorithmContext *const *algoContexts,
                          const nvinfer1::IAlgorithm *const *algoChoices, int32_t nbAlgorithms) noexcept override
    {
        std::ofstream algorithmFile(mCacheFileName);
        if (!algorithmFile.good())
        {
            sample::gLogError << "Cannot open algorithm cache file: " << mCacheFileName << " to write." << std::endl;
            abort();
        }

        std::cout << nbAlgorithms << "\n";

        for (int32_t i = 0; i < nbAlgorithms; i++)
        {
            std::cout << algoContexts[i]->getName() << "\n";
            std::cout << algoChoices[i]->getAlgorithmVariant().getImplementation() << "\n";
            std::cout << algoChoices[i]->getAlgorithmVariant().getTactic() << "\n";

            // Write number of inputs and outputs.
            const int32_t nbInputs = algoContexts[i]->getNbInputs();
            algorithmFile << nbInputs << "\n";
            const int32_t nbOutputs = algoContexts[i]->getNbOutputs();
            algorithmFile << nbOutputs << "\n";

            // Write input and output formats.
            for (int32_t j = 0; j < nbInputs + nbOutputs; j++)
            {
                algorithmFile << static_cast<int32_t>(algoChoices[i]->getAlgorithmIOInfoByIndex(j)->getTensorFormat())
                              << "\n";
                algorithmFile << static_cast<int32_t>(algoChoices[i]->getAlgorithmIOInfoByIndex(j)->getDataType())
                              << "\n";
            }
        }
        algorithmFile.close();
    }

    AlgorithmCacheWriter(const std::string &cacheFileName)
        : mCacheFileName(cacheFileName)
    {
    }

private:
    std::string mCacheFileName;
};

//------------------------------------------------------------------------------------------TensorRT--------------------------------------------------------------------------------------------------------
//!
//! \brief The SampleMNISTAPIParams structure groups the additional parameters required by
//!         the SampleMNISTAPI sample.
//!
struct SampleMNISTAPIParams : public samplesCommon::SampleParams
{
    int inputH;                  //!< The input height
    int inputW;                  //!< The input width
    int outputSize;              //!< The output size
    std::string weightsFile;     //!< The filename of the weights file
    std::string mnistMeansProto; //!< The proto file containing means
};

//! \brief  The SampleMNISTAPI class implements the MNIST API sample
//!
//! \details It creates the network for MNIST classification using the API
//!
class SampleMNISTAPI
{
public:
    SampleMNISTAPI(const SampleMNISTAPIParams &params)
        : mParams(params), mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

    //!
    //! \brief Cleans up any state created in the sample class
    //!
    bool teardown();

private:
    SampleMNISTAPIParams mParams; //!< The parameters for the sample.

    int mNumber{0}; //!< The number to classify

    std::map<std::string, nvinfer1::Weights> mWeightMap; //!< The weight name to weight value map

    std::vector<std::unique_ptr<samplesCommon::HostMemory>> weightsMemory; //!< Host weights memory holder

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Uses the API to create the MNIST Network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder> &builder,
                          SampleUniquePtr<nvinfer1::INetworkDefinition> &network, SampleUniquePtr<nvinfer1::IBuilderConfig> &config);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager &buffers, float *input);

    //!
    //! \brief Classifies digits and verify result
    //!
    bool verifyOutput(const samplesCommon::BufferManager &buffers, float *input, float *weight);

    //!
    //! \brief Loads weights from weights file
    //!
    std::map<std::string, nvinfer1::Weights> loadWeights(const std::string &file, float *weight);
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the MNIST network by using the API to create a model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool SampleMNISTAPI::build()
{
    mWeightMap = loadWeights(locateFile(mParams.weightsFile, mParams.dataDirs), weight);

    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatchFlag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatchFlag));
    if (!network)
    {
        return false;
    }

    AlgorithmCacheWriter selector(gCacheFileName);

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    config->setFlag(nvinfer1::BuilderFlag::kDISABLE_TIMING_CACHE);

    config->setAlgorithmSelector(&selector);

    auto constructed = constructNetwork(builder, network, config);
    if (!constructed)
    {
        return false;
    }

    ASSERT(network->getNbInputs() == 1);
    auto inputDims = network->getInput(0)->getDimensions();
    ASSERT(inputDims.nbDims == 4);

    ASSERT(network->getNbOutputs() == 1);
    auto outputDims = network->getOutput(0)->getDimensions();
    ASSERT(outputDims.nbDims == 4);

    return true;
}

//!
//! \brief Uses the API to create the MNIST Network
//!
//! \param network Pointer to the network that will be populated with the MNIST network
//!
//! \param builder Pointer to the engine builder
//!
bool SampleMNISTAPI::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder> &builder,
                                      SampleUniquePtr<nvinfer1::INetworkDefinition> &network, SampleUniquePtr<nvinfer1::IBuilderConfig> &config)
{
    // Create input tensor of shape { 1, 1, 28, 28 }
    ITensor *data = network->addInput(
        mParams.inputTensorNames[0].c_str(), DataType::kFLOAT, Dims4{1, input_channels, HW, HW});
    ASSERT(data);

    // Add convolution layer with 20 outputs and a 5x5 filter.
    IConvolutionLayer *conv1 = network->addConvolutionNd(
        *data, K, Dims{2, {RS, RS}}, mWeightMap["c1_weight"], mWeightMap["c1_bias"]);
    conv1->setStride(DimsHW{1, 1});
    conv1->setPadding(DimsHW{0, 0});
    ASSERT(conv1);

    // Add softmax layer to determine the probability.
    // ISoftMaxLayer *prob = network->addSoftMax(*sigmoid3->getOutput(0));
    // ASSERT(prob);
    conv1->getOutput(0)->setName(mParams.outputTensorNames[0].c_str());
    network->markOutput(*conv1->getOutput(0));

    // Build engine
    // config->setMaxWorkspaceSize(32_MiB);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllDynamicRanges(network.get(), 64.0f, 64.0f);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

    SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return false;
    }

    SampleUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
    if (!runtime)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleMNISTAPI::infer()
{
    samplesCommon::BufferManager buffers(mEngine);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    ASSERT(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers, input))
    {
        return false;
    }

    buffers.copyInputToDevice();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    buffers.copyOutputToHost();

    if (debug && !verifyOutput(buffers, input, weight))
    {
        printf("Verification failed\n");
        return false;
    }

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool SampleMNISTAPI::processInput(const samplesCommon::BufferManager &buffers, float *input)
{
    srand(time(0));

    // printf("Input processed for trt");

    float *hostDataBuffer = static_cast<float *>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    float temp = 1.0f;

    for (int i = 0; i < input_channels; i++)
    {
        for (int j = 0; j < HW; j++)
        {
            for (int k = 0; k < HW; k++)
            {
                // temp = (float)(rand() % 100);
                temp = input[i * HW * HW + j * HW + k];
                hostDataBuffer[i * HW * HW + j * HW + k] = temp;
            }
        }
    }

    // for (int i = 0; i < input_channels; i++)
    // {
    //     for (int j = 0; j < HW; j++)
    //     {
    //         for (int k = 0; k < HW; k++)
    //         {
    //             printf("%d_%d_%d: %f\n", i, j, k, hostDataBuffer[i * HW * HW + j * HW + k]);
    //         }
    //     }
    // }

    return true;
}

//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
bool SampleMNISTAPI::verifyOutput(const samplesCommon::BufferManager &buffers, float *input, float *weight)
{
    output = static_cast<float *>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    bool answer = true;
    // printf("The configuration is %d_%d_%d\n", input_channels, HW, K);

    // for (int i = 0; i < K; i++)
    // {
    //     for (int j = 0; j < PQ; j++)
    //     {
    //         for (int k = 0; k < PQ; k++)
    //         {
    //             if (output[i * PQ * PQ + j * PQ + k] != 25.0)
    //             {
    //                 printf("%d_%d_%d: %f\n", i, j, k, output[i * PQ * PQ + j * PQ + k]);
    //             }
    //         }
    //     }
    // }

    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < PQ; j++)
        {
            for (int k = 0; k < PQ; k++)
            {
                float tempC = 0.0f;
                for (int l = 0; l < input_channels; l++)
                {
                    for (int m = 0; m < RS; m++)
                    {
                        for (int t = 0; t < RS; t++)
                        {
                            tempC += weight[i * input_channels * RS * RS + l * RS * RS + m * RS + t] * input[l * HW * HW + (j + m) * HW + (k + t)];
                        }
                    }
                }
                if (abs(int(round(output[i * PQ * PQ + j * PQ + k]) - tempC)) > 5)
                {
                    printf("The error is here. The actual result is %f, should be %f on (%d, %d, %d)\n", output[i * PQ * PQ + j * PQ + k], tempC, i, j, k);
                    answer = false;
                }
            }
        }
    }

    return answer;
}

//!
//! \brief Cleans up any state created in the sample class
//!
bool SampleMNISTAPI::teardown()
{
    return true;
}

//!
//! \brief Loads weights from weights file
//!
//! \details TensorRT weight files have a simple space delimited format
//!          [type] [size] <data x size in hex>
//!
std::map<std::string, nvinfer1::Weights> SampleMNISTAPI::loadWeights(const std::string &file, float *weight)
{
    std::map<std::string, nvinfer1::Weights> weightMap;

    for (int i = 0; i < K; i++)
    {
        bias[i] = 0.0f;
    }

    Weights wt_c1_bias{DataType::kFLOAT, nullptr, 0};
    wt_c1_bias.type = DataType::kFLOAT;
    wt_c1_bias.values = bias;
    wt_c1_bias.count = K;
    weightMap["c1_bias"] = wt_c1_bias;

    for (int i = 0; i < K; i++)
    {
        for (int t = 0; t < input_channels; t++)
        {
            for (int j = 0; j < RS; j++)
            {
                for (int k = 0; k < RS; k++)
                {
                    weight[i * (input_channels * RS * RS) + t * (RS * RS) + j * RS + k] = (float)(rand() % 100);
                    // weight[i * (input_channels * RS * RS) + t * (RS * RS) + j * RS + k] = 1.0f;
                }
            }
        }
    }

    Weights wt_c1_weight{DataType::kFLOAT, nullptr, 0};
    wt_c1_weight.type = DataType::kFLOAT;
    wt_c1_weight.values = weight;
    wt_c1_weight.count = K * input_channels * RS * RS;
    weightMap["c1_weight"] = wt_c1_weight;

    return weightMap;
}
//!
//! \brief Initializes members of the params struct using the command line args
//!
SampleMNISTAPIParams initializeSampleParams(const samplesCommon::Args &args)
{
    SampleMNISTAPIParams params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back(".");
        params.dataDirs.push_back(".");
    }
    else //!< Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.inputTensorNames.push_back("data");
    params.outputTensorNames.push_back("prob");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    params.inputH = HW;
    params.inputW = HW;
    params.outputSize = K * PQ * PQ;
    params.weightsFile = "";
    params.mnistMeansProto = "";

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./sample_mnist_api [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
        << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--int8          Run in Int8 mode." << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}
#endif

void fillInputWithValues(float *input)
{
    srand(time(0));

    for (int b = 0; b < BATCH; b++)
    {
        for (int i = 0; i < input_channels; i++)
        {
            for (int j = 0; j < HW; j++)
            {
                for (int k = 0; k < HW; k++)
                {
                    // input[i * HW * HW + j * HW + k] = (float)(rand() % 2) - 1;
                    input[i * HW * HW + j * HW + k] = 1.0f;
                }
            }
        }
    }
}

#if TRT
#else
//--------------------------------------------------------------------------------------Auxilory functions---------------------------------------------------------------------------------------------------
/*Function to fill input and weight matrix with random values*/
void fillWeightWithValues(float *weight)
{
    srand(time(0));

    for (int i = 0; i < K; i++)
    {
        for (int t = 0; t < input_channels; t++)
        {
            for (int j = 0; j < RS; j++)
            {
                for (int k = 0; k < RS; k++)
                {
                    // weight[i * (input_channels * RS * RS) + t * (RS * RS) + j * RS + k] = (float)(rand() % 2) - 1;
                    weight[i * (input_channels * RS * RS) + t * (RS * RS) + j * RS + k] = 1.0f;
                }
            }
        }
    }
}

/*Function to verify the */
void verification(float *input, float *weight, float *output)
{
    for (int b = 0; b < BATCH; b++)
    {
        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < PQ; j++)
            {
                for (int k = 0; k < PQ; k++)
                {
                    float tempC = 0.0f;
                    for (int l = 0; l < input_channels; l++)
                    {
                        for (int m = 0; m < RS; m++)
                        {
                            for (int t = 0; t < RS; t++)
                            {
                                tempC += weight[i * input_channels * RS * RS + l * RS * RS + m * RS + t] * input[b * input_channels * HW * HW + l * HW * HW + (j + m) * HW + (k + t)];
                            }
                        }
                    }
                    if (abs(int(round(output[i * PQ * PQ + j * PQ + k]) - tempC)) > 1)
                    {
                        printf("The error is here. The actual result is %f, we get %f on (%d, %d, %d), the diff is %d\n", tempC, output[i * PQ * PQ + j * PQ + k], i, j, k, abs(int(round(output[i * PQ * PQ + j * PQ + k]) - tempC)));
                        printf("Error configuration (%d, %d, %d)\n", input_channels, HW, K);
                        exit(-1);
                    }
                }
            }
        }
    }
#if ARRAY_NAIVE
    printf("Array Naive convolution finished. It is checked with %d images, and correct with image sizes (%d, %d, %d) and kernel (%d, %d, %d) resulting in (%d, %d, %d)\n", images, input_channels, HW, HW, K, RS, RS, K, PQ, PQ);

#elif ARRAY_TILING
    printf("Array Tiling convolution finished. It is checked with %d images, and correct with image sizes (%d, %d, %d) and kernel (%d, %d, %d) resulting in (%d, %d, %d)\n", images, input_channels, HW, HW, K, RS, RS, K, PQ, PQ);

#elif DIRECT
#if CONV_SHARED
    printf("Shared direct convolution finished. It is checked with %d images, and correct with image sizes (%d, %d, %d) and kernel (%d, %d, %d) resulting in (%d, %d, %d)\n", images, input_channels, HW, HW, K, RS, RS, K, PQ, PQ);
#else
    printf("Global direct convolution finished. It is checked with %d images, and correct with image sizes (%d, %d, %d) and kernel (%d, %d, %d) resulting in (%d, %d, %d)\n", images, input_channels, HW, HW, K, RS, RS, K, PQ, PQ);
#endif

#elif CUDNN
    printf("CUDNN convolution finished. It is checked with %d images, and correct with image sizes (%d, %d, %d) and kernel (%d, %d, %d) resulting in (%d, %d, %d)\n", images, input_channels, HW, HW, K, RS, RS, K, PQ, PQ);
#else
#if GEMM_GLOBAL
    printf("Unroll globall gemm convolution finished. It is checked with %d images, and correct with image sizes (%d, %d, %d) and kernel (%d, %d, %d) resulting in (%d, %d, %d)\n", images, input_channels, HW, HW, K, RS, RS, K, PQ, PQ);
#else
    printf("Unroll cublass convolution finished. It is checked with %d images, and correct with image sizes (%d, %d, %d) and kernel (%d, %d, %d) resulting in (%d, %d, %d)\n", images, input_channels, HW, HW, K, RS, RS, K, PQ, PQ);
#endif
#endif
}
#endif

#if ARRAY_NAIVE
/*-------------------------------------------------Array Naive-------------------------------------------------------------------*/
__global__ void convolution_naive(float input[input_channels][HW][HW], float weight[K][input_channels][RS][RS], float output[K][PQ][PQ])
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int out_ch = idx % K;

    int output_x = (idx / K) % PQ;
    int output_y = (idx / K / PQ) % PQ;

    float tempC = 0.0f;

    for (int k = 0; k < input_channels; k++)
    {
        for (int i = 0; i < RS; i++)
        {
            for (int j = 0; j < RS; j++)
            {
                tempC += weight[out_ch][k][i][j] * input[k][output_x + i][output_y + j];
            }
        }
    }

    output[out_ch][output_x][output_y] = tempC;
}
#endif

#if ARRAY_TILING
/*-------------------------------------------------Array Tiling-------------------------------------------------------------------*/
__global__ void convolution_tiling(float input[input_channels][HW][HW], float weight[K][input_channels][RS][RS], float output[K][PQ][PQ])
{
    int row = threadIdx.y;
    int col = threadIdx.x;
    __shared__ float shm[input_channels][TILE_S][TILE_S];

    int control = (row + LIM * blockIdx.y) * HW + col + LIM * blockIdx.x;

    if (control < HW * HW)
    {
        for (int z = 0; z < input_channels; z++)
        {
            shm[z][row][col] = input[z][row + LIM * blockIdx.y][col + LIM * blockIdx.x];
        }
    }

    __syncthreads();

    float temp = 0.0f;

    if (row < LIM && col < LIM && col + LIM * blockIdx.x < PQ && row + blockIdx.y * LIM < PQ)
    {
        for (int k = 0; k < input_channels; k++)
        {
            for (int i = 0; i < RS; i++)
            {
                for (int j = 0; j < RS; j++)
                {
                    temp += shm[k][row + i][col + j] * weight[blockIdx.z][k][i][j];
                }
            }
        }
        output[blockIdx.z][row + blockIdx.y * LIM][col + LIM * blockIdx.x] = temp;
    }
}
#endif

#if DIRECT
/*-------------------------------------------------Direct convolution-------------------------------------------------------------*/
__global__ void kernel_conv_filter(float input[input_channels][HW][HW],
                                   float pre_output[K][PQ][PQ],
                                   float weight[K][input_channels][RS][RS])
{
#if CONV_SHARED
    int tidx = threadIdx.x;
    int bIdx = blockIdx.x;

    __shared__ float sh_img[input_channels][TILE_S][TILE_S];

    int img_row = tidx / TILE_S;
    int img_col = tidx % TILE_S;

    /* input image copy to shared memory */
    if (tidx < TILE_S * TILE_S)
    {
        for (int img_z = 0; img_z < input_channels; img_z++)
            sh_img[img_z][img_row][img_col] = input[img_z][blockIdx.y * LIM + img_row][blockIdx.x * LIM + img_col];
    }

    __syncthreads();

    int ch = tidx / (LIM * LIM);
    int w_row = (tidx % (LIM * LIM)) / LIM;
    int w_col = (tidx % (LIM * LIM)) % LIM;

    float sum = 0;
    if (w_row < LIM && w_col < LIM && ch < K && blockIdx.y * LIM + w_row < PQ && blockIdx.x * LIM + w_col < PQ)
    {
        for (int k = 0; k < input_channels; k++)
        {
            for (int i = 0; i < RS; i++)
                for (int j = 0; j < RS; j++)
                    sum += sh_img[k][w_row + i][w_col + j] * weight[blockIdx.z][k][i][j];
        }
        pre_output[blockIdx.z][blockIdx.y * LIM + w_row][blockIdx.x * LIM + w_col] = sum;
        // printf("ch=%d, bIdx_r=%d, w_row=%d, bIdx_c=%d, w_col=%d, sum=%f\n", ch, bIdx_r, w_row, bIdx_c, w_col, sum);
    }

#else
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int channel = idx % K;
    int output_x = (idx / K) % PQ;
    int output_y = (idx / K / PQ) % PQ;

    float tempC = 0.0f;
    for (int k = 0; k < input_channels; k++)
    {
        for (int i = 0; i < RS; i++)
        {
            for (int j = 0; j < RS; j++)
            {
                tempC += weight[channel][k][i][j] * input[k][i + output_x][j + output_y];
            }
        }
    }
    if (idx < K * PQ * PQ)
        pre_output[channel][output_x][output_y] = tempC;
#endif
}
#endif

#if UNROLL
/*-------------------------------------------------Unrolling -----------------------------------------------------------------------*/
inline bool is_a_ge_zero_and_a_lt_b(int a, int b)
{
    return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

void im2col_cpu(
    const float *data_im,
    const int channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    float *data_col)
{
    const int output_h = (height + 2 * pad_h -
                          (dilation_h * (kernel_h - 1) + 1)) /
                             stride_h +
                         1;
    const int output_w = (width + 2 * pad_w -
                          (dilation_w * (kernel_w - 1) + 1)) /
                             stride_w +
                         1;
    const int channel_size = height * width;
    for (int channel = channels; channel--; data_im += channel_size)
    {
        for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++)
        {
            for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++)
            {
                int input_row = -pad_h + kernel_row * dilation_h;
                for (int output_rows = output_h; output_rows; output_rows--)
                {
                    if (!is_a_ge_zero_and_a_lt_b(input_row, height))
                    {
                        for (int output_cols = output_w; output_cols; output_cols--)
                        {
                            *(data_col++) = 0;
                        }
                    }
                    else
                    {
                        int input_col = -pad_w + kernel_col * dilation_w;
                        for (int output_col = output_w; output_col; output_col--)
                        {
                            if (is_a_ge_zero_and_a_lt_b(input_col, width))
                            {
                                *(data_col++) = data_im[input_row * width + input_col];
                            }
                            else
                            {
                                *(data_col++) = 0;
                            }
                            input_col += stride_w;
                        }
                    }
                    input_row += stride_h;
                }
            }
        }
    }
}

//*/
// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                          \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n);                                       \
         i += blockDim.x * gridDim.x)

// https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu
__global__ void im2col_gpu_kernel(const int n,
                                  const float *data_im,
                                  const int height,
                                  const int width,
                                  const int ksize,
                                  const int pad,
                                  const int stride,
                                  const int height_col,
                                  const int width_col,
                                  float *data_col)
{

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for (; index < n; index += blockDim.x * gridDim.x)
    {
        int w_out = index % width_col;
        int h_index = index / width_col;
        int h_out = h_index % height_col;
        int channel_in = h_index / height_col;
        int channel_out = channel_in * ksize * ksize;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;
        float *data_col_ptr = data_col;
        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
        const float *data_im_ptr = data_im;
        data_im_ptr += (channel_in * height + h_in) * width + w_in;
        for (int i = 0; i < ksize; ++i)
        {
            for (int j = 0; j < ksize; ++j)
            {
                int h = h_in + i;
                int w = w_in + j;

                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ? data_im_ptr[i * width + j] : 0;
                data_col_ptr += height_col * width_col;
            }
        }
    }
}

void verify_im2col(float *A, float val)
{
    float maxError = 0.0f;

    int cnt = 0;
    for (int i = 0; i < RS * RS * PQ * PQ * input_channels; i++)
    {
        // printf("%f\n", A[i]);
        maxError = max(abs(A[i] - val), maxError);
        if (maxError != 0.0)
            cnt++;
    }
    printf("maxError = %f (cnt = %d),%d)\n", maxError, cnt, RS * RS * PQ * PQ * input_channels);
}

__global__ void ker2row_kernel(float weight_col[K][input_channels * RS * RS], float weight[K][input_channels][RS][RS])
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int channel = idx % K;
    int z = (idx / K) % input_channels;
    int x = (idx / K / input_channels) % RS;
    int y = (idx / K / input_channels / RS) % RS;

    if (idx < K * input_channels * RS * RS)
    {
        weight_col[channel][z * RS * RS + x * RS + y] = weight[channel][z][x][y];
    }
}

void verify_ker2row(float *A, float val)
{
    float maxError = 0.0f;

    int cnt = 0;
    for (int i = 0; i < K * input_channels * RS * RS; i++)
    {
        // printf("%f\n", A[i]);
        maxError = max(abs(A[i] - val), maxError);
        if (maxError != 0)
            cnt++;
    }
    printf("maxError = %f (cnt = %d),%d)\n", maxError, cnt, K * input_channels * RS * RS);
}

__global__ void gemm_shared_kernel(float *A, float *B, float *C, int m, int n, int k)
{
    // allocate shared memory for tiles
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0;

    // Loop over tiles
    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t)
    {
        // Load elements into shared memory
        if (row < m && t * TILE_SIZE + threadIdx.x < k)
        {
            tileA[threadIdx.y][threadIdx.x] = A[row * k + t * TILE_SIZE + threadIdx.x];
            printf("%f\n", A[row * k + t * TILE_SIZE + threadIdx.x]);
        }
        else
            tileA[threadIdx.y][threadIdx.x] = 0;

        if (col < n && t * TILE_SIZE + threadIdx.y < k)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * n + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        // Multiply tiles
        for (int k = 0; k < TILE_SIZE; ++k)
            value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();
    }

    // Store result
    if (row < m && col < n)
        C[row * n + col] = value;
}

#if GEMM_GLOBAL
__global__ void
gemm_global_kernel(float matB[K][input_channels * RS * RS], float matA[input_channels * RS * RS][PQ * PQ], float matC[K][PQ * PQ])
{

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int x = idx % K;
    int y = (idx / K) % (PQ * PQ);

    float tempC = 0.0;
    if (idx < K * PQ * PQ)
    {
        for (int i = 0; i < input_channels * RS * RS; i++)
        {
            tempC += matB[x][i] * matA[i][y];
        }
        matC[x][y] = tempC;
    }
}
#endif
#endif

/*-------------------------------------------------------------------------------------------Inference start-------------------------------------------------------------------------------------------*/

void pass(int argc, char **argv)
{
    cudaError_t err;
#if TRT
    samplesCommon::Args args;

    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    SampleMNISTAPI sample(initializeSampleParams(args));

    // sample::gLogInfo << "Building and running a GPU inference engine for MNIST API" << std::endl;

    sample.build();
#else
    fillWeightWithValues(weight);
    float *d_input, *d_weight, *d_output;

    cudaMalloc((void **)&d_input, BATCH * input_channels * HW * HW * sizeof(float));
    cudaMalloc((void **)&d_weight, RS * RS * K * input_channels * sizeof(float));
    cudaMalloc((void **)&d_output, BATCH * PQ * PQ * K * sizeof(float));

    cudaMemcpy(d_weight, weight, RS * RS * K * input_channels * sizeof(float), cudaMemcpyHostToDevice);
#if CUDNN
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    // Create input tensor
    cudnnTensorDescriptor_t input_descriptor;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           BATCH,
                                           input_channels,
                                           HW,
                                           HW));

    // Create convolutional layer
    cudnnConvolutionDescriptor_t convolution_descriptor;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                    0,
                                    0,
                                    1,
                                    1,
                                    1,
                                    1,
                                    CUDNN_CROSS_CORRELATION,
                                    CUDNN_DATA_FLOAT);

    // Create filter tensor
    cudnnFilterDescriptor_t filter_descriptor;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_descriptor));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_descriptor,
                                           CUDNN_DATA_FLOAT,
                                           CUDNN_TENSOR_NCHW,
                                           K,
                                           input_channels,
                                           RS,
                                           RS));

    // Create output tensor
    int batch_size, channels, height, width;
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
                                                      input_descriptor,
                                                      filter_descriptor,
                                                      &batch_size,
                                                      &channels,
                                                      &height,
                                                      &width));

    cudnnTensorDescriptor_t output_descriptor;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           batch_size,
                                           channels,
                                           height,
                                           width));
#endif

#if UNROLL
    // float *im2col_A_cpu = (float *)malloc(sizeof(float) * RS * RS * input_channels * PQ * PQ);
    float *im2col_A, *gemm_B, *gemm_C;

    cudaMalloc(&im2col_A, sizeof(float) * RS * RS * input_channels * PQ * PQ);
    cudaMalloc(&gemm_B, sizeof(float) * K * input_channels * RS * RS);
    cudaMalloc(&gemm_C, sizeof(float) * PQ * PQ * K);

    cublasHandle_t handle = blas_handle();
    cublasCreate(&handle);
#endif

#endif

    for (int batch = 0; batch < images; batch++)
    {
        fillInputWithValues(input);
        // printf("Hello\n");
#if TRT
#else
        cudaMemcpy(d_input, input, BATCH * input_channels * HW * HW * sizeof(float), cudaMemcpyHostToDevice);
#endif

#if ARRAY_NAIVE
        int threads = min(64, HW * HW);
        int total = K * (PQ * PQ);
        convolution_naive<<<(total + threads - 1) / threads, threads>>>((float(*)[HW][HW])d_input, (float(*)[input_channels][RS][RS])d_weight, (float(*)[PQ][PQ])d_output);
#endif

#if ARRAY_TILING
        dim3 threads(TILE_S, TILE_S);
        dim3 blocks((PQ + LIM - 1) / LIM, (PQ + LIM - 1) / LIM, K);
        convolution_tiling<<<blocks, threads>>>((float(*)[HW][HW])d_input, (float(*)[input_channels][RS][RS])d_weight, (float(*)[PQ][PQ])d_output);
#endif

#if DIRECT
#if CONV_SHARED
        const dim3 numBlocks((PQ + LIM - 1) / LIM, (PQ + LIM - 1) / LIM, K);
        const dim3 threadsPerBlock(1024);
        kernel_conv_filter<<<numBlocks, threadsPerBlock>>>((float(*)[HW][HW])d_input,
#else
        int total = K * PQ * PQ;
        int threads = 64;
        kernel_conv_filter<<<(total + threads - 1) / threads, threads>>>((float(*)[HW][HW])d_input,
#endif
                                                           (float(*)[PQ][PQ])d_output,
                                                           (float(*)[input_channels][RS][RS])d_weight);
#endif

#if CUDNN
        cudnnConvolutionFwdAlgo_t convolution_algorithm;
#if DARKNET
        size_t free_memory, total_memory;
        int requested_algo_count = 10, returned_algo_count = 0;
        float min_time = 1000000; // 1000 sec

        // FWD
        cudnnConvolutionFwdAlgoPerf_t conv_fwd_results[100];
        CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
                                                           input_descriptor,
                                                           filter_descriptor,
                                                           convolution_descriptor,
                                                           output_descriptor,
                                                           requested_algo_count, // (cudnnConvolutionFwdPreference_t)forward_algo,
                                                           &returned_algo_count, // workspace_size_specify,
                                                           conv_fwd_results));

        CHECK_CUDA(cudaMemGetInfo(&free_memory, &total_memory));

        // printf("%d\n", returned_algo_count);

        min_time = 1000000; // 1000 sec
        for (int i = 0; i < returned_algo_count; i++)
        {
            if (conv_fwd_results[i].status == CUDNN_STATUS_SUCCESS &&
                conv_fwd_results[i].algo != CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED &&
                conv_fwd_results[i].memory < free_memory &&
                conv_fwd_results[i].time < min_time)
            {
                convolution_algorithm = conv_fwd_results[i].algo;
                min_time = conv_fwd_results[i].time;
                printf("%d %d %d %d - cuDNN FWD algo: %d, time = %f ms \n", input_channels, HW, K, RS, convolution_algorithm, min_time);
            }
        }
#else
        convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
#endif

        // Allocate memory for workspace
        size_t workspace_size;
        CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                            input_descriptor,
                                                            filter_descriptor,
                                                            convolution_descriptor,
                                                            output_descriptor,
                                                            convolution_algorithm,
                                                            &workspace_size));
        void *workspace_data;
        cudaMalloc(&workspace_data, workspace_size * sizeof(float));

        // Perform convolution
        float alpha = 1.0f, beta = 0.0f;
        CHECK_CUDNN(cudnnConvolutionForward(cudnn,
                                            &alpha,
                                            input_descriptor,
                                            d_input,
                                            filter_descriptor,
                                            d_weight,
                                            convolution_descriptor,
                                            convolution_algorithm,
                                            workspace_data,
                                            workspace_size,
                                            &beta,
                                            output_descriptor,
                                            d_output));

        cudaFree(workspace_data);
#endif

#if TRT

        // trt_call(argc, argv);
        // sample.build();
        sample.infer();
#endif

#if UNROLL
        // im2col_gpu_kernel_ext<<<(N1+K1-1)/K1, K1>>>(PQ*PQ, d_input, HW, HW, RS, RS, 0, 0, STRIDE, STRIDE, 1, 1, PQ, PQ,ic_workspace);
        ///*
        im2col_gpu_kernel<<<(UNROLL_NB + UNROLL_TPB - 1) / UNROLL_TPB, UNROLL_TPB>>>(PQ * PQ * input_channels, // num_kernels, = channels * height_col * width_col;
                                                                                     (float *)d_input,         // data_im,
                                                                                     HW,                       // height,
                                                                                     HW,                       // width,
                                                                                     RS,                       // ksize,
                                                                                     0,                        // pad,
                                                                                     STRIDE,                   // stride,
                                                                                     PQ,                       // height_col,
                                                                                     PQ,                       // width_col,
                                                                                     (float *)im2col_A);       // data_col);

        // start = clock();
        // im2col_cpu((float *)input,
        //            input_channels,
        //            HW, HW, RS, RS,
        //            0, 0,
        //            STRIDE, STRIDE,
        //            0, 0,
        //            (float *)im2col_A_cpu);
        // end = clock();
        // im2col_time = im2col_time + (float)(end - start) / CLOCKS_PER_SEC;
        // cudaMemcpy(im2col_A, im2col_A_cpu, RS * RS * input_channels * PQ * PQ * sizeof(float), cudaMemcpyHostToDevice);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Im2col Error: %s\n", cudaGetErrorString(err));
        }

        // printf("Verifying im2col_A: ");
        // float *verification = (float *)malloc(sizeof(float) * RS * RS * PQ * PQ * input_channels);
        // cudaMemcpy(verification, im2col_A, sizeof(float) * RS * RS * PQ * PQ * input_channels, cudaMemcpyDeviceToHost);
        // verify_im2col(verification, 1.0f);

        int ker_tpb = 512;
        int ker_nb = K * input_channels * RS * RS;
        ker2row_kernel<<<(ker_nb + ker_tpb - 1) / ker_tpb, ker_tpb>>>((float(*)[input_channels * RS * RS]) gemm_B,
                                                                      (float(*)[input_channels][RS][RS])d_weight);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("ker2row Error: %s\n", cudaGetErrorString(err));
        }
        // cudaMemcpy(verification, gemm_B, sizeof(float) * K * input_channels * RS * RS, cudaMemcpyDeviceToHost);
        // verify_im2col(verification, 1.0f);

#if GEMM_GLOBAL
        int total = K * PQ * PQ;
        int threadsPerBlock = min(1024, PQ * PQ);
        gemm_global_kernel<<<(total + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>((float(*)[input_channels * RS * RS]) gemm_B, (float(*)[PQ * PQ]) im2col_A,
                                                                                                 (float(*)[PQ * PQ]) d_output);
#else
        // int m = K;                        // l.n / l.groups
        // int k = input_channels * RS * RS; // l.size*l.size
        // int n = PQ * PQ;                  // l.out_w*l.out_h

        // float *a = gemm_B;   // l.weights_gpu + j*l.nweights / l.groups;
        // float *b = im2col_A; // state.workspace
        // float *c = d_output; // l.output_gpu + (i*l.groups + j)*n*m;

        // gemm_ongpu(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
        // const float alpha = 1, beta = 0;
        // cudaError_t status = (cudaError_t)cublasSgemm(
        //     handle,
        //     CUBLAS_OP_N,
        //     CUBLAS_OP_N,
        //     n,
        //     m,
        //     k,
        //     &alpha,
        //     b,
        //     n,
        //     a,
        //     k,
        //     &beta,
        //     c,
        //     n);

        int m = PQ * PQ;
        int k = input_channels * RS * RS;
        int n = K;

        dim3 dimBlock(TILE_SIZE, TILE_SIZE);
        dim3 dimGrid((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);

        gemm_shared_kernel<<<dimGrid, dimBlock>>>(im2col_A, gemm_B, gemm_C, m, k, n);

        // if (status != cudaSuccess)
        // {
        //     printf("The error is %d", status);
        //     return;
        // }

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("cublass Error: %s\n", cudaGetErrorString(err));
        }
#endif

#endif

#if TRT
#else
        cudaMemcpy(output, d_output, BATCH * PQ * PQ * K * sizeof(float), cudaMemcpyDeviceToHost);
        err = cudaGetLastError();

        // for(int i = 0; i < PQ * PQ * K; i++){
        //	printf("%f\n",output[i]);
        // }

        if (err != cudaSuccess)
        {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        }
#endif
    }

#if TRT
    sample.teardown();
#else
    if (debug)
    {
        verification(input, weight, output);
    }
#if CUDNN
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudnnDestroy(cudnn);
#endif

#if UNROLL
    cudaFree(im2col_A);
    cudaFree(gemm_B);
    cudaFree(gemm_C);
    cublasDestroy(handle);
#endif

    cudaFree(d_output);
    cudaFree(d_weight);
    cudaFree(d_input);
#endif
}

int main(int argc, char **argv)
{
    pass(argc, argv);
    // printf("Im2col time takes %f seconds", im2col_time);
    // printf("Creating buffer Manager %f seconds\n",buffManager);
    // printf("Processing input %f seconds\n",process);
    // printf("Copying Input to Device %f seconds\n",ItD);
    // printf("Copying Output to Host %f seconds\n",OtH);
    // printf("Executing %f seconds\n",exec);
#if TRT
#else
    free(output);
    free(weight);
    free(input);
#endif
}
