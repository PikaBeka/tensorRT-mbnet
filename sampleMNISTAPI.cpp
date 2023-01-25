/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//!
//! sampleMNISTAPI.cpp
//! This file contains the implementation of the MNIST API sample. It creates the network
//! for MNIST classification using the API.
//! It can be run with the following command line:
//! Command: ./sample_mnist_api [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//! [--useDLACore=<int>]
//!
// g++ -std=c++11 -o sample -I /usr/local/cuda/targets/x86_64-linux/include/ -I /usr/local/cuda/include -L/$CUDA_HOME/lib64 *.cpp *.cc *.pb.cc -lnvinfer -lcuda -lcudart -lnvonnxparser -pthread -lprotobuf -lpthread -w

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "mbnet.h"

// #include "NvCaffeParser.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include "slenet_params.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

using samplesCommon::SampleUniquePtr;

const std::string gSampleName = "TensorRT.sample_mnist_api";

float *input = (float *)malloc(sizeof(float) * C * HW * HW);
float *weight = (float *)malloc(sizeof(float) * RS * RS * K * C);
float *bias = (float *)malloc(sizeof(float) * K);
float *output = (float *)malloc(sizeof(float) * K * PQ * PQ);

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

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

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
        mParams.inputTensorNames[0].c_str(), DataType::kFLOAT, Dims4{1, C, HW, HW});
    ASSERT(data);

    // Add convolution layer with 20 outputs and a 5x5 filter.
    IConvolutionLayer *conv1 = network->addConvolutionNd(
        *data, K, Dims{2, {5, 5}}, mWeightMap["c1_weight"], mWeightMap["c1_bias"]);
    conv1->setStride(DimsHW{1, 1});
    conv1->setPadding(DimsHW{0, 0});
    ASSERT(conv1);

    // Add softmax layer to determine the probability.
    // ISoftMaxLayer *prob = network->addSoftMax(*sigmoid3->getOutput(0));
    // ASSERT(prob);
    conv1->getOutput(0)->setName(mParams.outputTensorNames[0].c_str());
    network->markOutput(*conv1->getOutput(0));

    // Build engine
    config->setMaxWorkspaceSize(16_MiB);
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
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Read the input data into the managed buffers
    ASSERT(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers, input))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Verify results
    if (!verifyOutput(buffers, input, weight))
    {
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

    float *hostDataBuffer = static_cast<float *>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    float temp = 1.0f;

    for (int i = 0; i < C; i++)
    {
        for (int j = 0; j < HW; j++)
        {
            for (int k = 0; k < HW; k++)
            {
                temp = (float)(rand() % 100) - 100.0;
                input[i * HW * HW + j * HW + k] = temp;
                hostDataBuffer[i * HW * HW + j * HW + k] = temp;
            }
        }
    }

    // for (int i = 0; i < C; i++)
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
    printf("The configuration is %d_%d_%d\n", C, HW, K);

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
                for (int l = 0; l < C; l++)
                {
                    for (int m = 0; m < RS; m++)
                    {
                        for (int t = 0; t < RS; t++)
                        {
                            tempC += weight[i * C * RS * RS + l * RS * RS + m * RS + t] * input[l * HW * HW + (j + m) * HW + (k + t)];
                        }
                    }
                }
                if (output[i * PQ * PQ + j * PQ + k] != tempC)
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
        for (int t = 0; t < C; t++)
        {
            for (int j = 0; j < RS; j++)
            {
                for (int k = 0; k < RS; k++)
                {
                    weight[i * (C * RS * RS) + t * (RS * RS) + j * RS + k] = (float)(rand() % 100) - 100.0;
                    // weight[i * (C * RS * RS) + t * (RS * RS) + j * RS + k] = 1.0f;
                }
            }
        }
    }

    Weights wt_c1_weight{DataType::kFLOAT, nullptr, 0};
    wt_c1_weight.type = DataType::kFLOAT;
    wt_c1_weight.values = weight;
    wt_c1_weight.count = K * C * RS * RS;
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
        params.dataDirs.push_back("data/mnist/");
        params.dataDirs.push_back("data/samples/mnist/");
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
    params.weightsFile = "mnistapi.wts";
    params.mnistMeansProto = "mnist_mean.binaryproto";

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

int main(int argc, char **argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    SampleMNISTAPI sample(initializeSampleParams(args));

    sample::gLogInfo << "Building and running a GPU inference engine for MNIST API" << std::endl;

    if (!sample.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    if (!sample.infer())
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    if (!sample.teardown())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    return sample::gLogger.reportPass(sampleTest);
}
