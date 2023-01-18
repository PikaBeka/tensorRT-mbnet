# tensorRT-mbnet
micro bench-marking with tensorrt engine

To compile: g++ -std=c++11 -o sample -I /usr/local/cuda/targets/x86_64-linux/include/ -I /usr/local/cuda/include -L/$CUDA_HOME/lib64 *.cpp *.cc -lnvinfer -lcuda -lcudart -lnvonnxparser -pthread -lprotobuf -lpthread -w
To execute: ./sample --datadir=./data
