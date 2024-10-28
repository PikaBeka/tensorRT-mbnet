#define images 10000 // number of images in batch
#define BATCH 1

#define input_channels 1024
#define HW 29
#define K 255

#define RS 1
#define STRIDE 1

#define PQ (HW - RS + 1) // output height and width (146)

#define TILE_SIZE 32
#define TILE_S 8
#define LIM (TILE_S - RS + 1) // 4

#define ARRAY_NAIVE 0

#define ARRAY_TILING 0

#define DIRECT 0
#define CONV_SHARED 0

#define CUDNN 0
#define DARKNET 0

#define TRT 0

#define UNROLL 1
#define GEMM_GLOBAL 0

#define CONV_TPB 1024
#define GRID ((PQ + LIM - 1) / LIM) // (37)
#define CONV_NB (K * GRID * GRID)   // 1369 // threads per block (150)

#define UNROLL_TPB 1024
#define UNROLL_NB (PQ * PQ * input_channels)
