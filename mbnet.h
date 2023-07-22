#define images 1 // number of images in batch
#define BATCH 1

#define input_channels 6
#define HW 8
#define K 32

#define RS 5 // kernel height and width
#define STRIDE 1

#define PQ (HW - RS + 1) // output height and width (146)

#define TILE_S 8
#define LIM (TILE_S - RS + 1) // 4

#define ARRAY_NAIVE 0

#define ARRAY_TILING 0

#define DIRECT 0
#define CONV_SHARED 0

#define CUDNN 0
#define DARKNET 0

#define TRT 0

#define GEMM_GLOBAL 0

#define GRID ((PQ + LIM - 1) / LIM)                     // (37)
#define CONV_NB (GRID * GRID)                           // 1369
#define CONV_TPB MIN(1024, MAX(K *LIM *LIM, K *RS *RS)) // threads per block (150)

#define UNROLL_TPB MIN(1024, K *RS *RS)
#define UNROLL_NB ((PQ * PQ * C + UNROLL_TPB) / UNROLL_TPB)
