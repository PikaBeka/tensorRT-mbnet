#!/bin/bash

out_path=(unroll_cublass tensorrt cudnn cudnn_opt mbnet) # folder names created, output path for created txt files

metrics=(sm_efficiency achieved_occupancy warp_execution_efficiency inst_per_warp gld_efficiency gst_efficiency shared_efficiency shared_utilization
           l2_utilization global_hit_rate tex_cache_hit_rate tex_utilization ipc inst_issued inst_executed issue_slot_utilization dram_utilization)

# configuratin files in the format (C, HW, K)
#C=(1 1 1 1 1 1 1 1 1 1 3 3 3 6 6 6 6 6 6 6 6 6 16 16 16 16 16 16 32 32) # 30
#HW=(256 400 320 256 128 32 256 64 256 64 150 64 32 150 128 70 32 16 8 32 16 8 32 16 8 32 16 8 32 8) # 30
#K=(3 6 6 6 6 6 9 9 12 12 16 16 16 16 16 16 16 16 16 32 32 32 32 32 32 64 64 64 64 64) # 30

# AlexNet
#C=(3 96 256 384 384)
#HW=(64 26 12 12 12)
#K=(96 256 384 384 256)
#RS=(11 5 3 3 3)
#TILE_S=(14 8 6 5 5)

# VGGNet-16
#C=(3 64 64 128 128 256 256 256 512 512 512 512 512)
#HW=(224 224 112 112 56 56 56 28 28 28 14 14 14)
#K=(64 64 128 128 256 256 256 512 512 512 512 512 512)
#RS=(3 3 3 3 3 3 3 3 3 3 3 3 3)

# YOLOV4
C=(3 32 64 64 64 32 64 64 64 128 64 64 64 64 64 64 64 128 256 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 256 512 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 512 1024 512 512 512 512 512 512 512 512 512 512 512 1024 512 1024 512 512 1024 512 256 256 256 512 256 512 256 128 128 128 256 128 256 128 256 255 256 256 512 256 512 256 512 255 512 512 1024 512 1024 512 1024 )
HW=(608 304 304 306 308 310 310 312 157 157 159 161 163 163 165 165 167 85 85 87 89 91 91 93 93 95 95 97 97 99 99 101 101 103 103 105 105 107 55 55 57 59 61 61 63 63 65 65 67 67 69 69 71 71 73 73 75 75 77 40 40 42 44 46 46 48 48 50 50 52 52 54 56 58 58 49 51 51 53 55 57 59 59 61 61 63 65 67 69 69 71 71 73 73 38 38 40 40 42 42 44 44 23 23 25 25 27 27 29 29 )
K=(32 64 64 64 32 64 64 64 128 64 64 64 64 64 64 64 128 256 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 256 512 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 512 1024 512 512 512 512 512 512 512 512 512 512 512 1024 512 1024 512 512 1024 512 256 256 256 512 256 512 256 128 128 128 256 128 256 128 256 255 256 256 512 256 512 256 512 255 512 512 1024 512 1024 512 1024 255 )
RS=(3 3 1 1 1 3 1 1 3 1 1 1 3 1 3 1 1 3 1 1 1 3 1 3 1 3 1 3 1 3 1 3 1 3 1 3 1 1 3 1 1 1 3 1 3 1 3 1 3 1 3 1 3 1 3 1 3 1 1 3 1 1 1 3 1 3 1 3 1 3 1 1 1 3 1 1 3 1 1 1 1 3 1 3 1 1 1 1 3 1 3 1 3 1 3 1 3 1 3 1 3 1 3 1 3 1 3 1 3 1 )

#YOLOV4-TINY
#C=(3 32 64 64 32 32 64 128 64 64 128 256 128 128 256 512 256 512 255 128 256)
#HW=(416 208 104 104 104 104 53 53 53 53 27 27 27 27 14 14 16 16 18 20 20)
#K=(32 64 64 32 32 64 128 64 64 128 256 128 128 256 512 256 512 255 128 256 255)
#RS=(3 3 3 3 3 1 3 3 3 1 3 3 3 1 3 1 3 1 1 3 1)

#C=(3)
#HW=(64)
#K=(16)
#RS=(5)
#TILE_S=(8)

#C=(3 3 3 6 6 6 6 6 6 6 6 6 16 16 16 16 16 16 32 32 64 128 128 128 256 256 256 512)
#HW=(150 64 32 150 128 70 32 16 8 32 16 8 32 16 8 32 16 8 32 8 64 56 28 14 256 128 64 56)
#K=(16 16 16 16 16 16 16 16 16 32 32 32 32 32 32 64 64 64 64 64 256 256 512 256 128 64 56 28)

#C=(3 16 32 64 128 256 512 1024 256 512 256 384 256)
#HW=(416 208 104 52 26 13 13 13 13 13 13 13 13)
#K=(16 32 64 128 256 512 1024 256 512 255 128 256 255)
#RS=(3 3 3 3 3 3 3 1 3 1 1 3 1)

#C=(3 64 64 128 128 256 256 256 256 512 512 512 512 512 512 3 64 192 192 192 256 256 256 64 64 128 128 256)
#HW=(224 224 112 112 56 56 56 28 28 28 14 14 14 224 56 28 28 28 28 28 28 56 56 28 28 14 14 7)
#K=(64 64 128 128 256 256 256 512 512 512 512 512 512 64 192 64 96 16 128 128 32 64 128 128 256 256 512 512)
#RS=(3 3 3 3 3 3 3 3 3 3 3 3 3 7 3 1 3 5 1 3 3 3 3 3 3 3 3 3)

# ----- ------ List of all sizes
#C=(3 3 3 6 6 6 6 6 6 6 6 6 16 16 16 16 16 16 32 32 64 128 128 128 256 256 256 512 1 1 1 1 1 1 1 1 1 1 3 96 256 384 384 3 16 32 64 128 512 1024 256 512 256 384 256 3 64 64 128 128 256 256 512 512 3 64 512 512 192 192 192 256 256 256 64 64 128 128 256)

#HW=(150 64 32 150 128 70 32 16 8 32 16 8 32 16 8 32 16 8 32 8 64 56 28 14 256 128 64 56 256 400 320 256 128 32 256 64 256 64 64 26 12 12 12 416 208 104 52 26 13 13 13 13 13 13 13 224 224 112 112 56 56 28 28 56 28 28 14 224 28 28 28 28 56 56 28 28 14 14 7)

#K=(16 16 16 16 16 16 16 16 16 32 32 32 32 32 32 64 64 64 64 64 256 256 512 256 128 64 56 28 3 6 6 6 6 6 9 9 12 12 96 256 384 384 256 16 32 64 128 256 1024 256 512 255 128 256 255 64 64 128 128 256 256 512 512 192 64 96 512 64 16 128 128 32 64 128 128 256 256 512 512)

#RS=(5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 11 5 3 3 3 3 3 3 3 3 3 1 3 1 1 3 1 3 3 3 3 3 3 3 3 3 1 3 3 7 5 1 3 3 3 3 3 3 3 3 3)

#C=(6 6 6 6 6 256 1 192 128 64 256 512 1 1 16 1)
#HW=(16 16 8 8 128 56 32 28 56 64 56 28 320 256 208 256)
#K=(16 32 32 16 16 64 6 16 256 256 256 512 6 6 32 9)
#RS=(5 5 5 5 5 3 5 5 5 5 3 3 5 5 3 5)

#C=(1)
#HW=(320)
#K=(6)
#RS=(5)

#input file to change macro define
in_file=mbnet.h

is_metrics=false
is_trace=false
# is_jetson = /usr/local/cuda-10.2/bin/

# This block of code required to clear all macro defines
sed -i 's/define ARRAY_NAIVE .*/define ARRAY_NAIVE 0/' $in_file
sed -i 's/define ARRAY_TILING .*/define ARRAY_TILING 0/' $in_file
sed -i 's/define DIRECT .*/define DIRECT 0/' $in_file
sed -i 's/define CONV_SHARED .*/define CONV_SHARED 0/' $in_file
sed -i 's/define CUDNN .*/define CUDNN 0/' $in_file
sed -i 's/define DARKNET .*/define DARKNET 0/' $in_file
sed -i 's/define TRT .*/define TRT 0/' $in_file
sed -i 's/define GEMM_GLOBAL .*/define GEMM_GLOBAL 0/' $in_file
sed -i 's/define UNROLL .*/define UNROLL 0/' $in_file

mkdir -p "metrics"

method=$1
metric=$2

echo ${method}
echo ${metric}

if [ "${method}" != "array_naive" ] && [ "${method}" != "array_tiling" ] && [ "${method}" != "direct_shared" ] && [ "${method}" != "unroll_cublass" ] && [ "${method}" != "unroll_global" ] && [ "${method}" != "tensorrt" ] && [ "${method}" != "cudnn" ] && [ "${method}" != "cudnn_opt" ] && [ "${method}" != "mbnet_method" ]; then
    echo "ERROR: Please supply one of the methods: array_naive, array_tiling, direct_shared, unroll_cublass, tensorrt, cudnn"
    exit
fi

mkdir -p "${method}"
mkdir -p "trace/${method}"

for i in ${!metrics[@]}; do
    for j in ${!out_path[@]}; do
        mkdir -p "metrics/${metrics[$i]}/${out_path[$j]}"
    done
done

if [[ "${metric}" != "None" ]]; then
    is_metrics=true
fi

echo ${is_metrics}

for i in ${!C[@]}; do # loop to place all configuration files into use

    sed -i 's/define ARRAY_NAIVE .*/define ARRAY_NAIVE 0/' $in_file
    sed -i 's/define ARRAY_TILING .*/define ARRAY_TILING 0/' $in_file
    sed -i 's/define DIRECT .*/define DIRECT 0/' $in_file
    sed -i 's/define CONV_SHARED .*/define CONV_SHARED 0/' $in_file
    sed -i 's/define CUDNN .*/define CUDNN 0/' $in_file
    sed -i 's/define DARKNET .*/define DARKNET 0/' $in_file
    sed -i 's/define TRT .*/define TRT 0/' $in_file
    sed -i 's/define GEMM_GLOBAL .*/define GEMM_GLOBAL 0/' $in_file
    sed -i 's/define UNROLL .*/define UNROLL 0/' $in_file

    if [[ "${method}" == "array_naive" ]]; then
	echo 'Running mbnet with array_naive method\n'
        sed -i 's/define ARRAY_NAIVE .*/define ARRAY_NAIVE 1/' ${in_file}
    fi

    if [[ "${method}" == "array_tiling" ]]; then
	echo 'Running mbnet with array_tiling method\n'
        sed -i 's/define ARRAY_TILING .*/define ARRAY_TILING 1/' $in_file
    fi

    if [[ "${method}" == "direct_shared" ]]; then
	echo 'Running mbnet with direct_shared method\n'
        sed -i 's/define DIRECT .*/define DIRECT 1/' $in_file
        sed -i 's/define CONV_SHARED .*/define CONV_SHARED 1/' $in_file
    fi

    if [[ "${method}" == "unroll_cublass" ]]; then
	echo 'Running mbnet with unroll_cublass method\n'
        sed -i 's/define UNROLL .*/define UNROLL 1/' $in_file
    fi

    if [[ "${method}" == "unroll_global" ]]; then
	echo 'Running mbnet with unroll_global method\n'
        sed -i 's/define UNROLL .*/define UNROLL 1/' $in_file
	sed -i 's/define GEMM_GLOBAL .*/define GEMM_GLOBAL 1/' $in_file
    fi

    if [[ "${method}" == "tensorrt" ]]; then
	sed -i 's/define TRT .*/define TRT 1/' $in_file
	echo 'Running mbnet with tenssort method\n'
    fi

    if [[ "${method}" == "cudnn" ]]; then
	echo 'Running mbnet with cudnn method\n'
        sed -i 's/define CUDNN .*/define CUDNN 1/' $in_file
    fi

    if [[ "${method}" == "cudnn_opt" ]]; then
	echo 'Running mbnet with cudnn optimized method\n'
        sed -i 's/define CUDNN .*/define CUDNN 1/' $in_file
	sed -i 's/define DARKNET .*/define DARKNET 1/' $in_file
    fi

    if [[ "${method}" == "mbnet_method" ]]; then
	if [[ ${HW[$i]} -lt 18 ]];
        then
            if [[ ${RS[$i]} -lt 4 ]];
            then
            	if [[ ${RS[$i]} -lt 2 ]];
            	then
            		echo 'Running mbnet with unroll_cublass method\n'
    			sed -i 's/define UNROLL .*/define UNROLL 1/' $in_file
            	else
            		if [[ ${C[$i]} -lt 499 ]];
            		then
            			echo 'Running mbnet with unroll_cublass method\n'
    				sed -i 's/define UNROLL .*/define UNROLL 1/' $in_file
            		else
            			echo 'Running mbnet with unroll_cublass method\n'
    				sed -i 's/define UNROLL .*/define UNROLL 1/' $in_file
            		fi
            	fi
            else
            	echo 'Running mbnet with unroll_cublass method\n'
    		sed -i 's/define UNROLL .*/define UNROLL 1/' $in_file
            fi
      else
            if [[ ${K[$i]} -lt 392 ]];
            then
            	if [[ ${C[$i]} -lt 405 ]];
	     	then
       			if [[ ${K[$i]} -lt 203 ]];
	  		then
     				echo 'Running mbnet with tensorrt method\n'
    				sed -i 's/define TRT .*/define TRT 1/' $in_file
     			else
				echo 'Running mbnet with unroll_cublass method\n'
    				sed -i 's/define UNROLL .*/define UNROLL 1/' $in_file
			fi
       		else
	 		echo 'Running mbnet with unroll_cublass method\n'
    			sed -i 's/define UNROLL .*/define UNROLL 1/' $in_file
	 	fi
            else
            	echo 'Running mbnet with unroll_cublass method\n'
    		sed -i 's/define UNROLL .*/define UNROLL 1/' $in_file
            fi
        fi
    fi

    sed -i 's/define input_channels .*/define input_channels '${C[$i]}'/' ${in_file} # change C
    sed -i 's/define HW .*/define HW '${HW[$i]}'/' ${in_file} # change HW
    sed -i 's/define K .*/define K '${K[$i]}'/' ${in_file} # change K
    sed -i 's/define RS .*/define RS '${RS[$i]}'/' ${in_file} # change RS
    #sed -i 's/define TILE_S .*/define TILE_S '${TILE_S[$i]}'/' ${in_file} # change TILE_S

    /usr/local/cuda/bin/nvcc -o mbnet trt_dependencies/*.cpp trt_dependencies/*.cc mbnet.cu -lnvinfer -lcuda -lnvonnxparser -lcudart -lcublas -lcudnn -lprotobuf -lpthread -lstdc++ -lm -w

    if [[ "$is_metrics" = true ]]
    then
    #echo 'metrics run'
        /usr/local/cuda/bin/nvprof --aggregate-mode on --log-file metrics/${metric}/${method}/nvprof_comp_${C[$i]}_${HW[$i]}_${K[$i]}_${RS[$i]}.txt --metrics ${metric} ./mbnet #sm_efficiency,achieved_occupancy,warp_execution_efficiency,inst_per_warp,gld_efficiency,gst_efficiency,shared_efficiency,shared_utilization,l2_utilization,global_hit_rate,tex_cache_hit_rate,	tex_utilization,ipc,inst_issued,inst_executed,issue_slot_utilization,dram_utilization ./mbnet # stroe nvprof into the txt file
    else
        if [[ "$is_trace" = true ]]
        then
            /usr/local/cuda/bin/nvprof --log-file trace/${method}/nvprof_comp_${C[$i]}_${HW[$i]}_${K[$i]}_${RS[$i]}.txt --print-gpu-trace ./mbnet
        else
            /usr/local/cuda/bin/nvprof --log-file ${method}/nvprof_comp_${C[$i]}_${HW[$i]}_${K[$i]}_${RS[$i]}.txt ./mbnet # stroe nvprof into the txt file 
        fi
    fi
done
