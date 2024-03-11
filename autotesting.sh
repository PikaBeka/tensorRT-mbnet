#!/bin/bash

out_path=(unroll_cublass tensorrt cudnn cudnn_opt mbnet_method)
metric=None

for j in ${!out_path[@]}; do
	./tester.sh ${out_path[$j]} ${metric}
done
