#!/bin/bash

out_path=(direct_shared unroll_cublass tenssort cudnn cudnn_opt)
metric=None

for j in ${!out_path[@]}; do
	./tester.sh ${out_path[$j]} ${metric}
done