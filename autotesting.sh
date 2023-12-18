#!/bin/bash

out_path=(tensorrt)
metric=None

for j in ${!out_path[@]}; do
	./tester.sh ${out_path[$j]} ${metric}
done
