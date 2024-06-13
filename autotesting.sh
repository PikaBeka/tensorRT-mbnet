#!/bin/bash

out_path=(unroll_global)
metric=None

for j in ${!out_path[@]}; do
	./tester.sh ${out_path[$j]} ${metric}
	#sleep 5m
done
