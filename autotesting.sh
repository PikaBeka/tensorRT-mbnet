#!/bin/bash

out_path=(unroll_cublass cudnn cudnn_opt tensorrt)
metrics=(sm_efficiency achieved_occupancy warp_execution_efficiency inst_per_warp gld_efficiency gst_efficiency shared_efficiency shared_utilization
           l2_utilization global_hit_rate tex_cache_hit_rate tex_utilization ipc inst_issued inst_executed issue_slot_utilization dram_utilization)
metric=gld_efficiency

for j in ${!out_path[@]}; do
	./tester.sh ${out_path[$j]} ${metric}
	sleep 1m
done
