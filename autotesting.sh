#!/bin/bash

out_path=(unroll_cublass tensorrt cudnn cudnn_opt mbnet_method)
metrics=(
	None
	# sm_efficiency achieved_occupancy warp_execution_efficiency inst_per_warp gld_efficiency gst_efficiency shared_efficiency shared_utilization
	# l2_utilization global_hit_rate tex_cache_hit_rate tex_utilization ipc inst_issued inst_executed issue_slot_utilization dram_utilization
)
# metric=(stall_sync stall_other stall_not_selected stall_memory_throttle stall_memory_dependency stall_inst_fetch stall_exec_dependency stall_constant_memory_dependency)

for j in ${!out_path[@]}; do
	for i in ${!metrics[@]}; do
		./tester.sh ${out_path[$j]} ${metrics[$i]}
	done
	sleep 10
done
