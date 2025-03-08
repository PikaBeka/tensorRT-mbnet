#!/bin/bash

out_path=(mbnet_method)
metrics=(sm_efficiency achieved_occupancy warp_execution_efficiency inst_per_warp gld_efficiency gst_efficiency shared_efficiency shared_utilization
           l2_utilization global_hit_rate tex_cache_hit_rate tex_utilization ipc inst_issued inst_executed issue_slot_utilization dram_utilization)
# metric=(stall_sync stall_other stall_not_selected stall_memory_throttle stall_memory_dependency stall_inst_fetch stall_exec_dependency stall_constant_memory_dependency)

for j in ${!out_path[@]}; do
	# for i in ${!metric[@]}; do
	./tester.sh ${out_path[$j]}
	sleep 10
	# done
done
