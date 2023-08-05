nvprof_paths = ['direct_shared',
                'unroll_cublass', 'tensorrt', 'cudnn', 'cudnn_opt']  # folder paths

# nvprof_paths = ['unroll_cublass']  # folder paths

metrics = ['sm_efficiency', 'achieved_occupancy', 'warp_execution_efficiency', 'inst_per_warp', 'gld_efficiency', 'gst_efficiency', 'shared_efficiency', 'shared_utilization',
           'l2_utilization', 'global_hit_rate', 'tex_cache_hit_rate', 'tex_utilization', 'ipc', 'inst_issued', 'inst_executed', 'issue_slot_utilization', 'dram_utilization']