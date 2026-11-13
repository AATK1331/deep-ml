def classify_llm_phases(num_params: int, sequence_length: int, batch_size: int, bytes_per_param: int, peak_flops: float, peak_bandwidth: float) -> dict:
    """
    Analyze prefill and decode phases of LLM inference using the Roofline Model.

    Args:
        num_params: Total number of model parameters
        sequence_length: Number of input tokens processed during prefill
        batch_size: Number of sequences processed in parallel during decode
        bytes_per_param: Memory footprint per parameter (e.g., 2 for FP16)
        peak_flops: Hardware peak compute throughput (FLOP/s)
        peak_bandwidth: Hardware peak memory bandwidth (bytes/s)

    Returns:
        Dictionary containing ridge_point and analysis dicts for 'prefill' and 'decode',
        each with total_flops, memory_bytes, arithmetic_intensity, bottleneck,
        achieved_flops, and utilization_percent.
    """
    ridge_point=peak_flops/peak_bandwidth

    prefill_flops = 2 * num_params * sequence_length
    prefill_bytes = num_params * bytes_per_param
    prefill_ai = prefill_flops / prefill_bytes
    
    if prefill_ai >= ridge_point: 
        prefill_bottleneck = 'compute-bound'
        prefill_achieved = peak_flops
    else:
        prefill_bottleneck = 'memory-bound'
        prefill_achieved = prefill_ai * peak_bandwidth
        
    prefill_util = (prefill_achieved / peak_flops) * 100


    decode_flops = 2 * num_params * batch_size
    decode_bytes = num_params * bytes_per_param
    decode_ai = decode_flops / decode_bytes
    
    if decode_ai >= ridge_point:
        decode_bottleneck = 'compute-bound'
        decode_achieved = peak_flops
    else:
        decode_bottleneck = 'memory-bound'
        decode_achieved = decode_ai * peak_bandwidth
        
    decode_util = (decode_achieved / peak_flops) * 100
    
    return {
        'ridge_point': round(ridge_point, 4),
        'prefill': {
            'total_flops': prefill_flops,
            'memory_bytes': prefill_bytes, 
            'arithmetic_intensity': round(prefill_ai, 4), 
            'bottleneck': prefill_bottleneck, 
            'achieved_flops': round(prefill_achieved, 4), 
            'utilization_percent': round(prefill_util, 4)
        }, 
        'decode': {
            'total_flops': decode_flops, 
            'memory_bytes': decode_bytes, 
            'arithmetic_intensity': round(decode_ai, 4), 
            'bottleneck': decode_bottleneck, 
            'achieved_flops': round(decode_achieved, 4), 
            'utilization_percent': round(decode_util, 4)  
        }
    }

    
    
    
         