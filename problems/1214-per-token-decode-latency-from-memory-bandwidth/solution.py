def estimate_decode_latency(num_params, bytes_per_param, bandwidth_bytes_per_s):
    # Return [latency_ms, tokens_per_sec]
    


   latency_ms=(num_params*bytes_per_param)/bandwidth_bytes_per_s*1000

   tokens_per_sec=1000/latency_ms

   return [latency_ms,tokens_per_sec]