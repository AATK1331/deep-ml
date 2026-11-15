def estimate_kv_cache_size(model_config: dict, batch_size: int, seq_len: int) -> dict:
    """
    Estimate the KV cache memory footprint for a Transformer model.

    Args:
        model_config: Dictionary with model architecture parameters
        batch_size: Number of sequences in the batch
        seq_len: Number of cached tokens

    Returns:
        Dictionary with cache size estimates
    """
    # Your code here
    
    H_kv=model_config.get('num_kv_heads',model_config['num_attention_heads'])
    d_head=model_config['hidden_size']/model_config['num_attention_heads']

    kv_cache_elements=batch_size*model_config['num_layers']*2*H_kv*seq_len*d_head

    kv_cache_size_bytes=kv_cache_elements*model_config['dtype_bytes']

    kv_cache_size_mb=kv_cache_size_bytes/(1024*1024)

    per_layer_size_mb=kv_cache_size_mb/model_config['num_layers']


    per_token_size_kb=(model_config['num_layers']*2*H_kv*d_head*model_config['dtype_bytes'])/1024

    return {
        'kv_cache_elements':int(kv_cache_elements),
        'kv_cache_size_bytes':int(kv_cache_size_bytes),
        'kv_cache_size_mb':round(kv_cache_size_mb,4),
        'per_layer_size_mb':round(per_layer_size_mb,4),
        'per_token_size_kb':round(per_token_size_kb,4)
    }