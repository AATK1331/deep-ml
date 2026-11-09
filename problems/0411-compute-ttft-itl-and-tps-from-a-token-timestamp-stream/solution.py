def compute_inference_metrics(timestamps: list[float]) -> dict:
	"""
	Compute LLM inference performance metrics from token timestamps.
	
	Args:
		timestamps: List of floats where timestamps[0] is the request start time
		            and timestamps[1:] are the times when each output token was generated.
	
	Returns:
		Dictionary with keys 'ttft', 'tps', 'itl' containing the metric values.
	"""
	
	ttft=timestamps[1]-timestamps[0]
	num_tokens=len(timestamps)-1
	total_time=timestamps[-1]-timestamps[0]
	tps=num_tokens/total_time if total_time>0 else 0.0
	if num_tokens<=1:
		itl=0.0
	else:
		itl=(timestamps[-1]-timestamps[1])/(num_tokens-1)

	return {
		"ttft":ttft,
		"tps":tps,
		"itl":itl,
	}