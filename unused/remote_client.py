class RemoteLLMClient:
    def __init__(
        self, 
        model_names: list[str],
        max_requests_per_minute: int,
        max_tokens_per_minute: int,
        sampling_params: Union[SamplingParams, list[SamplingParams]] = SamplingParams(),
        model_weights: Union[list[float], Literal["uniform", "rate_limit"]] = "uniform",
        max_attempts: int = 5,
        request_timeout: int = 30,
        use_qps: bool = False,
        debug: bool = False
    ):
        self.client = ModalLLMClient(
            model_names=model_names,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            sampling_params=sampling_params,
            model_weights=model_weights,
            max_attempts=max_attempts,
            request_timeout=request_timeout,
            use_qps=use_qps,
            debug=debug
        )

    def process_prompts_sync(
        self,
        prompts: Union[list[str], list[list[dict]]],
        return_completions_only: bool = False,
        show_progress=True
    ):
        import asyncio
        return asyncio.run(
            self.process_prompts_async(
                prompts=prompts,
                return_completions_only=return_completions_only,
                show_progress=show_progress
            )
        )
        
    async def process_prompts_async(
        self,
        prompts: Union[list[str], list[list[dict]]],
        return_completions_only: bool = False,
        show_progress=True
    ):
        from .api_requests.base import APIResponse
        outputs = self.client.process_prompts.remote(prompts)
        resps = [
            APIResponse.from_dict(x) for x in outputs
        ]
        if return_completions_only:
            return [r.completion for r in resps]
        
        return resps
    
    @classmethod
    def from_config(cls, config: ClientConfig):
        return cls(
            model_names=config.model_names,
            max_requests_per_minute=config.max_requests_per_minute,
            max_tokens_per_minute=config.max_tokens_per_minute,
            sampling_params=config.sampling_params,
            model_weights=config.model_weights,
            max_attempts=config.max_attempts,
            request_timeout=config.request_timeout
        )