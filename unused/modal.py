async def process_modal_prompts_async(
    ids: Union[np.ndarray, list[int]],
    prompts: list[Prompt],  # each prompt is just a list of messages
    models: list[str],
    model_weights: list[float],
    sampling_params: list[SamplingParams],
    batch_size: int = 1_000,
    progress_bar: Optional[tqdm] = None
):
    # change ids to integer list
    if isinstance(ids, np.ndarray):
        ids = ids.tolist()

    # normalize weights
    model_weights = [w / sum(model_weights) for w in model_weights]
    
    # make sure ids and prompts are the same length
    if len(ids) != len(prompts):
        raise ValueError("ids and prompts must be the same length.")
    
    # if dry run, just directly create list of APIResponse objects with no completion and return them
    # look up the models
    completion_fns = [
        f'{registry[model]["name"]}-completions-{registry[model]["gpus"][0]}' for model in models
    ]
    completion_fns = [
        modal.Function.lookup(f, "Model.generate") for f in completion_fns
    ]

    # split into batches
    batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
    batch_ids = [ids[i:i+batch_size] for i in range(0, len(ids), batch_size)]

    # iterate over batches, assigning each to model randomly & creating async task
    tasks = []
    for i, b in zip(batch_ids, batches):
        model_idx = np.random.choice(range(len(models)), p=model_weights)
        tasks.append(asyncio.create_task(
            completion_fns[model_idx].remote.aio(i, b, sampling_params[model_idx].__dict__)
        ))
    
    # gather them as they're completed, return the results
    results = []
    for task in asyncio.as_completed(tasks):
        results.extend(await task)
        if progress_bar:
            progress_bar.update(batch_size)

    return [
        APIResponse(**r) for r in results
    ]



# split prompts between api and modal
modal_weight = sum([
    self.model_weights[i] for i, model in enumerate(self.models) if registry[model]["api_spec"] == "modal"
])
modal_ids_mask = np.random.binomial(1, modal_weight, size=len(remaining_ids)).astype(bool)
modal_ids = remaining_ids[modal_ids_mask]
api_ids = remaining_ids[~modal_ids_mask]
print(f"Split into {len(modal_ids)} Modal prompts and {len(api_ids)} api prompts.")


# decide which prompts go to which models
modal_prompts = [prompts[i] for i in modal_ids] # indexes into original prompts
api_prompts = [prompts[i] for i in api_ids] # indexes into original prompts
modal_models = [model for model in self.models if registry[model]["api_spec"] == "modal"]
modal_weights = [self.model_weights[i] for i, model in enumerate(self.models) if registry[model]["api_spec"] == "modal"]
modal_sampling_params = [self.sampling_params[i] for i, model in enumerate(self.models) if registry[model]["api_spec"] == "modal"]
api_models = [model for model in self.models if registry[model]["api_spec"] != "modal"]
api_weights = [self.model_weights[i] for i, model in enumerate(self.models) if registry[model]["api_spec"] != "modal"]
api_sampling_params = [self.sampling_params[i] for i, model in enumerate(self.models) if registry[model]["api_spec"] != "modal"]