"""Explicit, independently selectable inference experiments; defaults stay unchanged."""


def configure_native(binary, *, smallm_kernel='dot', head_warps=None, prefill_gemm='blas',
                     native_smallm=False):
    """Reject optional kernels on older binaries instead of silently ignoring them."""
    import ctypes
    import os
    if smallm_kernel not in ('dot', 'wmma', 'wmma-register') or head_warps not in (None, 1, 4, 8, 16):
        raise ValueError('Unsupported native optimization setting')
    if prefill_gemm not in ('blas', 'wmma'):
        raise ValueError('Unsupported prefill GEMM setting')
    abi = None
    if smallm_kernel != 'dot' or head_warps is not None:
        try:
            version = ctypes.CDLL(str(binary)).quantlab_exl3_optimization_abi
        except AttributeError as error:
            raise ValueError('This optimization needs the newer WMMA/split-K experimental binary') from error
        version.argtypes = []
        version.restype = ctypes.c_int
        abi = version()
        if abi not in (1, 2) or (smallm_kernel == 'wmma-register' and abi < 2):
            raise ValueError('Unsupported native optimization ABI')
    hgemm_abi = None
    if prefill_gemm == 'wmma':
        try:
            version = ctypes.CDLL(str(binary)).quantlab_exl3_hgemm_abi
        except AttributeError as error:
            raise ValueError('WMMA prefill needs a binary with the prefill GEMM extension') from error
        version.argtypes = []
        version.restype = ctypes.c_int
        hgemm_abi = version()
        if hgemm_abi != 1:
            raise ValueError('Unsupported prefill GEMM ABI')
    codebooks = [0]
    if native_smallm:
        library = ctypes.CDLL(str(binary))
        try:
            capability = library.quantlab_exl3_smallm_codebooks
        except AttributeError:
            pass  # Older verified binaries implement only the default codebook.
        else:
            capability.argtypes = []
            capability.restype = ctypes.c_int
            mask = capability()
            if mask not in (1, 5):
                raise ValueError('Unsupported native small-M codebook capability')
            codebooks = [cb for cb in (0, 2) if mask & (1 << cb)]
    os.environ['EXL3_HGEMM_IMPL'] = prefill_gemm
    os.environ['EXL3_SMALLM_WMMA'] = {'dot':'0', 'wmma':'1', 'wmma-register':'2'}[smallm_kernel]
    if head_warps is None:
        os.environ.pop('EXL3_SMALLM_HEAD_WARPS', None)
    else:
        os.environ['EXL3_SMALLM_HEAD_WARPS'] = str(head_warps)
    return dict(smallm_kernel=smallm_kernel, head_warps=head_warps, optimization_abi=abi,
                prefill_gemm=prefill_gemm, prefill_gemm_abi=hgemm_abi,
                smallm_codebooks=codebooks)


def install_optimizations(model, draft, *, gpu_embedding=False, batch_greedy=False,
                          gpu_draft=False, gpu_draft_metadata=False):
    import torch
    from exllamav3 import Generator
    from exllamav3.modules.embedding import Embedding
    from exllamav3.architecture.qwen3_5_mtp import Qwen3_5MTPModel

    if gpu_draft and (not gpu_embedding or not isinstance(draft, Qwen3_5MTPModel)):
        raise ValueError('GPU drafting requires GPU embedding and Qwen3.5 integrated MTP')
    if gpu_draft_metadata and not gpu_draft:
        raise ValueError('GPU draft metadata requires GPU drafting')
    embedding_bytes = 0
    if gpu_embedding:
        embedding = model.modules[0]
        if type(embedding) is not Embedding or model.loaded_tp:
            raise ValueError('GPU embedding experiment requires a plain single-device embedding')
        devices = {str(torch.device(child.device)) for component in (model, draft)
                   if component is not None for child in component
                   if getattr(child, 'device', None) is not None
                   and torch.device(child.device).type == 'cuda'}
        if len(devices) != 1:
            raise ValueError('GPU embedding experiment requires exactly one GPU')
        device = model.modules[model.logit_layer_idx].device
        embedding.embedding.to(device)
        embedding.device = torch.device(device)
        original_forward = embedding.forward
        def forward(x, params, *args, **kwargs):
            if params.get('indexed_embeddings'):
                raise ValueError('GPU embedding experiment currently supports text inputs only')
            # The target's layer loader normally performs this step, but the
            # attached MTP input layer calls the shared embedding directly.
            x = embedding.prepare_for_device(x, params)
            return original_forward(x, params, *args, **kwargs)
        embedding.forward = forward
        embedding_bytes = embedding.embedding.weight.numel() * embedding.embedding.weight.element_size()
    # The runtime loads one model per process. These class defaults are read by
    # fresh request generators, including those created by the HTTP engine.
    Generator.quantlab_batch_greedy = bool(batch_greedy)
    Generator.quantlab_gpu_draft = bool(gpu_draft)
    Generator.quantlab_gpu_draft_metadata = bool(gpu_draft_metadata)
    return dict(gpu_embedding=gpu_embedding, embedding_bytes_on_gpu=embedding_bytes,
                batch_greedy=batch_greedy, gpu_draft=gpu_draft,
                gpu_draft_metadata=gpu_draft_metadata)


def can_batch_greedy(job, logits):
    """Only independent, unmodified greedy positions can share one readback."""
    from exllamav3.generator.sampler import ArgmaxSampler
    if type(job.sampler) is not ArgmaxSampler:
        return False
    from exllamav3.generator.sampler.custom import SS_Argmax, SS_Fused
    steps = job.sampler.steps
    pure = (len(steps) == 1 and (type(steps[0]) is SS_Argmax or
            (type(steps[0]) is SS_Fused and steps[0].mode == SS_Fused.MODE_GREEDY)))
    return ('forward' not in vars(job.sampler)
            and pure and not job.sampler.reqs_past_ids and not job.sampler.reqs_torch_seed
            and logits.shape[0] == 1 and logits.shape[1] > 1
            and job.new_tokens >= 0 and job.min_new_tokens == 0
            and not job.filters and not job.banned_strings
            and job.forced_ids is None and job.device_logit_mask is None
            and not job.return_probs and not job.return_top_tokens)


def observe_native_attention(model, draft):
    """Count actual native attention dispatches for the optional graph trial."""
    from exllamav3.modules.attn import Attention
    records = []
    for component in (model, draft):
        if component is None:
            continue
        for module in component:
            if not isinstance(module, Attention):
                continue
            row = dict(key=module.key, native_calls=0, fallback_calls=0)
            original = module.bc_attn_step
            def wrap(original, row):
                def step(*args, **kwargs):
                    result = original(*args, **kwargs)
                    row['fallback_calls' if result is None else 'native_calls'] += 1
                    return result
                return step
            module.bc_attn_step = wrap(original, row)
            records.append(row)
    return records
