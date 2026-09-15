"""Observe real visual-backbone inputs and loss gradients in E4 workers."""
import hashlib
import json
import os
from pathlib import Path
import uuid


def tensor_identity(tensor):
    import torch
    value = tensor.detach().contiguous().cpu()
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()}


def observed_forward(self, *args, **kwargs):
    import torch
    from .qwen35_visual_fused import forward_with_visual_evidence
    pixels, grid = kwargs.get('pixel_values'), kwargs.get('image_grid_thw')
    if pixels is None or grid is None or kwargs.get('pixel_values_videos') is not None:
        raise ValueError('This frozen image-training diagnostic requires image pixels and no video')
    event = uuid.uuid4().hex
    root = Path(os.environ['VETO_VISUAL_FORWARD_AUDIT'])
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'visual-backbone-{os.getpid()}.jsonl'

    def record(kind, **values):
        with path.open('a') as stream:
            stream.write(json.dumps({'event': event, 'kind': kind,
                'job_id': os.environ.get('SLURM_JOB_ID'), **values}, allow_nan=False) + '\n')

    received = []
    def before(module, positional, named):
        actual = positional[0] if positional else named['hidden_states']
        actual_grid = named.get('grid_thw', positional[1] if len(positional) > 1 else None)
        if not torch.equal(actual, pixels.to(actual.dtype)) or not torch.equal(actual_grid, grid):
            raise ValueError('Visual backbone pixels/grid differ from the actor forward input')
        received.append(True)
        record('visual_backbone_input', actor_pixels=tensor_identity(pixels),
               consumed_pixels=tensor_identity(actual), grid=tensor_identity(grid),
               exact_after_native_dtype_cast=True, grad_enabled=torch.is_grad_enabled())

    def after(module, positional, output):
        features = output.pooler_output
        if not features.requires_grad:
            raise ValueError('Visual features are detached from the training graph')
        def gradient(value):
            finite = bool(torch.isfinite(value).all())
            if not finite:
                raise ValueError('Nonfinite gradient at the visual evidence interface')
            record('visual_feature_gradient', finite=finite,
                   nonzero_elements=int(torch.count_nonzero(value)),
                   l1=float(value.float().abs().sum()), shape=list(value.shape))
            return value
        features.register_hook(gradient)

    pre = self.model.visual.register_forward_pre_hook(before, with_kwargs=True)
    post = self.model.visual.register_forward_hook(after)
    try:
        result = forward_with_visual_evidence(self, *args, **kwargs)
        if len(received) != 1:
            raise ValueError('Expected exactly one observed image-backbone forward')
        return result
    finally:
        pre.remove()
        post.remove()


def prepare_worker():
    from .visual_training_bootstrap import prepare_worker as original
    from .qwen35_visual_fused import install_visual_fused_backend
    result = original()
    install_visual_fused_backend()
    import verl.models.transformers.monkey_patch as native
    dispatcher = native.patch_forward_with_backends
    if not getattr(dispatcher, '_visual_backbone_observed', False):
        def observed_dispatch(model, use_fused_kernels=False, fused_kernels_backend=None):
            dispatcher(model, use_fused_kernels, fused_kernels_backend)
            if model.config.model_type == 'qwen3_5' and use_fused_kernels:
                model.__class__.forward = observed_forward
        observed_dispatch._visual_backbone_observed = True
        observed_dispatch._visual_evidence_forward = True
        native.patch_forward_with_backends = observed_dispatch
    return {**result, 'qwen35_visual_fused_repair': True, 'actual_visual_forward_and_gradient_observer': True}
