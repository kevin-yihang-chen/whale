"""Bounded E4 optimizer validation using synthetic tensors, never model training.

CPU mode compares native ZeRO-1 AdamW foreach and scalar updates. GPU mode
allocates the reviewed actor parameter shapes, text-only gradients and the
native transport buffer size, then checks one scalar update. No forward pass,
sample, actual checkpoint weight or model-quality result is produced.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    import torch
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from .training_bootstrap import prepare_worker

    prepare_worker()
    from verl.workers.disagg_fsdp_workers import DisaggregatedAsyncActorWorker

    torch.set_num_threads(1)
    configuration = OmegaConf.load(args.config)
    optimizer_config = configuration.actor_rollout_ref.actor.optim
    graph = json.loads(args.graph.read_text())
    if graph['model_class'] != 'Qwen3_5ForConditionalGeneration' or graph['shape_mismatches']:
        raise ValueError('Requires the reviewed Qwen3.5 graph')
    device = 'cuda' if args.mode == 'gpu' else 'cpu'
    if device == 'cuda':
        if not os.environ.get('SLURM_JOB_ID') or torch.cuda.device_count() != 1:
            raise ValueError('GPU probe requires a Slurm allocation exposing exactly one GPU')
        torch.cuda.set_device(0)
    report = {'kind': 'synthetic_native_optimizer_probe', 'mode': args.mode,
              'status': 'RUNNING', 'job_id': os.environ.get('SLURM_JOB_ID'),
              'source_sha256': digest(__file__), 'config_sha256': digest(args.config),
              'graph_sha256': digest(args.graph), 'new_model_calls': 0,
              'actual_checkpoint_updates': 0, 'synthetic_optimizer_steps': 0,
              'limitations': ['Synthetic optimizer tensors only; no model forward/backward or training result.',
                              'GPU probe does not reproduce live FSDP activations or all runtime allocations.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)

    def build(module, foreach):
        options = OmegaConf.create(OmegaConf.to_container(optimizer_config, resolve=True))
        options.override_optimizer_config = {'foreach': foreach}
        optimizer, scheduler = DisaggregatedAsyncActorWorker._build_zero1_optimizer_and_scheduler(
            SimpleNamespace(rank=0), module, options)
        if optimizer.optim.defaults['foreach'] is not foreach:
            raise ValueError('Native optimizer did not retain the explicit foreach setting')
        return optimizer

    try:
        with tempfile.TemporaryDirectory(prefix='whale-optimizer-') as temp:
            dist.init_process_group('nccl' if device == 'cuda' else 'gloo', rank=0, world_size=1,
                                    init_method='file://' + str(Path(temp) / 'rendezvous'))
            if device == 'cpu':
                modules = [torch.nn.ParameterList([
                    torch.nn.Parameter(torch.full(shape, .5))
                    for shape in [(64, 128), (128,), (256, 32)]]) for _ in range(2)]
                optimizers = [build(module, flag) for module, flag in zip(modules, [True, False])]
                for step in range(3):
                    for module, optimizer in zip(modules, optimizers):
                        for parameter in module:
                            parameter.grad = torch.full_like(parameter, .01 * (step + 1))
                        optimizer.step()
                    report['synthetic_optimizer_steps'] += 2
                maximum = 0.
                for a, b in zip(*modules):
                    torch.testing.assert_close(a, b, rtol=0, atol=1e-7)
                    maximum = max(maximum, float((a - b).detach().abs().max()))
                report['max_abs_foreach_vs_scalar_delta'] = maximum
            else:
                import cupy
                shapes = {name: shape for name, shape in graph['active_shapes'].items()
                          if name != 'lm_head.weight'}
                parameters = torch.nn.ParameterList([
                    torch.nn.Parameter(torch.full(shape, .5, device=device, dtype=torch.float32))
                    for shape in shapes.values()])
                optimizer = build(parameters, False)
                buffer_mib = int(configuration.actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes)
                buffers = [cupy.empty(buffer_mib * 1024**2, dtype=cupy.uint8) for _ in range(2)]
                for buffer in buffers:
                    buffer.fill(0)
                gradient_elements = 0
                for name, parameter in zip(shapes, parameters):
                    if not name.startswith('model.visual.'):
                        parameter.grad = torch.full_like(parameter, .01)
                        gradient_elements += parameter.numel()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                report['memory_before_step'] = dict(zip(['free_bytes', 'total_bytes'], torch.cuda.mem_get_info()))
                optimizer.step()
                torch.cuda.synchronize()
                report['synthetic_optimizer_steps'] = 1
                checks = []
                for name, parameter in zip(shapes, parameters):
                    if parameter.grad is not None:
                        sample = parameter.detach().reshape(-1)[0].item()
                        if not .49 < sample < .5:
                            raise ValueError(f'Synthetic update missing or invalid for {name}')
                        checks.append(sample)
                report.update(parameter_tensors=len(shapes), parameter_elements=sum(p.numel() for p in parameters),
                              gradient_elements=gradient_elements, transport_buffer_mib=2 * buffer_mib,
                              updated_tensor_samples=len(checks), peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                              memory_after_step=dict(zip(['free_bytes', 'total_bytes'], torch.cuda.mem_get_info())))
            report['status'] = 'PASS'
    except Exception as exc:
        report.update(status='FAILED', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        with args.output.open('x') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
        if dist.is_initialized():
            dist.destroy_process_group()
        print(json.dumps(report))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['cpu', 'gpu'], required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
