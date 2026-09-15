"""Check actual new weights inside vLLM and execute one bounded E4 request.

This is an engineering handoff check, not a held-out accuracy evaluation. Eight
embedding coordinates known to differ from base must match the audited export.
"""
import argparse
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import time

from .audit_native_training_batch import require
from .local_completion import checkpoint_manifest
from .visual_task import file_sha256


@contextmanager
def managed_inference_engine(plan):
    """Release vLLM workers explicitly when the bounded measurement ends."""
    from vllm import LLM
    llm = LLM(model=plan['model'], **plan['engine_options'])
    try:
        yield llm
    finally:
        llm.llm_engine.engine_core.shutdown(timeout=10.)


def inference_contract(base, updated):
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig
    configs = [AutoConfig.from_pretrained(p, local_files_only=True) for p in (base, updated)]
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()
                    if k not in {'dtype', '_name_or_path', 'transformers_version', '_from_model_config'}}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value
    require(clean(configs[0].to_dict()) == clean(configs[1].to_dict()), 'Inference model configuration differs')
    generations = [GenerationConfig.from_pretrained(p, local_files_only=True)
                   if (p / 'generation_config.json').exists() else GenerationConfig.from_model_config(c)
                   for p, c in zip((base, updated), configs)]
    require(clean(generations[0].to_dict()) == clean(generations[1].to_dict()), 'Generation configuration differs')
    tokenizers = [AutoTokenizer.from_pretrained(p, local_files_only=True) for p in (base, updated)]
    require(tokenizers[0].get_vocab() == tokenizers[1].get_vocab(), 'Tokenizer vocabulary differs')
    for field in ('chat_template', 'eos_token_id', 'pad_token_id', 'bos_token_id', 'all_special_ids'):
        require(getattr(tokenizers[0], field) == getattr(tokenizers[1], field), f'Tokenizer {field} differs')
    messages = [{'role': 'user', 'content': 'Reply with the single word ready.'}]
    ids = [t.apply_chat_template(messages, tokenize=True, return_dict=False,
                                 add_generation_prompt=True, enable_thinking=False)
           for t in tokenizers]
    require(ids[0] == ids[1], 'Rendered engineering prompt tokens differ')
    require(isinstance(ids[0], list) and ids[0] and all(type(x) is int for x in ids[0]),
            'Expected a nonempty flat token ID list')
    return {'model_config_equal_except_precision_metadata': True, 'generation_config_equal': True,
            'tokenizer_vocabulary_and_control_ids_equal': True, 'prompt_token_ids': ids[0],
            'messages': messages, 'enable_thinking': False, 'generation_eos_token_id': generations[0].eos_token_id}


def changed_embedding_coordinates(base, updated, count=8):
    import torch
    from safetensors import safe_open
    key = 'model.language_model.embed_tokens.weight'
    with ExitStack() as stack:
        tensors = []
        for root in (base, updated):
            matching = []
            for path in root.glob('*.safetensors'):
                reader = stack.enter_context(safe_open(str(path), framework='pt', device='cpu'))
                if key in reader.keys():
                    matching.append(reader.get_tensor(key))
            require(len(matching) == 1, 'Embedding missing or duplicated')
            tensors.append(matching[0])
        a, b = tensors
        require(a.shape == b.shape and a.dtype == b.dtype == torch.bfloat16, 'Unexpected embedding layout')
        found = []
        for row in range(0, a.shape[0], 128):
            changed = torch.nonzero(a[row:row + 128] != b[row:row + 128])
            for local_row, column in changed[:count - len(found)].tolist():
                i = row + local_row
                found.append({'token_id': i, 'column': column, 'base': float(a[i, column]), 'updated': float(b[i, column])})
            if len(found) == count:
                return found
        raise ValueError('Insufficient changed embedding coordinates for handoff proof')


def worker_embedding_samples(model, coordinates):
    import torch
    device = next(model.parameters()).device
    ids = torch.tensor([p['token_id'] for p in coordinates], dtype=torch.long, device=device)
    with torch.inference_mode():
        embeddings = model.embed_input_ids(ids)
        values = [float(embeddings[i, p['column']].float().cpu()) for i, p in enumerate(coordinates)]
    return {'model_class': type(model).__name__, 'embedding_dtype': str(embeddings.dtype), 'values': values}


def check(plan_path):
    from importlib.metadata import version
    plan = json.loads(plan_path.read_text())
    require(plan['kind'] == 'updated_vllm_handoff_plan' and plan['max_new_model_calls'] == 1, 'Wrong handoff plan')
    require(plan['sampling']['max_tokens'] == 16 and plan['sampling'].get('n', 1) == 1, 'Unexpected generation budget')
    require({k: version(k) for k in plan['versions']} == plan['versions'], 'Runtime version changed')
    for name, digest in plan['source_sha256'].items():
        require(file_sha256(Path(name)) == digest, f'Frozen handoff source changed: {name}')
    transition = json.loads(Path(plan['transition']).read_text())
    require(transition['native_fp32']['status'] == transition['exported_bf16']['status'] == 'CHANGED', 'No audited update')
    require(transition['export_exact_native_bf16_cast'], 'Export did not pass full cast audit')
    base, updated = Path(plan['base']), Path(plan['model'])
    for root, name in ((base, 'base'), (updated, 'exported')):
        require(checkpoint_manifest(root) == transition[name], f'{name} checkpoint differs from audit')
    contract = inference_contract(base, updated)
    coordinates = changed_embedding_coordinates(base, updated)
    return plan, contract, coordinates


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    plan, contract, coordinates = check(args.plan)
    if args.check_only:
        print(json.dumps({'status': 'PREFLIGHT_PASS', 'contract': contract, 'coordinates': coordinates}))
    else:
        require(bool(os.environ.get('SLURM_JOB_ID')), 'Expected Slurm allocation')
        from vllm import SamplingParams
        started = time.monotonic()
        with managed_inference_engine(plan) as llm:
            sampled = llm.collective_rpc('sample_updated_embeddings', timeout=60.,
                                        kwargs={'coordinates': [{k: p[k] for k in ('token_id', 'column')}
                                                                for p in coordinates]})
            require(len(sampled) == 1 and sampled[0]['model_class'] == 'Qwen3_5ForConditionalGeneration', 'Unexpected worker model')
            require(sampled[0]['values'] == [p['updated'] for p in coordinates], 'vLLM parameters differ from audited update')
            require(all(p['base'] != p['updated'] for p in coordinates), 'Probe coordinates do not distinguish base weights')
            request_path = Path(f"results/updated-vllm-request-{os.environ['SLURM_JOB_ID']}.json")
            with request_path.open('x') as f:
                json.dump({'kind': 'updated_vllm_request_start', 'job_id': os.environ['SLURM_JOB_ID'],
                           'plan_sha256': file_sha256(args.plan), 'prompt_token_ids': contract['prompt_token_ids'],
                           'max_completion_tokens': 16, 'calls_attempted': 1,
                           'returned_tokens': None, 'note': 'Consumption remains unknown until a completed result exists.'}, f, indent=2)
                f.write('\n')
                f.flush()
                os.fsync(f.fileno())
            output = llm.generate([{'prompt_token_ids': contract['prompt_token_ids']}],
                                  SamplingParams(**plan['sampling']), use_tqdm=False)
            require(len(output) == 1 and len(output[0].outputs) == 1, 'Unexpected output count')
            response = output[0].outputs[0]
            require(0 < len(response.token_ids) <= plan['sampling']['max_tokens'], 'Empty or over-budget response')
            result = {'kind': 'updated_vllm_handoff', 'status': 'PASS', 'job_id': os.environ['SLURM_JOB_ID'],
                      'plan_sha256': file_sha256(args.plan), 'transition_sha256': file_sha256(Path(plan['transition'])),
                      'contract': contract, 'changed_coordinates': coordinates, 'actual_worker_samples': sampled,
                      'new_model_calls': 1, 'generated_tokens': len(response.token_ids), 'token_ids': list(response.token_ids),
                      'content': response.text, 'finish_reason': response.finish_reason,
                      'seconds': time.monotonic() - started,
                      'limitations': ['Engineering request only; no accuracy or performance improvement claim.',
                                      'This loads the exported checkpoint into a fresh vLLM engine; it does not fingerprint the prior in-place NCCL receiver.',
                                      'The original native BF16 merger rounds 3840 originally FP32 elements. Matched serialization or a cast-only control is required for performance comparisons.']}
            with Path(f"results/updated-vllm-handoff-{result['job_id']}.json").open('x') as f:
                json.dump(result, f, indent=2)
                f.write('\n')
            print(json.dumps(result), flush=True)
