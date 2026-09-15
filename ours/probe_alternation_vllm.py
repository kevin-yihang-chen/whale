"""Verify the next E4 checkpoint inside a fresh vLLM worker with one short request."""
import argparse
import json
import os
from pathlib import Path
import time

from .alternation_training import write_new
from .audit_native_training_batch import require
from .probe_updated_vllm import check, managed_inference_engine
from .visual_task import file_sha256


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    plan, contract, coordinates = check(args.plan)
    transition = json.loads(Path(plan['transition']).read_text())
    require(transition['kind'] == 'native_alternation_checkpoint_transition', 'Expected next-phase parameter audit')
    if args.check_only:
        print(json.dumps({'status': 'PREFLIGHT_PASS', 'contract': contract, 'coordinates': coordinates}))
    else:
        require(bool(os.environ.get('SLURM_JOB_ID')), 'Expected Slurm allocation')
        from vllm import SamplingParams
        job, started = os.environ['SLURM_JOB_ID'], time.monotonic()
        with managed_inference_engine(plan) as llm:
            sampled = llm.collective_rpc('sample_updated_embeddings', timeout=60.,
                kwargs={'coordinates': [{k: c[k] for k in ('token_id', 'column')} for c in coordinates]})
            require(len(sampled) == 1 and sampled[0]['model_class'] == 'Qwen3_5ForConditionalGeneration', 'Wrong worker model')
            require(sampled[0]['values'] == [c['updated'] for c in coordinates] and
                    all(c['base'] != c['updated'] for c in coordinates), 'Actual worker parameters did not change correctly')
            write_new(Path(f'results/alternation-vllm-request-{job}.json'),
                {'kind': 'native_alternation_request_start', 'job_id': job,
                 'plan_sha256': file_sha256(args.plan), 'prompt_token_ids': contract['prompt_token_ids'],
                 'actual_worker_samples': sampled, 'changed_coordinates': coordinates, 'weights_verified': True,
                 'calls_attempted': 1, 'max_completion_tokens': 16, 'returned_tokens': None})
            outputs = llm.generate([{'prompt_token_ids': contract['prompt_token_ids']}],
                                   SamplingParams(**plan['sampling']), use_tqdm=False)
            require(len(outputs) == 1 and len(outputs[0].outputs) == 1, 'Unexpected response count')
            response = outputs[0].outputs[0]
            require(0 < len(response.token_ids) <= 16, 'Empty or over-budget response')
            result = {'kind': 'native_alternation_vllm_handoff', 'status': 'PASS', 'job_id': job,
                'plan_sha256': file_sha256(args.plan), 'transition_sha256': file_sha256(Path(plan['transition'])),
                'contract': contract, 'changed_coordinates': coordinates, 'actual_worker_samples': sampled,
                'new_model_calls': 1, 'generated_tokens': len(response.token_ids),
                'token_ids': list(response.token_ids), 'content': response.text, 'finish_reason': response.finish_reason,
                'measurement_seconds': time.monotonic() - started,
                'limitations': ['One engineering request, not heldout evaluation or evidence of task improvement.',
                    'Eight distinguishing embedding values verified; full served parameter equality is not measured.',
                    'Fresh exported-checkpoint serving, not a numeric fingerprint of the original NCCL receiver.',
                    'Incoming and exported active parameters are BF16; native FP32 changes and surviving export changes are audited separately.']}
            write_new(Path(f'results/alternation-vllm-handoff-{job}.json'), result)
            print(json.dumps(result), flush=True)
