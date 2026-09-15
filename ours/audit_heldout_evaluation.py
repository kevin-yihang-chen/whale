"""Full heldout reply/token and independent board audit; no selection feedback."""
from collections import Counter, defaultdict, deque
from dataclasses import asdict
import json
from pathlib import Path

from .audit_mh_phase import board_review, canonical, read_lines
from .audit_native_training_batch import require
from .local_completion import CompletionResponse
from .visual_task import file_sha256


def audit_evaluation(plan, plan_path, private):
    from autoharness_chess_puzzle import runner
    from autoharness_chess_puzzle.harness import load_harness
    from transformers import AutoTokenizer
    from .controlled_heldout import validate_request
    request = json.loads((private / 'request.json').read_text())
    validate_request(plan,plan_path,private,request)
    require(request['plan_sha256'] == file_sha256(plan_path), 'Different evaluation plan')
    harness = Path(request['kwargs']['harness_path'])
    require(file_sha256(harness) == request['harness_sha256'], 'Evaluated harness changed')
    tokenizer = AutoTokenizer.from_pretrained(plan['target_config']['model'], local_files_only=True)
    examples = {e.example_id: e for e in runner._read_examples(plan['dataset'], limit=64, seed=plan['seed'])}
    all_outputs, records = {}, []
    total_calls = total_tokens = length_stops = missing_thinking_end = 0
    for replica in range(4):
        directory = private / f'replica-{replica}'
        receipt = json.loads((directory / 'result.json').read_text())
        require(receipt['status'] == 'COMPLETED' and receipt['replica'] == replica and
                receipt['plan_sha256'] == file_sha256(plan_path) and receipt['harness_sha256'] == request['harness_sha256'],
                'Incomplete or mixed evaluation shard')
        for name, digest in receipt['artifact_sha256'].items():
            require(file_sha256(directory / name) == digest, f'Changed shard artifact: {name}')
        proof = json.loads((directory / 'worker-weights.json').read_text())
        require(proof['status'] == 'PASS' and proof['plan_sha256'] == file_sha256(plan_path) and
                proof['actual_worker']['values'] == [c['updated'] for c in plan['worker_probe_coordinates']],
                'Actual worker weight proof differs')
        randomization=json.loads((directory/'process-randomization.json').read_text())
        require(all(randomization[k]==plan['seed'] for k in ('trial_seed','data_sampler_seed','process_rng_seed',
                'training_vllm_engine_seed','mh_request_seed','torch_cpu_initial_seed')) and
                randomization['bitwise_gpu_reproducibility_claimed'] is False,'Wrong heldout process randomization')
        require(receipt['data_role']=='test' and receipt['condition']==plan['condition'] and receipt['seed']==plan['seed'],
                'Mixed heldout shard identity')
        ids = json.loads((directory / 'ordered-ids.json').read_text())
        outputs = json.loads((directory / 'native-outputs.json').read_text())
        require(ids == [key for key in examples if key in plan['shard_ids'][replica]] and
                len(ids) == len(outputs) == 16 and not set(ids) & set(all_outputs), 'Wrong shard coverage')
        all_outputs.update(zip(ids, outputs, strict=True))
        starts, ends = {}, {}
        for event in read_lines(directory / 'request-events.jsonl'):
            require(event['event'] in {'start', 'complete'}, 'Unsuccessful request event')
            target = starts if event['event'] == 'start' else ends
            require(event['id'] not in target, 'Repeated request event')
            target[event['id']] = event
        calls = read_lines(directory / 'generations.jsonl')
        require(set(starts) == set(ends) and len(starts) == len(calls), 'Incomplete request accounting')
        outer = Counter(canonical([e['messages'], e['max_tokens'], ends[key]['usage']]) for key,e in starts.items())
        inner = Counter(canonical([c['request']['messages'], c['request']['max_tokens'], c['raw_response']['usage']]) for c in calls)
        require(outer == inner and [c['call'] for c in calls] == list(range(1,len(calls)+1)), 'Request journals differ')
        available = defaultdict(deque)
        for call in calls:
            payload, response = call['request'], call['raw_response']
            require(call['weights_sha256'] == plan['target_config']['expected_weights_sha256'], 'Wrong call checkpoint')
            for key in ('model','temperature','top_p','top_k','seed','chat_template_kwargs'):
                require(payload[key] == plan['target_config'][key], f'Wrong sampling option: {key}')
            require(response['model'] == payload['model'] and len(response['choices']) == 1, 'Wrong response identity')
            choice, = response['choices']; usage = response['usage']; content = choice['message']['content']
            require(0 < len(choice['token_ids']) == usage['completion_tokens'] <= payload['max_tokens'] <= 8129,
                    'Completion-token evidence differs')
            require(len(response['prompt_token_ids']) == usage['prompt_tokens'] and
                    usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens'], 'Usage totals differ')
            require(tokenizer.decode(choice['token_ids'], skip_special_tokens=True) == content, 'Generated token decoding differs')
            require(tokenizer.apply_chat_template(payload['messages'], tokenize=True, return_dict=False,
                    add_generation_prompt=True, **payload['chat_template_kwargs']) == response['prompt_token_ids'],
                    'Prompt-token rendering differs')
            available[canonical([payload['messages'],payload['max_tokens'],content])].append(call)
            total_calls += 1; total_tokens += usage['completion_tokens']
            length_stops += choice['finish_reason'] == 'length'
            missing_thinking_end += 248069 not in choice['token_ids']
        loaded = load_harness(harness)
        class ReplayClient:
            def __init__(self, expected):
                self.expected = deque(e for e in expected['harness_trace'] if e.get('actor') == 'assistant')
            def complete_response(self, messages, *, max_tokens=None):
                require(bool(self.expected), 'Extra replay request')
                event = self.expected.popleft()
                key = canonical([[asdict(m) for m in messages],max_tokens,event['raw_response']])
                require(bool(available[key]), 'Replay request not found in actual responses')
                call = available[key].popleft()
                return CompletionResponse(event['raw_response'],call['raw_response']['usage'])
        for key, output in zip(ids, outputs, strict=True):
            client = ReplayClient(output)
            replayed = runner.run_puzzle_rollout(harness=loaded,example=examples[key],llm=client,
                                                assistant_token_budget=8129,policy_max_tokens=8129)
            require(not client.expected and replayed == output, 'Native replay differs')
            board = board_review(examples[key],output)
            require(0 < board['assistant_tokens'] <= 8129, 'Trajectory exceeds token budget')
            records.append({'puzzle_id':key,'replica':replica,**board})
        require(not any(available.values()), 'Unmatched actual responses')
    require(set(all_outputs) == set(examples) and len(records) == 64, 'Incomplete global coverage')
    summary = json.loads((private / 'merged/val.json').read_text())
    combined = [all_outputs[key] for key in examples]
    require(runner.summarize_outputs(combined, summary['metadata']) == summary, 'Merged summary differs')
    require(summary['solved_examples'] == sum(r['solved'] for r in records) and
            total_tokens == sum(r['assistant_tokens'] for r in records), 'Independent totals differ')
    require(total_calls<=64*18 and total_tokens<=64*8129,'Heldout execution exceeded its frozen budget')
    return {'kind':'controlled_heldout_evaluation_audit','condition':plan['condition'],'data_role':'test','status':'PASS','plan_sha256':file_sha256(plan_path),
        'harness_sha256':request['harness_sha256'],'seed':plan['seed'],'examples':64,
        'solved':summary['solved_examples'],'calls':total_calls,'generated_tokens':total_tokens,
        'length_stop_calls':length_stops,'calls_without_thinking_end':missing_thinking_end,'records':records,
        'audit_source_sha256':file_sha256(Path(__file__)),'new_model_calls':0,
        'effective_retry_budgets':list(runner.retry_budgets(loaded)),'effective_max_turns':runner.effective_max_turns(loaded),
        'limitations':['Complete heldout inference audit; a comparative gain requires all declared trial results.',
            'Eight distinguishing embedding coordinates per live worker, not all served parameters.',
            'A native success may be parsed from truncated thinking.']}
