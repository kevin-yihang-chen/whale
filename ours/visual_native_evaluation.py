"""Measure E1 through the same native visual agent used by E4 training.

The caller supplies an initialized AgentLoopManager and a fixed dataset/config.
This module does not launch a model, certify its weights, select candidates or
change the data role. Incomplete executions retain evidence without a score.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .evidence import PairPrediction, VisualPair, VisualPairEvaluator, fingerprint
from .visual_task import SYSTEM_PROMPT, binary_answer_verifier, file_sha256


def verifier_identity(protocol=None):
    sources = {name: file_sha256(Path(__file__).with_name(name))
        for name in ('visual_task.py', 'visual_evidence_reward.py')}
    if protocol is not None:
        from .chart_answer_protocol import PROTOCOL, identity
        if protocol != PROTOCOL:
            raise ValueError('Unknown verifier protocol')
        sources['chart_protocol'] = identity()
    return fingerprint(sources)


def pair_inputs(manifest_path):
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    role, partition = manifest.get('role'), manifest.get('partition')
    if (role not in {'engineering', 'C', 'V', 'T', 'H'} or
            (role == 'H') != (partition == 'H-pair')):
        raise ValueError('Paired evaluation requires an explicit audit data role')
    pairs = tuple(VisualPair(**row) for row in manifest['pairs'])
    if not pairs or len({p.pair_id for p in pairs}) != len(pairs):
        raise ValueError('Paired manifest must have unique nonempty coverage')
    if fingerprint([asdict(p) for p in pairs]) != manifest['audit_data_sha256']:
        raise ValueError('Pair manifest identity differs')
    if set(manifest['image_files']) != {sha for p in pairs for sha in p.images_sha256}:
        raise ValueError('Image coverage differs from the pair manifest')
    payloads = {}
    for sha, relative in manifest['image_files'].items():
        image = (path.parent / relative).resolve()
        if not image.is_relative_to(path.parent):
            raise ValueError('Image path leaves the manifest directory')
        payload = image.read_bytes()
        if hashlib.sha256(payload).hexdigest() != sha:
            raise ValueError('Image bytes differ from their manifest')
        payloads[sha] = payload
    rows = []
    protocol = manifest.get('answer_protocol')
    if protocol is not None:
        from .chart_answer_protocol import PROTOCOL, DATA_SOURCE, decode_truth
        if protocol != PROTOCOL:
            raise ValueError('Unknown chart data protocol')
    for pair in pairs:
        for side in (0, 1):
            truth = pair.answers[side]
            if protocol is None and truth not in {'A', 'B'}:
                raise ValueError('This shared visual verifier requires A/B labels')
            if protocol is not None:
                decode_truth(truth)
            rows.append({'data_source': DATA_SOURCE if protocol else 'visual_evidence_binary',
                'visual_sample_id': fingerprint({'pair_id': pair.pair_id, 'side': side}),
                'prompt': [{'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': '<image>\n' + pair.question}],
                'images': [{'bytes': payloads[pair.images_sha256[side]]}],
                'reward_model': {'style': 'rule', 'ground_truth': truth},
                'extra_info': {'index': len(rows)}})
    return manifest, pairs, rows


def write_pair_parquet(manifest_path, output):
    """Build the shared bytes-only dataset without putting labels in prompts."""
    import pandas as pd
    manifest, pairs, rows = pair_inputs(manifest_path)
    with Path(output).open('xb') as stream:
        pd.DataFrame(rows).to_parquet(stream, index=False)
    return {'role': manifest['role'], 'pairs': len(pairs), 'examples': len(rows),
        'manifest_sha256': file_sha256(Path(manifest_path)), 'parquet_sha256': file_sha256(Path(output))}


async def evaluate_pairs(manager, dataset, *, manifest_path, identity, output, batch_size):
    """Dispatch complete pairs through native batches, then construct an E1 receipt.

    Input coverage is checked before any generation. Native response order is
    matched using sample identity, never assumed to be task completion order.
    The caller must separately certify the serving weights and decoding config.
    """
    from verl import DataProto
    from verl.utils.dataset.rl_dataset import collate_fn
    from .visual_harness_dataset import VisualHarnessDataset

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('Evaluation batch size must be a positive integer')
    if not isinstance(dataset, VisualHarnessDataset):
        raise ValueError('Evaluation must use the shared executable visual dataset')
    workers = len(manager.agent_loop_workers)
    if not workers or any(min(batch_size, len(dataset) - offset) % workers for offset in range(0, len(dataset), batch_size)):
        raise ValueError('Each native batch must divide evenly across agent workers')
    manifest, pairs, reference = pair_inputs(manifest_path)
    protocol = manifest.get('answer_protocol')
    if protocol:
        from .chart_answer_protocol import verify as answer_verifier, SharedAnswerHarness
        if not isinstance(dataset.visual_harness, SharedAnswerHarness):
            raise ValueError('Shared chart data requires host-owned answer commitment')
    else:
        answer_verifier = binary_answer_verifier
    if identity.audit_data_sha256 != manifest['audit_data_sha256'] or identity.verifier_sha256 != verifier_identity(protocol):
        raise ValueError('Evaluation identity differs from actual audit data or verifier')
    expected = {row['visual_sample_id']: row for row in reference}
    incoming = list(dataset.dataframe)
    ids = [row.get('visual_sample_id') for row in incoming]
    if len(ids) != len(expected) or len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise ValueError('Dataset filtering or duplication changed complete pair coverage')
    for row in incoming:
        target = expected[row['visual_sample_id']]
        if any(row.get(key) != target[key] for key in ('prompt', 'images', 'reward_model', 'data_source')) or row.get('videos'):
            raise ValueError('Dataset image, question, label or domain differs from the manifest')
    harness = dataset.visual_harness
    harness.unchanged()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        with (output / name).open('x') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
    write('start.json', {'status': 'STARTED', 'identity': asdict(identity),
        'manifest_sha256': file_sha256(Path(manifest_path)), 'role': manifest['role'],
        'harness_sha256': harness.sha256, 'expected_examples': len(reference), 'batch_size': batch_size})
    records, artifacts = {}, {}
    try:
        for offset in range(0, len(dataset), batch_size):
            items = [dataset[i] for i in range(offset, min(offset + batch_size, len(dataset)))]
            batch = DataProto.from_single_dict(collate_fn(items))
            batch.meta_info.update(validate=True)
            result = await manager.generate_sequences(batch)
            archive = output / f'batch-{offset // batch_size:04d}.pkl'
            result.save_to_disk(archive)
            artifacts[archive.name] = file_sha256(archive)
            meta = result.non_tensor_batch
            actual_ids = meta['visual_sample_id'].tolist()
            if len(result) != len(items) or len(set(actual_ids)) != len(items) or set(actual_ids) != {i['visual_sample_id'] for i in items}:
                raise ValueError('Native output coverage differs from its input batch')
            for index, sample_id in enumerate(actual_ids):
                if sample_id in records or meta['visual_harness_sha256'][index] != harness.sha256:
                    raise ValueError('Repeated output or changed candidate identity')
                raw, committed = meta['visual_final_answer'][index], meta['visual_committed_answer'][index]
                if not isinstance(raw, str) or not isinstance(committed, str) or harness.invoke('parse_answer', text=raw) != committed:
                    raise ValueError('Recorded answer differs from the fixed candidate parser')
                reward = int(answer_verifier(committed, expected[sample_id]['reward_model']['ground_truth']))
                score = result.batch['rm_scores'][index]
                mask = result.batch['response_mask'][index]
                attention = result.batch['attention_mask'][index, -len(mask):]
                if not bool(((mask == 0) | (mask == 1)).all()) or bool((mask > attention).any()):
                    raise ValueError('Native generated-token mask is invalid')
                reward_positions = score.nonzero().flatten().tolist()
                final_position = int(attention.sum().item()) - 1
                if score.sum().item() != reward or reward_positions != ([final_position] if reward else []):
                    raise ValueError('Native reward differs from the shared verifier or final-token placement')
                calls = meta['visual_policy_calls'][index]
                generated = meta['visual_generated_tokens'][index]
                if type(calls) is not int or calls < 1:
                    raise ValueError('Missing actual generation-call count')
                if type(generated) is not int or generated < int(mask.sum().item()):
                    raise ValueError('Generation-token accounting is smaller than retained assistant tokens')
                records[sample_id] = {'sample_id': sample_id, 'raw_answer': raw, 'committed_answer': committed,
                    'correct': reward, 'policy_calls': calls, 'native_turns': int(meta['__num_turns__'][index]),
                    'generated_tokens': generated, 'retained_assistant_tokens': int(mask.sum().item()),
                    'assistant_and_observation_tokens': int(attention.sum().item()),
                    'tool_trace': meta['visual_harness_tool_trace'][index],
                    'nudges': int(meta['visual_harness_nudges'][index])}
            write(f'batch-{offset // batch_size:04d}.json', [records[k] for k in actual_ids])
        harness.unchanged()
        predictions = [PairPrediction(p.pair_id, tuple(records[fingerprint({'pair_id': p.pair_id, 'side': side})]['committed_answer']
            for side in (0, 1))) for p in pairs]
        receipt = VisualPairEvaluator(answer_verifier).evaluate(pairs, predictions,
            identity=identity, harness_sha256=harness.sha256, role=manifest['role'])
        ordered = [records[row['visual_sample_id']] for row in reference]
        report = {'status': 'COMPLETE_NATIVE_PAIR_EVALUATION', 'audit': asdict(receipt),
            'paired_accuracy': receipt.paired_accuracy, 'marginal_accuracy': receipt.marginal_accuracy,
            'examples': len(ordered), 'policy_calls': sum(r['policy_calls'] for r in ordered),
            'mean_native_turns': sum(r['native_turns'] for r in ordered) / len(ordered),
            'generated_tokens': sum(r['generated_tokens'] for r in ordered), 'records': ordered,
            'batch_artifact_sha256': artifacts, 'serving_weights_verified_by_this_module': False,
            'decoding_verified_by_this_module': False,
            'limitations': ['Data/answer/mask checks do not certify serving weights, decoding, model capability or VETO gains.',
                'Native turn count includes user/tool messages; actual generation calls are reported separately.']}
        write('result.json', report)
        return receipt, report
    except BaseException as error:
        write('failure.json', {'status': 'INCOMPLETE_NO_AGGREGATE_SCORE', 'completed_examples': len(records),
            'error_type': type(error).__name__, 'error': str(error), 'batch_artifact_sha256': artifacts})
        raise
