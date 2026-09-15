"""Replay E1 answer extraction on stored outputs; no new model evaluation.

This post-hoc diagnostic asks whether a reported gain depends on the generated
harness's parser. It is not a prompt ablation or a new performance measurement.
"""
import argparse
import json
from pathlib import Path

from .isolated_visual_harness import load_isolated_visual_harness
from .visual_dataset_materialization import single_rows
from .visual_native_evaluation import pair_inputs
from .visual_task import binary_answer_verifier, file_sha256


def replay(plan_path, partition, incoming_path, output):
    plan = json.loads(plan_path.read_text())
    assert plan['kind'] == 'visual_search_candidate_evaluation'
    role = partition.split('-')[0]
    data = plan['partitions'][role]
    assert file_sha256(Path(data['manifest'])) == data['manifest_sha256']
    rows = single_rows(data['manifest'])[1] if role == 'H' else pair_inputs(data['manifest'])[2]
    truth = {r['visual_sample_id']: r['reward_model']['ground_truth'] for r in rows}
    result_path = Path(plan['output']) / partition / 'result.json'
    result = json.loads(result_path.read_text())
    actual = {r['sample_id']: r for r in result['records']}
    assert len(actual) == len(result['records']) == len(truth) == 128 and set(actual) == set(truth)
    for name, digest in result['batch_artifact_sha256'].items():
        assert file_sha256(result_path.parent / name) == digest
    candidate = load_isolated_visual_harness(Path(plan['config']['data']['visual_harness_path']))
    assert candidate.sha256 == plan['harness_sha256']
    incoming = load_isolated_visual_harness(incoming_path)
    outcomes = []
    for sample, row in actual.items():
        assert candidate.invoke('parse_answer', text=row['raw_answer']) == row['committed_answer']
        original_score = int(binary_answer_verifier(row['committed_answer'], truth[sample]))
        assert original_score == row['correct']
        answer = incoming.invoke('parse_answer', text=row['raw_answer'])
        outcomes.append({'sample_id': sample, 'candidate_answer': row['committed_answer'],
            'incoming_parser_answer': answer, 'candidate_correct': original_score,
            'incoming_parser_correct': int(binary_answer_verifier(answer, truth[sample]))})
    report = {'kind': 'stored_answer_parser_replay', 'status': 'COMPLETE_PARSER_REPLAY',
        'partition': partition, 'role': role, 'examples': 128,
        'plan_sha256': file_sha256(plan_path), 'result_sha256': file_sha256(result_path),
        'candidate_harness_sha256': candidate.sha256, 'incoming_harness_sha256': incoming.sha256,
        'source_sha256': file_sha256(Path(__file__)),
        'candidate_correct': sum(r['candidate_correct'] for r in outcomes),
        'incoming_parser_correct': sum(r['incoming_parser_correct'] for r in outcomes),
        'changed_committed_answers': sum(r['candidate_answer'] != r['incoming_parser_answer'] for r in outcomes),
        'records': outcomes, 'new_model_calls': 0, 'new_optimizer_steps': 0,
        'limitations': ['Post-hoc answer-parser replay on the same saved model outputs.',
                       'Not a prompt ablation, independent evaluation or evidence of VETO efficacy.']}
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--partition', choices=('H', 'C', 'H-repeat', 'C-repeat'), required=True)
    parser.add_argument('--incoming', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    replay(args.plan.resolve(), args.partition, args.incoming.resolve(), args.output.resolve())
