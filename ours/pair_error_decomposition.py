"""Audit the information in E1 relative to same-pair marginal accuracy.

For fractions q0/q1/q2 of pairs with zero/one/two correct answers, E1 is
P=q2=2M-1+q0. Equal q0 on an archive makes epsilon=0 paired and marginal
non-regression gates equivalent on that archive, not on arbitrary candidates.
"""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path

from .visual_task import file_sha256


def decompose(correctness):
    assert correctness and all(len(row) == 2 and all(type(x) in (int, bool) and x in (0, 1)
                                                      for x in row) for row in correctness)
    n = len(correctness)
    counts = Counter(sum(row) for row in correctness)
    marginal = Fraction(counts[1] + 2 * counts[2], 2 * n)
    paired, zero = Fraction(counts[2], n), Fraction(counts[0], n)
    assert paired == 2 * marginal - 1 + zero
    return {'pairs': n, 'both_wrong': counts[0], 'one_correct': counts[1], 'both_correct': counts[2],
            'marginal_accuracy': float(marginal), 'paired_accuracy': float(paired),
            'both_wrong_fraction': str(zero),
            'side0_accuracy': sum(row[0] for row in correctness) / n,
            'side1_accuracy': sum(row[1] for row in correctness) / n,
            'identity_verified': True}


def analyze(spec_path, output):
    spec = json.loads(spec_path.read_text())
    assert spec['kind'] == 'observed_pair_error_decomposition'
    cases, groups = {}, {}
    for name, item in spec['cases'].items():
        path = Path(item['result'])
        assert file_sha256(path) == item['result_sha256']
        result = json.loads(path.read_text())
        assert result['status'].startswith('COMPLETE')
        audit = result['audit']
        assert audit['role'] == item['role'] and len(set(audit['pair_ids'])) == len(audit['correctness'])
        value = decompose(audit['correctness'])
        assert value['marginal_accuracy'] == result['marginal_accuracy']
        assert value['paired_accuracy'] == result['paired_accuracy']
        value.update(role=audit['role'], weights_sha256=audit['identity']['weights_sha256'],
                     harness_sha256=audit['harness_sha256'], result_sha256=item['result_sha256'])
        cases[name] = value
        group = audit['role'] + ':' + audit['identity']['audit_data_sha256']
        groups.setdefault(group, []).append(name)
    report = {'status': 'COMPLETE_OBSERVED_PAIR_ERROR_DECOMPOSITION',
        'spec_sha256': file_sha256(spec_path), 'source_sha256': file_sha256(Path(__file__)),
        'identity': 'P = 2*M - 1 + q0', 'cases': cases,
        'groups': {key: {'cases': names,
            'same_both_wrong_fraction': len({cases[n]['both_wrong_fraction'] for n in names}) == 1,
            'epsilon_zero_pair_and_marginal_gate_equivalent_on_listed_cases':
                len({cases[n]['both_wrong_fraction'] for n in names}) == 1}
            for key, names in groups.items()},
        'new_model_calls': 0, 'new_optimizer_steps': 0,
        'limitations': ['Observed error patterns only; not a theorem that the two gates are always equivalent.',
            'H task accuracy is a different dataset; this identity concerns marginal and joint correctness on the same pairs.',
            'Side-wise accuracies are descriptive, not a replacement for a frozen scientific endpoint.',
            'No acceptance receipt, scientific go decision or test-set result is created.']}
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: {x: v[x] for x in ('pairs', 'both_wrong', 'one_correct', 'both_correct')}
                      for k, v in cases.items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.spec.resolve(), args.output.resolve())
