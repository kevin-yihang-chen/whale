"""Versioned chart tasks for E1 and E4 using existing certified image pairs.

No model outcomes determine membership. All source groups and partitions are
preserved. Numeric eligibility uses source values and the saved pixel oracle;
images are immutable hard links, not newly rendered or copied chart data.
"""
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from decimal import Decimal
import json
import os
from pathlib import Path

from .chart_answer_protocol import (PROTOCOL, encode_truth, verify, decimal_text,
                                    disjoint_numeric_answers)
from .evidence import VisualPair, fingerprint
from .plotqa_evidence_pairs import SIZES, exchange, truth, save
from .visual_task import file_sha256

TASKS = ('comparison', 'read_value', 'difference')


def answers(recipe, task, *, decoded=None):
    cells = recipe['cells']
    values = [recipe['source']['chart']['values']]
    if recipe['paired']:
        values.append(exchange(values[0], cells))
    if decoded is not None:
        values = decoded
    output = []
    for table in values:
        a, b = (Decimal(str(table[r][c])) for r, c in cells)
        if task == 'comparison':
            output.append(encode_truth('binary', truth(table, cells)))
        elif task == 'read_value':
            output.append(encode_truth('numeric', a))
        elif task == 'difference':
            output.append(encode_truth('numeric', a - b))
        else:
            raise ValueError('Unknown task')
    return output


def eligibility(recipe, pixel_tables):
    eligible = []
    for task in TASKS:
        target = answers(recipe, task)
        observed = answers(recipe, task, decoded=pixel_tables)
        if not all(verify(json.loads(p)['value'], t) for p, t in zip(observed, target)):
            continue
        if task != 'comparison' and recipe['paired'] and not disjoint_numeric_answers(
                json.loads(target[0])['value'], json.loads(target[1])['value']):
            continue
        eligible.append(task)
    return eligible


def balanced_assignment(recipes, eligible):
    """Deterministic capacity matching; keep every source, never select by score."""
    groups = [r['source_table_id'] for r in recipes]
    quota = {t: len(groups) // 3 + int(i < len(groups) % 3) for i, t in enumerate(TASKS)}
    buckets = {t: [] for t in TASKS}
    assigned = {}
    def place(group, seen):
        for task in sorted(eligible[group], key=lambda t: (len(buckets[t]) >= quota[t], TASKS.index(t))):
            if task in seen:
                continue
            if len(buckets[task]) < quota[task]:
                buckets[task].append(group)
                assigned[group] = task
                return True
            for old in list(buckets[task]):
                if place(old, seen | {task}):
                    buckets[task].remove(old)
                    buckets[task].append(group)
                    assigned[group] = task
                    return True
        return False
    for group in sorted(groups, key=lambda g: (len(eligible[g]), fingerprint({'assignment': 20260915, 'source': g}))):
        if not place(group, set()):
            raise ValueError(f'Balanced numeric tasks cannot be certified without dropping sources: {Counter(t for g in groups for t in eligible[g])}')
    return assigned


def question(recipe, task):
    chart = recipe['source']['chart']
    names = [f'"{chart["categories"][c]}" in series "{chart["series_names"][r]}"' for r, c in recipe['cells']]
    if task == 'comparison':
        return recipe['question']
    subject = f'the value for {names[0]}' if task == 'read_value' else f'the value for {names[0]} minus the value for {names[1]}'
    return f'In the chart, what is {subject}? Return the numeric value in the plotted axis units, without a unit or percent sign.'


def build(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    old_plan = json.loads((source / 'plan.json').read_text())
    rendered = json.loads((source / 'rendered/result.json').read_text())
    if rendered['status'] != 'COMPLETED_RENDER_AND_PIXEL_ORACLE' or rendered['plan_sha256'] != file_sha256(source / 'plan.json'):
        raise ValueError('Uncertified source data')
    if old_plan['recipes_sha256'] != file_sha256(source / 'recipes.json') or rendered['pixel_audit_sha256'] != file_sha256(source / 'rendered/pixel-audits.jsonl'):
        raise ValueError('Changed construction evidence')
    recipes = json.loads((source / 'recipes.json').read_text())
    pixels = {}
    for line in (source / 'rendered/pixel-audits.jsonl').read_text().splitlines():
        row = json.loads(line)
        pixels.setdefault(row['source_table_id'], {})[row['side']] = row['decoded_values']
    allowed = {r['source_table_id']: eligibility(r, [v for _, v in sorted(pixels[r['source_table_id']].items())]) for r in recipes}
    assignments = {}
    for role in SIZES:
        selected = [r for r in recipes if r['role'] == role]
        assignments.update(balanced_assignment(selected, allowed))
    output.mkdir()
    plan = {'name': 'PlotQA-EvidencePairs-Multitask-v1', 'answer_protocol': PROTOCOL,
        'source_plan': str(source / 'plan.json'), 'source_plan_sha256': file_sha256(source / 'plan.json'),
        'source_render_sha256': file_sha256(source / 'rendered/result.json'),
        'source_sha256': file_sha256(Path(__file__)), 'sizes': old_plan['sizes'], 'split': old_plan['split'],
        'assignment_seed': 20260915, 'tasks': TASKS, 'model_calls': 0,
        'limitations': ['Derived tasks, not official PlotQA scores.', 'C is optimization data; T and R are sealed for model inference.',
            'Numerical eligibility is decided by data/pixel oracles, never by model performance.']}
    save(output / 'plan.json', plan)
    partitions = {}
    try:
        for role in SIZES:
            old_path = source / f'rendered/{role}/manifest.json'
            if file_sha256(old_path) != rendered['manifest_sha256'][role]:
                raise ValueError('Changed source manifest')
            old = json.loads(old_path.read_text())
            directory = output / role
            (directory / 'images').mkdir(parents=True)
            manifest = {**deepcopy(old), 'name': plan['name'], 'answer_protocol': PROTOCOL,
                'pairs': [], 'examples': [], 'task_by_source': {}, 'split_plan_sha256': file_sha256(output / 'plan.json')}
            manifest.pop('audit_data_sha256', None)
            for sha, relative in old['image_files'].items():
                src, dst = old_path.parent / relative, directory / relative
                if file_sha256(src) != sha:
                    raise ValueError('Changed source image')
                os.link(src, dst)
            old_examples = {r['source_id']: r for r in old['pairs'] or old['examples']}
            for recipe in (r for r in recipes if r['role'] == role):
                group = recipe['source_table_id']
                task = assignments[group]
                manifest['task_by_source'][group] = task
                target = answers(recipe, task)
                previous = old_examples[group]
                if recipe['paired']:
                    if recipe['edited_side_first']:
                        target.reverse()
                    manifest['pairs'].append(asdict(VisualPair(group, group, question(recipe, task), previous['images_sha256'], target)))
                else:
                    manifest['examples'].append({**previous, 'question': question(recipe, task), 'answer': target[0]})
            if manifest['pairs']:
                manifest['audit_data_sha256'] = fingerprint(manifest['pairs'])
            path = directory / 'manifest.json'
            save(path, manifest)
            partitions[role] = {'manifest': str(path), 'manifest_sha256': file_sha256(path),
                'source_tables': len(manifest['source_tables']), 'examples': SIZES[role] * (2 if manifest['pairs'] else 1),
                'task_counts': dict(Counter(manifest['task_by_source'].values()))}
        result = {'status': 'COMPLETED_MULTITASK_MANIFESTS', 'plan_sha256': file_sha256(output / 'plan.json'),
            'partitions': partitions, 'model_calls': 0, 'images_rerendered': 0, 'source_groups_moved': 0,
            'image_storage': 'immutable hard links to certified source bytes'}
        save(output / 'result.json', result)
        return result
    except BaseException as exc:
        save(output / 'failure.json', {'status': 'INCOMPLETE', 'error': str(exc)})
        raise


def materialize(directory, output):
    import pandas as pd
    from .visual_dataset_materialization import single_rows
    from .visual_native_evaluation import pair_inputs
    directory, output = Path(directory).resolve(), Path(output).resolve()
    result = json.loads((directory / 'result.json').read_text())
    if result['status'] != 'COMPLETED_MULTITASK_MANIFESTS' or result['plan_sha256'] != file_sha256(directory / 'plan.json'):
        raise ValueError('Incomplete multitask construction')
    output.mkdir(exist_ok=False)
    partitions = {}
    for role in ('W', 'H', 'C', 'V'):
        item = result['partitions'][role]
        if file_sha256(Path(item['manifest'])) != item['manifest_sha256']:
            raise ValueError('Changed multitask manifest')
        loaded = single_rows(item['manifest']) if role in {'W', 'H'} else pair_inputs(item['manifest'])
        rows = loaded[-1]
        if len(rows) != item['examples']:
            raise ValueError('Incomplete native input coverage')
        path = output / f'{role}.parquet'
        with path.open('xb') as stream:
            pd.DataFrame(rows).to_parquet(stream, index=False)
        partitions[role] = {**item, 'parquet': str(path), 'parquet_sha256': file_sha256(path)}
    report = {'status': 'COMPLETED_SHARED_CHART_MATERIALIZATION', 'dataset': str(directory),
        'dataset_result_sha256': file_sha256(directory / 'result.json'), 'partitions': partitions,
        'model_calls': 0, 'test_or_recheck_inference': False}
    save(output / 'result.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('build', 'materialize'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = (build if args.phase == 'build' else materialize)(args.source, args.output)
    print(json.dumps(result, indent=2))
