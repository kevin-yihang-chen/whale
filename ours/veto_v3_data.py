"""Build the source-disjoint H-pair proposal set and fresh C256 VETO-v3 audit.

Membership and task assignment use only PlotQA source tables and the pixel
oracle. No model output, historical score, R example, or sealed T example is
read. H-pair reuses the registered H sources for proposal feedback; C-v3 is
drawn deterministically from source tables outside every historical split.
"""

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

from .chart_answer_protocol import PROTOCOL, decode_truth, verify
from .evidence import VisualPair, fingerprint
from .plotqa_evidence_pairs import canonical_table, comparison, draw, exchange, pixel_values, save, truth
from .plotqa_multitask import answers, balanced_assignment, eligibility, question
from .visual_native_evaluation import write_pair_parquet
from .visual_task import file_sha256

SELECTION_SEED = 20260916
H_PAIR_SIZE = 128
C_SIZE = 256


def read(path):
    return json.loads(Path(path).read_text())


def _source_groups(inventory_path):
    inventory_path = Path(inventory_path).resolve()
    inventory = read(inventory_path)
    groups_path = inventory_path.parent / 'eligible-groups.jsonl'
    if (inventory.get('status') != 'COMPLETED_SOURCE_INVENTORY' or
            file_sha256(groups_path) != inventory['artifact_sha256']['eligible-groups.jsonl']):
        raise ValueError('Incomplete or changed PlotQA source inventory')
    groups = {}
    for line in groups_path.read_text().splitlines():
        row = json.loads(line)
        group = canonical_table(row['chart'])
        item = groups.setdefault(group, {'representative': row, 'source_ids': [], 'source_group_ids': []})
        item['source_ids'].append(row['source_image_id'])
        item['source_group_ids'].append(row['source_group_id'])
    return inventory, groups_path, groups


def _paired_recipe(group, entry, role, *, edited_side_first):
    source = entry['representative']
    chart = source['chart']
    cells = comparison(chart, group)
    (a, b), (c, d) = cells
    prompt = (f'In the chart, is the value for "{chart["categories"][b]}" in series '
              f'"{chart["series_names"][a]}" greater than the value for '
              f'"{chart["categories"][d]}" in series "{chart["series_names"][c]}"? '
              'Answer A for yes or B for no.')
    changed = exchange(chart['values'], cells)
    return {'source_table_id': group, 'role': role, 'source': source,
        'equivalent_source_image_ids': entry['source_ids'],
        'equivalent_source_group_ids': entry['source_group_ids'],
        'question': prompt, 'cells': cells, 'paired': True,
        'edited_side_first': edited_side_first,
        'answers': [truth(chart['values'], cells), truth(changed, cells)]}


def prepare(output, *, historical_source, inventory_path, multitask_h_manifest):
    output = Path(output).resolve()
    historical_source = Path(historical_source).resolve()
    inventory_path = Path(inventory_path).resolve()
    multitask_h_manifest = Path(multitask_h_manifest).resolve()
    if output.exists():
        raise FileExistsError(output)
    historical_plan = read(historical_source / 'plan.json')
    historical_recipes_path = historical_source / 'recipes.json'
    if (historical_plan.get('recipes_sha256') != file_sha256(historical_recipes_path) or
            historical_plan.get('inventory_sha256') != file_sha256(inventory_path)):
        raise ValueError('Historical split or inventory changed')
    old_recipes = read(historical_recipes_path)
    h_recipes = [deepcopy(row) for row in old_recipes if row['role'] == 'H']
    if len(h_recipes) != H_PAIR_SIZE:
        raise ValueError('Historical H split does not contain 128 sources')
    h_manifest = read(multitask_h_manifest)
    if (h_manifest.get('role') != 'H' or h_manifest.get('partition') != 'H' or
            set(h_manifest.get('source_tables', ())) != {row['source_table_id'] for row in h_recipes}):
        raise ValueError('Pair feedback must use the registered ordinary H sources')
    inventory, groups_path, groups = _source_groups(inventory_path)
    historical = {group for split in historical_plan['split'].values() for group in split}
    if len(historical) != sum(historical_plan['sizes'].values()) or not historical <= set(groups):
        raise ValueError('Historical source split is incomplete or overlaps')
    unused = sorted(set(groups) - historical,
                    key=lambda group: fingerprint({'veto_v3_seed': SELECTION_SEED, 'table': group}))
    if len(unused) < C_SIZE:
        raise ValueError('Insufficient unselected source tables for fresh C256')
    c_groups = unused[:C_SIZE]
    for row in h_recipes:
        row['paired'] = True
        row['edited_side_first'] = False
        row['role'] = 'H'
        row['answers'] = [truth(row['source']['chart']['values'], row['cells']),
                          truth(exchange(row['source']['chart']['values'], row['cells']), row['cells'])]
    c_recipes = [_paired_recipe(group, groups[group], 'C', edited_side_first=int(group[-1], 16) % 2 == 1)
                 for group in c_groups]
    output.mkdir()
    save(output / 'recipes.json', h_recipes + c_recipes)
    plan = {'status':'PREPARED_VETO_V3_PAIR_DATA', 'name':'PlotQA-EvidencePairs-VETO-v3',
        'selection_seed':SELECTION_SEED, 'sizes':{'H-pair':H_PAIR_SIZE, 'C':C_SIZE},
        'split':{'H-pair':[row['source_table_id'] for row in h_recipes], 'C':c_groups},
        'historical_split_sha256':fingerprint(historical_plan['split']),
        'historical_source_plan':str(historical_source/'plan.json'),
        'historical_source_plan_sha256':file_sha256(historical_source/'plan.json'),
        'historical_recipes_sha256':file_sha256(historical_recipes_path),
        'inventory':str(inventory_path), 'inventory_sha256':file_sha256(inventory_path),
        'eligible_groups_sha256':file_sha256(groups_path),
        'multitask_h_manifest':str(multitask_h_manifest),
        'multitask_h_manifest_sha256':file_sha256(multitask_h_manifest),
        'recipes_sha256':file_sha256(output/'recipes.json'), 'source_sha256':file_sha256(Path(__file__)),
        'distinct_named_tables':len(groups), 'fresh_C_overlap_with_historical':0, 'model_calls':0,
        'limitations':['H-pair is proposal data derived from registered H sources.',
            'C-v3 is optimization data selected before new model calls from previously unused source tables.',
            'No R or T examples, model outcomes, or historical candidate scores determine membership.']}
    save(output / 'plan.json', plan)
    return plan


def _render_recipe(recipe, directory, pixel_stream):
    group = recipe['source_table_id']
    chart, cells = recipe['source']['chart'], recipe['cells']
    numeric_tables = [chart['values'], exchange(chart['values'], cells)]
    image_sha256, decoded_tables = [], []
    for side, numeric in enumerate(numeric_tables):
        relative = f'images/{fingerprint({"v3_table":group,"side":side})}.png'
        path = directory / relative
        geometry = draw(chart, numeric, path)
        decoded = pixel_values(path, geometry)
        tolerance = 2 * geometry['units_per_pixel']
        if (any(abs(a-b) > tolerance for expected, observed in zip(numeric, decoded)
                for a, b in zip(expected, observed)) or truth(decoded, cells) != truth(numeric, cells)):
            raise ValueError(f'Pixel oracle failed for {group}/{side}')
        digest = file_sha256(path)
        image_sha256.append(digest); decoded_tables.append(decoded)
        pixel_stream.write(json.dumps({'source_table_id':group, 'role':recipe['role'], 'side':side,
            'image_sha256':digest, 'geometry':geometry, 'decoded_values':decoded,
            'tolerance':tolerance, 'computed_answer':truth(numeric,cells),
            'pixel_answer':truth(decoded,cells)}, allow_nan=False)+'\n')
    return image_sha256, decoded_tables


def preference_preserving_assignment(recipes, eligible, preferred):
    """Meet exact task quotas while changing as few registered H questions as possible."""
    groups = [row['source_table_id'] for row in recipes]
    tasks = ('comparison','read_value','difference')
    quota = {task:len(groups)//3 + int(index < len(groups)%3) for index,task in enumerate(tasks)}
    # A state records the first two task counts; the third is implied by index.
    layers = [{(0,0):(0,None,None)}]
    for index,group in enumerate(groups):
        current = layers[-1]; following = {}
        choices = sorted(eligible[group], key=lambda task:(task != preferred[group], tasks.index(task)))
        for state,(cost,_,_) in current.items():
            used_third = index-state[0]-state[1]
            for task in choices:
                nxt=(state[0]+int(task==tasks[0]), state[1]+int(task==tasks[1]))
                next_third=used_third+int(task==tasks[2])
                if nxt[0]>quota[tasks[0]] or nxt[1]>quota[tasks[1]] or next_third>quota[tasks[2]]:
                    continue
                candidate=(cost+int(task != preferred[group]),state,task)
                if nxt not in following or candidate[0] < following[nxt][0]:
                    following[nxt]=candidate
        if not following:
            raise ValueError('H-pair tasks cannot preserve fixed coverage and exact quotas')
        layers.append(following)
    state=(quota[tasks[0]],quota[tasks[1]])
    if state not in layers[-1]:
        raise ValueError('H-pair task quotas are infeasible')
    assignment={}
    for index in range(len(groups),0,-1):
        _,previous,task=layers[index][state]
        assignment[groups[index-1]]=task
        state=previous
    return assignment


def render(directory):
    directory = Path(directory).resolve()
    plan = read(directory / 'plan.json')
    recipes_path = directory / 'recipes.json'
    if (plan.get('status') != 'PREPARED_VETO_V3_PAIR_DATA' or
            plan.get('source_sha256') != file_sha256(Path(__file__)) or
            plan.get('recipes_sha256') != file_sha256(recipes_path)):
        raise ValueError('Changed V3 data plan, recipes, or builder')
    for key in ('historical_source_plan', 'inventory', 'multitask_h_manifest'):
        if file_sha256(Path(plan[key])) != plan[key+'_sha256']:
            raise ValueError(f'Changed V3 source input: {key}')
    historical_plan = read(plan['historical_source_plan'])
    if fingerprint(historical_plan['split']) != plan['historical_split_sha256']:
        raise ValueError('Historical split changed')
    groups_path = Path(plan['inventory']).parent / 'eligible-groups.jsonl'
    if file_sha256(groups_path) != plan['eligible_groups_sha256']:
        raise ValueError('Eligible source inventory changed')
    recipes = read(recipes_path)
    by_role = {'H-pair':[row for row in recipes if row['role']=='H'],
               'C':[row for row in recipes if row['role']=='C']}
    if {key:len(value) for key,value in by_role.items()} != plan['sizes']:
        raise ValueError('V3 recipe coverage changed')
    dataset = directory / 'dataset'
    (dataset / 'H-pair' / 'images').mkdir(parents=True)
    (dataset / 'C' / 'images').mkdir(parents=True)
    rendered = {}
    try:
        with (directory / 'pixel-audits.jsonl').open('x') as pixel_stream:
            for role, rows in by_role.items():
                output = dataset / role
                rendered[role] = {}
                for index, recipe in enumerate(rows):
                    images, decoded = _render_recipe(recipe, output, pixel_stream)
                    rendered[role][recipe['source_table_id']] = (images, decoded)
                    if (index + 1) % 64 == 0:
                        print(json.dumps({'status':'RENDERING_VETO_V3_PAIRS','role':role,
                                          'source_tables':index+1}), flush=True)
        registered_h = read(plan['multitask_h_manifest'])
        h_allowed = {row['source_table_id']:eligibility(row, rendered['H-pair'][row['source_table_id']][1])
                     for row in by_role['H-pair']}
        assignments = preference_preserving_assignment(
            by_role['H-pair'], h_allowed, registered_h['task_by_source'])
        h_task_changes = sum(assignments[group] != registered_h['task_by_source'][group]
                             for group in assignments)
        allowed = {row['source_table_id']:eligibility(row, rendered['C'][row['source_table_id']][1])
                   for row in by_role['C']}
        assignments.update(balanced_assignment(by_role['C'], allowed))
        partitions = {}
        historical = {group for split in historical_plan['split'].values() for group in split}
        for role, rows in by_role.items():
            output = dataset / role
            manifest = {'role':'H' if role=='H-pair' else 'C', 'partition':role if role=='H-pair' else 'C-v3',
                'name':plan['name'], 'answer_protocol':PROTOCOL, 'image_files':{}, 'pairs':[], 'examples':[],
                'source_tables':[row['source_table_id'] for row in rows], 'task_by_source':{},
                'split_plan_sha256':file_sha256(directory/'plan.json')}
            if role == 'H-pair':
                manifest['base_side_by_pair'] = {}
            for recipe in rows:
                group = recipe['source_table_id']; task = assignments[group]
                images = list(rendered[role][group][0]); target = answers(recipe, task)
                if not all(verify(decode_truth(expected)['value'], expected) for expected in target):
                    raise ValueError('Shared answer protocol rejected a constructed target')
                if recipe['edited_side_first']:
                    images.reverse(); target.reverse()
                for side, digest in enumerate(images):
                    manifest['image_files'][digest] = f'images/{fingerprint({"v3_table":group,"side":1-side if recipe["edited_side_first"] else side})}.png'
                manifest['task_by_source'][group] = task
                manifest['pairs'].append(asdict(VisualPair(group, group, question(recipe, task),
                                                           tuple(images), tuple(target))))
                if role == 'H-pair':
                    manifest['base_side_by_pair'][group] = 0
            manifest['audit_data_sha256'] = fingerprint(manifest['pairs'])
            manifest_path = output / 'manifest.json'; save(manifest_path, manifest)
            parquet = directory / f'{role}.parquet'
            materialized = write_pair_parquet(manifest_path, parquet)
            partitions['H-pair' if role=='H-pair' else 'C'] = {
                'manifest':str(manifest_path), 'manifest_sha256':file_sha256(manifest_path),
                'parquet':str(parquet), 'parquet_sha256':file_sha256(parquet),
                'source_tables':len(rows), 'examples':2*len(rows),
                'task_counts':dict(Counter(manifest['task_by_source'].values())),
                'audit_data_sha256':manifest['audit_data_sha256'], **materialized}
        if set(plan['split']['C']) & historical or set(plan['split']['C']) & set(plan['split']['H-pair']):
            raise ValueError('Fresh C sources overlap proposal or historical data')
        report = {'status':'COMPLETED_VETO_V3_PAIR_MATERIALIZATION',
            'plan_sha256':file_sha256(directory/'plan.json'), 'recipes_sha256':file_sha256(recipes_path),
            'pixel_audit_sha256':file_sha256(directory/'pixel-audits.jsonl'), 'partitions':partitions,
            'source_overlap':{'C_vs_historical':0,'C_vs_H_pair':0}, 'model_calls':0,
            'H_pair_task_changes_from_single_H':h_task_changes,
            'test_or_recheck_inference':False, 'scientific_method_verified':False}
        save(directory / 'result.json', report)
        return report
    except BaseException as exc:
        save(directory / 'failure.json', {'status':'INCOMPLETE_VETO_V3_DATA',
            'error_type':type(exc).__name__, 'error':str(exc)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare','render'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--historical-source', type=Path)
    parser.add_argument('--inventory', type=Path)
    parser.add_argument('--multitask-h-manifest', type=Path)
    args = parser.parse_args()
    if args.phase == 'prepare':
        if not all((args.historical_source,args.inventory,args.multitask_h_manifest)):
            parser.error('prepare requires all source paths')
        value = prepare(args.output, historical_source=args.historical_source,
                        inventory_path=args.inventory, multitask_h_manifest=args.multitask_h_manifest)
    else:
        value = render(args.output)
    print(json.dumps({'status':value['status'], 'sizes':value.get('sizes')}, sort_keys=True))
