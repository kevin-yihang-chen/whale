"""PlotQA-EvidencePairs: source-derived, verifiable chart inputs for E1/E4.

W/H use one unedited chart each. C/V/T/R use the same question on two complete
re-renderings with two numerical cells exchanged. This is a derived binary
comparison task, not the official PlotQA question distribution or score.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from .evidence import VisualPair, fingerprint
from .visual_task import file_sha256

SIZES = {'W': 2048, 'H': 128, 'C': 64, 'V': 256, 'T': 1024, 'R': 1024}
PALETTE = ('#4477aa', '#ee6677', '#228833', '#aa3377')


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def canonical_table(chart):
    """Merge identical named data across titles, category/series order and style."""
    return fingerprint(sorted([
        {'series': name.strip(), 'cells': sorted([(category.strip(), str(Decimal(str(value)).normalize()))
            for category, value in zip(chart['categories'], values)])}
        for name, values in zip(chart['series_names'], chart['values'])], key=lambda row: row['series']))


def comparison(chart, group):
    cells = [(series, column) for series, values in enumerate(chart['values']) for column in range(len(values))]
    ranked = sorted(cells, key=lambda cell: fingerprint({'group': group, 'cell': cell}))
    scale = max(value for row in chart['values'] for value in row)
    for first in ranked:
        for second in ranked:
            if abs(chart['values'][first[0]][first[1]]-chart['values'][second[0]][second[1]]) >= .05*scale:
                return [list(first), list(second)]
    raise ValueError('No independently resolvable comparison')


def truth(values, cells):
    a, b = (Decimal(str(values[series][column])) for series, column in cells)
    if a == b:
        raise ValueError('Ambiguous equal-value comparison')
    return 'A' if a > b else 'B'


def exchange(values, cells):
    changed = deepcopy(values)
    (a, b), (c, d) = cells
    changed[a][b], changed[c][d] = changed[c][d], changed[a][b]
    if truth(values, cells) == truth(changed, cells):
        raise ValueError('Edit failed to change the answer')
    return changed


def prepare(inventory_path, output):
    inventory_path, output = Path(inventory_path).resolve(), Path(output).resolve()
    inventory = json.loads(inventory_path.read_text())
    groups_path = inventory_path.parent/'eligible-groups.jsonl'
    if inventory['status'] != 'COMPLETED_SOURCE_INVENTORY' or file_sha256(groups_path) != inventory['artifact_sha256']['eligible-groups.jsonl']:
        raise ValueError('Incomplete or changed source inventory')
    groups = {}
    with groups_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            group = canonical_table(row['chart'])
            entry = groups.setdefault(group, {'representative': row, 'source_ids': [], 'source_group_ids': []})
            entry['source_ids'].append(row['source_image_id'])
            entry['source_group_ids'].append(row['source_group_id'])
    ordered = sorted(groups, key=lambda group: fingerprint({'split_seed': 20260910, 'table': group}))
    if len(ordered) < sum(SIZES.values()):
        raise ValueError('Insufficient distinct source tables for all frozen roles')
    output.mkdir(exist_ok=False)
    recipes, offset = [], 0
    for role, count in SIZES.items():
        for group in ordered[offset:offset+count]:
            entry = groups[group]
            source = entry['representative']
            chart = source['chart']
            cells = comparison(chart, group)
            a, b = cells
            question = (f'In the chart, is the value for "{chart["categories"][a[1]]}" in series '
                f'"{chart["series_names"][a[0]]}" greater than the value for '
                f'"{chart["categories"][b[1]]}" in series "{chart["series_names"][b[0]]}"? '
                'Answer A for yes or B for no.')
            paired = role not in {'W', 'H'}
            recipes.append({'source_table_id': group, 'role': role,
                'source': source, 'equivalent_source_image_ids': entry['source_ids'],
                'equivalent_source_group_ids': entry['source_group_ids'],
                'question': question, 'cells': cells, 'paired': paired,
                'edited_side_first': paired and int(group[-1], 16) % 2 == 1,
                'answers': [truth(chart['values'], cells), truth(exchange(chart['values'], cells), cells)] if paired else [truth(chart['values'], cells)]})
        offset += count
    split = {role: [r['source_table_id'] for r in recipes if r['role'] == role] for role in SIZES}
    if len(set(group for values in split.values() for group in values)) != sum(SIZES.values()):
        raise ValueError('Source table crossed data partitions')
    save(output/'recipes.json', recipes)
    report = {'status': 'PREPARED_DERIVED_DATA_RECIPES', 'name': 'PlotQA-EvidencePairs',
        'inventory': str(inventory_path), 'inventory_sha256': file_sha256(inventory_path),
        'recipes_sha256': file_sha256(output/'recipes.json'), 'source_sha256': file_sha256(Path(__file__)),
        'sizes': SIZES, 'split_seed': 20260910, 'split': split, 'distinct_named_tables': len(groups),
        'unselected_source_tables': len(groups)-len(recipes), 'model_calls': 0,
        'limitations': ['Binary numerical comparison on re-rendered source tables, not an official PlotQA score.',
            'The edit swaps two numerical cells; it is not claimed to be a minimal single-factor intervention.',
            'W and H contain only original values; paired data is not silently added to baseline training.',
            'C is optimization data; R is a separate re-evaluation pool, initially its first 512 pairs.',
            'T is generated and verified by deterministic data checks, never used for model or harness selection.',
            'Exact named-table grouping also ignores title and axis units conservatively; near-duplicate leakage still requires review.',
            'Rendering, pixel verification, image necessity and the scientific go decision are not certified by preparation.']}
    save(output/'plan.json', report)
    return report


def draw(chart, values, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from textwrap import fill
    fig, ax = plt.subplots(figsize=(10, 7), dpi=100)
    fig.subplots_adjust(left=.13, right=.98, bottom=.25, top=.74)
    width = .8/len(values)
    maximum = max(v for row in values for v in row)
    centers = []
    for series, row in enumerate(values):
        x = [column-.4+(series+.5)*width for column in range(len(row))]
        ax.bar(x, row, width=width*.88, color=PALETTE[series], edgecolor='none', antialiased=False,
            zorder=3, label=fill(chart['series_names'][series], 40))
        centers.append(x)
    ax.set_ylim(0, maximum*1.15)
    ax.set_xticks(range(len(chart['categories'])), [fill(v, 14) for v in chart['categories']], fontsize=8, rotation=30, ha='right')
    ax.set_xlabel(fill(chart['x_label'], 100), fontsize=10)
    ax.set_ylabel(fill(chart['y_label'], 55), fontsize=10)
    ax.grid(axis='y', color='#dddddd', zorder=0)
    fig.suptitle(fill(chart['title'], 100), fontsize=11, y=.98)
    fig.legend(loc='upper center', bbox_to_anchor=(.55,.91), ncol=2, fontsize=8, frameon=False)
    fig.canvas.draw()
    height = fig.canvas.get_width_height()[1]
    bottom = height-ax.transData.transform((0,0))[1]
    top = height-ax.transData.transform((0,maximum*1.15))[1]
    observation = {'centers_x': [[ax.transData.transform((x,0))[0] for x in row] for row in centers],
        'top': top, 'bottom': bottom, 'units_per_pixel': maximum*1.15/(bottom-top), 'palette': PALETTE}
    fig.savefig(path, metadata={'Software': 'PlotQA-EvidencePairs'})
    plt.close(fig)
    return observation


def pixel_values(path, geometry):
    """Recover bar heights from saved RGB pixels, without receiving their values."""
    from PIL import Image
    with Image.open(path) as image:
        image = image.convert('RGB')
        values = []
        for series, xs in enumerate(geometry['centers_x']):
            color = tuple(bytes.fromhex(geometry['palette'][series].lstrip('#')))
            row = []
            for x in xs:
                ys = [y for y in range(int(geometry['top'])+1, int(geometry['bottom'])-1)
                    if image.getpixel((round(x),y)) == color]
                row.append((geometry['bottom']-min(ys))*geometry['units_per_pixel'] if ys else 0.)
            values.append(row)
    return values


def render(directory):
    directory = Path(directory).resolve()
    plan = json.loads((directory/'plan.json').read_text())
    if plan['source_sha256'] != file_sha256(Path(__file__)) or plan['recipes_sha256'] != file_sha256(directory/'recipes.json'):
        raise ValueError('Changed derived-data renderer or recipes')
    recipes = json.loads((directory/'recipes.json').read_text())
    output = directory/'rendered'
    output.mkdir(exist_ok=False)
    for role in SIZES:
        (output/role/'images').mkdir(parents=True)
    manifests = {role: {'role': 'V' if role == 'R' else role, 'partition': role, 'name': 'PlotQA-EvidencePairs',
        'image_files': {}, 'pairs': [], 'examples': [], 'source_tables': [], 'split_plan_sha256': file_sha256(directory/'plan.json')}
        for role in SIZES}
    try:
        with (output/'pixel-audits.jsonl').open('x') as audit:
            for index, recipe in enumerate(recipes):
                role, group = recipe['role'], recipe['source_table_id']
                chart, cells = recipe['source']['chart'], recipe['cells']
                values = [chart['values']]
                if recipe['paired']:
                    values.append(exchange(chart['values'], cells))
                answers = [truth(v, cells) for v in values]
                if answers != recipe['answers']:
                    raise ValueError('Recipe answer computation differs')
                images = []
                for side, numeric in enumerate(values):
                    relative = f'images/{fingerprint({"table": group, "side": side})}.png'
                    path = output/role/relative
                    geometry = draw(chart, numeric, path)
                    decoded = pixel_values(path, geometry)
                    tolerance = 2*geometry['units_per_pixel']
                    if any(abs(a-b)>tolerance for r,s in zip(numeric,decoded) for a,b in zip(r,s)) or truth(decoded,cells) != answers[side]:
                        raise ValueError(f'Saved pixels do not certify values for {role}/{group}/{side}')
                    sha = file_sha256(path)
                    images.append(sha)
                    manifests[role]['image_files'][sha] = relative
                    audit.write(json.dumps({'source_table_id': group, 'role': role, 'side': side, 'image_sha256': sha,
                        'geometry': geometry, 'decoded_values': decoded, 'tolerance': tolerance,
                        'computed_answer': answers[side], 'pixel_answer': truth(decoded,cells)}, allow_nan=False)+'\n')
                manifest = manifests[role]
                manifest['source_tables'].append(group)
                if recipe['paired']:
                    if recipe['edited_side_first']:
                        images.reverse()
                        answers.reverse()
                    manifest['pairs'].append(asdict(VisualPair(group, group, recipe['question'], images, answers)))
                else:
                    manifest['examples'].append({'sample_id': group, 'source_id': group, 'question': recipe['question'],
                        'image_sha256': images[0], 'answer': answers[0]})
                if (index+1)%128 == 0:
                    print(json.dumps({'status': 'RENDERING_AND_VERIFYING', 'source_tables': index+1}), flush=True)
        for role, manifest in manifests.items():
            if len(manifest['source_tables']) != SIZES[role]:
                raise ValueError('Incomplete source-role coverage')
            if manifest['pairs']:
                manifest['audit_data_sha256'] = fingerprint(manifest['pairs'])
            save(output/role/'manifest.json', manifest)
        save(output/'result.json', {'status': 'COMPLETED_RENDER_AND_PIXEL_ORACLE', 'model_calls': 0,
            'plan_sha256': file_sha256(directory/'plan.json'), 'source_sha256': file_sha256(Path(__file__)),
            'source_tables': len(recipes), 'images': sum(len(m['image_files']) for m in manifests.values()),
            'manifest_sha256': {role: file_sha256(output/role/'manifest.json') for role in SIZES},
            'pixel_audit_sha256': file_sha256(output/'pixel-audits.jsonl'),
            'limitations': plan['limitations']+['Pixel geometry and binary answer checks passed; text readability, external overlap and model image necessity still need review.']})
    except BaseException as error:
        save(output/'failure.json', {'status': 'INCOMPLETE_RENDER_OR_PIXEL_ORACLE', 'error_type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'render'), required=True)
    parser.add_argument('--inventory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.phase == 'prepare':
        if args.inventory is None:
            parser.error('Preparation requires a completed source inventory')
        report = prepare(args.inventory, args.output)
        print(json.dumps({'status': report['status'], 'sizes': report['sizes']}), flush=True)
    else:
        render(args.output)
