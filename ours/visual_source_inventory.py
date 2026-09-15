"""Verifier-side PlotQA inventory for source-grouped E1 data construction.

The released annotations contain NaN tokens. Preserve original source bytes,
parse individual records with Python's JSON decoder, and explicitly exclude
non-finite records. This inventory is not a rendered or validated paired set.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

from .evidence import fingerprint
from .visual_task import file_sha256


def array_records(stream, *, chunk_size=1024**2, maximum_record_chars=8*1024**2):
    """Read one top-level JSON object array with bounded record memory.

    Nonstandard numeric constants remain visible to the exclusion validator;
    syntax errors, truncated arrays and trailing input fail the whole inventory.
    """
    decoder = json.JSONDecoder()
    buffer, position, eof = '', 0, False

    def refill():
        nonlocal buffer, position, eof
        if eof:
            return False
        block = stream.read(chunk_size)
        buffer = buffer[position:]+block
        position = 0
        eof = not block
        if len(buffer) > maximum_record_chars+chunk_size:
            raise ValueError('Source record exceeds the bounded parser memory')
        return bool(block)

    def whitespace():
        nonlocal position
        while True:
            position = re.compile(r'[ \r\n\t]*').match(buffer, position).end()
            if position < len(buffer) or eof:
                return
            refill()

    refill()
    whitespace()
    if buffer[position:position+1] != '[':
        raise ValueError('Expected a top-level annotation array')
    position += 1
    index, allow_end = 0, True
    while True:
        whitespace()
        if buffer[position:position+1] == ']':
            if not allow_end:
                raise ValueError('Trailing comma in the annotation array')
            position += 1
            whitespace()
            if position != len(buffer):
                raise ValueError('Trailing source content')
            return
        while True:
            try:
                record, end = decoder.raw_decode(buffer, position)
                break
            except json.JSONDecodeError:
                if eof:
                    raise ValueError('Malformed or incomplete annotation record') from None
                if len(buffer)-position > maximum_record_chars:
                    raise ValueError('Oversized annotation record')
                refill()
        if not isinstance(record, dict):
            raise ValueError('Every source annotation must be an object')
        raw_hash = hashlib.sha256(buffer[position:end].encode('utf-8')).hexdigest()
        position = end
        yield index, record, raw_hash
        index += 1
        whitespace()
        separator = buffer[position:position+1]
        if separator == ']':
            allow_end = True
        elif separator == ',':
            position += 1
            allow_end = False
        else:
            raise ValueError('Missing annotation record separator')


def finite_tree(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_tree(v) for v in value)
    return True


def source_group(record):
    """Group identical titled source tables despite rendering style or order."""
    info = record['general_figure_info']
    models = [{'name': str(m['name']).strip(), 'x': m['x'], 'y': m['y']} for m in record['models']]
    models.sort(key=lambda m: fingerprint(m))
    return fingerprint({'title': info['title']['text'].strip(), 'models': models})


def chart_schema(record):
    """Conservative vertical-bar subset for the first derived renderer.

    All displayed numeric values must be finite and nonnegative. Categories
    and series identities must be unique; ambiguous labels are excluded.
    """
    if record['type'] != 'vbar_categorical':
        raise ValueError('not_vertical_bar')
    models = record['models']
    if not 1 <= len(models) <= 4:
        raise ValueError('unsupported_series_count')
    xs = models[0]['x']
    if not 3 <= len(xs) <= 16 or (xs != list(range(len(xs))) and not all(isinstance(x, str) for x in xs)):
        raise ValueError('unsupported_category_coordinates')
    if any(m['x'] != xs or len(m['y']) != len(xs) for m in models):
        raise ValueError('inconsistent_series_coordinates')
    names = [str(m['name']).strip() for m in models]
    if len(set(names)) != len(names) or any(not n or len(n) > 80 for n in names):
        raise ValueError('ambiguous_series_names')
    info = record['general_figure_info']
    labels = info['x_axis']['major_labels']['values']
    ticks = info['x_axis']['major_ticks']['values']
    if len(labels) != len(ticks):
        raise ValueError('tick_label_count_mismatch')
    mapping = {}
    for tick, label in zip(ticks, labels):
        label = str(label).strip()
        if tick in mapping and mapping[tick] != label:
            raise ValueError('ambiguous_tick_labels')
        mapping[tick] = label
    positions = list(range(len(xs)))
    if set(mapping) != set(positions):
        raise ValueError('category_label_coverage')
    categories = [mapping[x] for x in positions]
    if all(isinstance(x, str) for x in xs) and [x.strip() for x in xs] != categories:
        raise ValueError('source_categories_differ_from_display_labels')
    if len(set(categories)) != len(categories) or any(not n or len(n) > 64 for n in categories):
        raise ValueError('ambiguous_category_names')
    values = [m['y'] for m in models]
    flat = [v for series in values for v in series]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in flat):
        raise ValueError('unsupported_numeric_values')
    if max(flat) == 0 or max(flat)-min(flat) < .05*max(flat):
        raise ValueError('insufficient_rendered_value_separation')
    return {'title': info['title']['text'], 'x_label': info['x_axis']['label']['text'],
        'y_label': info['y_axis']['label']['text'], 'series_names': names,
        'categories': categories, 'values': values}


def inventory(receipt_path, output):
    receipt_path, output = Path(receipt_path).resolve(), Path(output).resolve()
    receipt = json.loads(receipt_path.read_text())
    source = Path(receipt['path'])
    if receipt['status'] != 'DOWNLOADED_SOURCE_BYTES' or receipt['source'] != 'plotqa_train_annotations':
        raise ValueError('Expected completed official PlotQA train acquisition')
    if file_sha256(source) != receipt['sha256'] or source.stat().st_size != receipt['bytes']:
        raise ValueError('Acquired source identity differs')
    output.mkdir(exist_ok=False)
    counts, exclusions = Counter(), Counter()
    groups, eligible_groups, image_ids = set(), set(), set()
    record = {'status': 'STARTED_SOURCE_INVENTORY', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_receipt': str(receipt_path), 'source_receipt_sha256': file_sha256(receipt_path),
        'source_sha256': receipt['sha256'], 'implementation_sha256': file_sha256(Path(__file__)),
        'model_calls': 0, 'rendered_images': 0}
    (output/'start.json').write_text(json.dumps(record, indent=2)+'\n')
    try:
        with source.open(encoding='utf-8') as stream, (output/'exclusions.jsonl').open('x') as rejected, \
                (output/'eligible-groups.jsonl').open('x') as eligible:
            for index, row, raw_hash in array_records(stream):
                counts['records'] += 1
                reason = None
                image_id = str(row.get('image_index', ''))
                if not image_id or image_id in image_ids:
                    raise ValueError('Missing or repeated source image identity')
                image_ids.add(image_id)
                if not finite_tree(row):
                    reason = 'nonfinite_source_record'
                else:
                    counts['type:'+str(row.get('type'))] += 1
                    try:
                        group = source_group(row)
                        groups.add(group)
                        schema = chart_schema(row)
                        if group in eligible_groups:
                            reason = 'duplicate_source_table'
                        else:
                            eligible_groups.add(group)
                            eligible.write(json.dumps({'source_group_id': group, 'source_image_id': image_id,
                                'record_index': index, 'source_record_sha256': raw_hash, 'chart': schema},
                                ensure_ascii=False, allow_nan=False)+'\n')
                    except (KeyError, TypeError, ValueError) as error:
                        reason = str(error)
                if reason is not None:
                    exclusions[reason] += 1
                    rejected.write(json.dumps({'record_index': index, 'source_image_id': image_id,
                        'source_record_sha256': raw_hash, 'reason': reason}, allow_nan=False)+'\n')
        record.update(status='COMPLETED_SOURCE_INVENTORY', counts=dict(counts), exclusions=dict(exclusions),
            distinct_finite_source_tables=len(groups), eligible_vertical_bar_groups=len(eligible_groups),
            artifact_sha256={n: file_sha256(output/n) for n in ('eligible-groups.jsonl', 'exclusions.jsonl')},
            limitations=['The vertical-bar subset is a declared derived-data candidate pool, not official PlotQA task scores.',
                'Table identity groups exact titled values independent of style and series order; it is not a near-duplicate detector.',
                'No rendered pairs, pixel oracle, source split, scientific sample sizes or leakage gate are certified yet.',
                'Source nonfinite records are excluded explicitly; original downloaded bytes are preserved.'])
        (output/'result.json').write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record), flush=True)
    except BaseException as error:
        record.update(status='INCOMPLETE_SOURCE_INVENTORY', error_type=type(error).__name__, error=str(error),
            partial_counts=dict(counts), partial_exclusions=dict(exclusions))
        (output/'failure.json').write_text(json.dumps(record, indent=2)+'\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inventory(args.receipt, args.output)
