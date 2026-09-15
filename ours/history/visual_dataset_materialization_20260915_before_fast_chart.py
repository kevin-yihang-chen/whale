"""Shared native W/H/C/V inputs before E4 rollouts and E1 acceptance tests.

Check the completed data construction and every image, then expose only pixels
and questions in model messages. T and the independent recheck pool are not
materialized by this development entrypoint.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .evidence import fingerprint
from .visual_native_evaluation import pair_inputs
from .visual_task import SYSTEM_PROMPT, file_sha256


def single_rows(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest['role'] not in {'W','H'} or manifest['role'] != manifest['partition'] or manifest['pairs']:
        raise ValueError('Single-image training/search requires its explicit data role')
    examples = manifest['examples']
    if not examples or len({r['sample_id'] for r in examples}) != len(examples):
        raise ValueError('Missing or duplicated single-image example')
    if set(manifest['source_tables']) != {r['source_id'] for r in examples} or len(manifest['source_tables']) != len(examples):
        raise ValueError('Single-image source coverage differs')
    if set(manifest['image_files']) != {r['image_sha256'] for r in examples}:
        raise ValueError('Single-image bytes coverage differs')
    rows = []
    for index, row in enumerate(examples):
        image = (path.parent/manifest['image_files'][row['image_sha256']]).resolve()
        if not image.is_relative_to(path.parent) or file_sha256(image) != row['image_sha256']:
            raise ValueError('Image path or byte identity differs')
        if row['answer'] not in {'A','B'} or not isinstance(row['question'],str) or not row['question'].strip():
            raise ValueError('Invalid task question or binary answer')
        rows.append({'data_source': 'visual_evidence_binary',
            'visual_sample_id': fingerprint({'role': manifest['role'], 'sample_id': row['sample_id']}),
            'prompt': [{'role':'system','content':SYSTEM_PROMPT},
                {'role':'user','content':'<image>\n'+row['question']}],
            'images': [{'bytes':image.read_bytes()}],
            'reward_model': {'style':'rule','ground_truth':row['answer']}, 'extra_info':{'index':index}})
    return manifest, rows


def materialize(directory, output):
    import pandas as pd
    directory, output = Path(directory).resolve(), Path(output).resolve()
    plan_path, result_path = directory/'plan.json', directory/'rendered/result.json'
    plan, result = [json.loads(p.read_text()) for p in (plan_path,result_path)]
    if result['status'] != 'COMPLETED_RENDER_AND_PIXEL_ORACLE' or result['plan_sha256'] != file_sha256(plan_path):
        raise ValueError('The full rendering and independent pixel checks must complete first')
    if file_sha256(directory/'rendered/pixel-audits.jsonl') != result['pixel_audit_sha256']:
        raise ValueError('Pixel audit changed')
    sources = [source for values in plan['split'].values() for source in values]
    if len(sources) != len(set(sources)) or any(len(plan['split'][k]) != v for k,v in plan['sizes'].items()):
        raise ValueError('Source groups overlap or split sizes differ')
    output.mkdir(exist_ok=False)
    artifacts = {}
    try:
        for role in ('W','H','C','V'):
            path = directory/f'rendered/{role}/manifest.json'
            if file_sha256(path) != result['manifest_sha256'][role]:
                raise ValueError('Completed data manifest changed')
            if role in {'W','H'}:
                manifest, rows = single_rows(path)
                expected_examples = plan['sizes'][role]
            else:
                manifest, pairs, rows = pair_inputs(path)
                expected_examples = 2*plan['sizes'][role]
                if len(pairs) != plan['sizes'][role] or fingerprint([asdict(p) for p in pairs]) != manifest['audit_data_sha256']:
                    raise ValueError('Paired data identity differs')
            if manifest['source_tables'] != plan['split'][role] or len(rows) != expected_examples:
                raise ValueError('Materialized data differs from the frozen source split')
            parquet = output/f'{role}.parquet'
            with parquet.open('xb') as stream:
                pd.DataFrame(rows).to_parquet(stream,index=False)
            artifacts[role] = {'examples':len(rows),'source_tables':len(manifest['source_tables']),
                'manifest':str(path),'manifest_sha256':file_sha256(path),
                'parquet':str(parquet),'parquet_sha256':file_sha256(parquet)}
        report = {'status':'COMPLETED_NATIVE_DEVELOPMENT_MATERIALIZATION','dataset_plan_sha256':file_sha256(plan_path),
            'render_result_sha256':file_sha256(result_path),'source_sha256':file_sha256(Path(__file__)),
            'partitions':artifacts,'model_calls':0,'optimizer_steps':0,
            'limitations':['No task-model, harness or learning-rate selection is performed.',
                'T and R image payloads were not read by this materialization.',
                'C is optimization data and V is development data; their data roles are retained.',
                'Rendering verification does not establish image necessity or the scientific phenomenon.']}
        with (output/'result.json').open('x') as stream:
            json.dump(report,stream,indent=2)
            stream.write('\n')
        print(json.dumps({'status':report['status'],'examples':{k:v['examples'] for k,v in artifacts.items()}}),flush=True)
    except BaseException as error:
        with (output/'failure.json').open('x') as stream:
            json.dump({'status':'INCOMPLETE_MATERIALIZATION','error_type':type(error).__name__,'error':str(error)},stream,indent=2)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    materialize(args.dataset,args.output)
