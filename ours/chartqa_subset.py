"""Freeze and acquire the public ChartQA512 subset without model-based selection."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
from pathlib import Path
import urllib.parse
import urllib.request

from .fast_chart_protocol import ROOT, OUTPUT, storage_check
from .evidence import fingerprint
from .native_visual_service import write_new
from .visual_task import file_sha256

REVISION='044eabfc306abfe9340c5741f0093aefc5973d06'
SOURCE=OUTPUT/'public-sources/chartqa'


def fixed_rows(annotations, split):
    groups={}
    for index,row in enumerate(annotations):
        if set(row)!={'imgname','query','label'} or Path(row['imgname']).name!=row['imgname']:
            raise ValueError('Unexpected public annotation schema')
        groups.setdefault(row['imgname'],[]).append((index,row))
    names=sorted(groups,key=lambda name:fingerprint({'protocol':'chartqa512-v1','image':name}))
    if len(names)<256:raise ValueError('Insufficient distinct source charts')
    result=[]
    for name in names[:256]:
        index,row=min(groups[name],key=lambda item:fingerprint({'question':item[1]['query'],'index':item[0]}))
        result.append({'split':split,'source_image':name,'source_annotation_index':index,
            'sample_id':fingerprint({'split':split,'index':index,'annotation':row}),
            'question':row['query'],'answer':str(row['label'])})
    return result


def prepare(path,output):
    output=Path(output).resolve()
    if not output.is_relative_to(OUTPUT) or output.exists():raise ValueError('Require fresh compact subset output')
    rows=[];annotations={}
    for split in ('human','augmented'):
        p=SOURCE/f'test_{split}.json'
        rows+=fixed_rows(json.loads(p.read_text()),split)
        annotations[str(p)]=file_sha256(p)
    tree_path=OUTPUT/'public-sources/chartqa-tree.json'
    tree={x['path']:x for x in json.loads(tree_path.read_text())['tree']}
    artifacts={}
    for name in sorted({r['source_image'] for r in rows}):
        for folder,suffix in (('png','.png'),('tables','.csv')):
            rel=f'ChartQA Dataset/test/{folder}/{Path(name).stem}{suffix}'
            if rel not in tree:raise ValueError(f'Missing original ChartQA image/table: {rel}')
            local=f'{folder}/{Path(name).stem}{suffix}'
            artifacts[local]={'repository_path':rel,'git_blob_sha1':tree[rel]['sha'],
                'bytes':tree[rel]['size'],'url':f'https://raw.githubusercontent.com/vis-nlp/ChartQA/{REVISION}/'+urllib.parse.quote(rel)}
    plan={'kind':'chartqa_fixed_open_answer_subset','name':'ChartQA fixed subset,256human+256augmented',
        'output':str(output),'revision':REVISION,'rows':rows,'artifacts':artifacts,
        'annotation_sha256':annotations,'tree_sha256':file_sha256(tree_path),
        'selection':'Hash-ranked source charts; one hash-ranked original question per chart per split.',
        'scoring':'Original ChartQA paper relaxed accuracy,5% numeric; exact case-insensitive text. Also report legacy repository exact match.',
        'source_sha256':file_sha256(Path(__file__)),'model_results_consulted':False,
        'official_full_benchmark_score':False}
    write_new(Path(path),plan);return plan


def normalized_cell(value):
    text=str(value).strip()
    try:
        number=Decimal(text.replace(',',''))
        if number.is_finite():return str(number.normalize())
    except InvalidOperation:pass
    return ' '.join(text.casefold().split())


def table_signature(categories,series,values):
    cells=sorted((normalized_cell(cat),normalized_cell(name),normalized_cell(values[j][i]))
        for i,cat in enumerate(categories) for j,name in enumerate(series))
    return fingerprint(cells)


def acquire(plan_path):
    plan=json.loads(Path(plan_path).read_text());out=Path(plan['output'])
    if plan['source_sha256']!=file_sha256(Path(__file__)):raise ValueError('Frozen subset builder changed')
    for name,digest in plan['annotation_sha256'].items():
        if file_sha256(Path(name))!=digest:raise ValueError('Original annotations changed')
    storage_check(1)
    out.mkdir();(out/'png').mkdir();(out/'tables').mkdir()
    (out/'plan.json').write_bytes(Path(plan_path).read_bytes())
    def fetch(item):
        name,record=item
        with urllib.request.urlopen(record['url'],timeout=90) as response:
            data=response.read(record['bytes']+1)
        if len(data)!=record['bytes'] or hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()!=record['git_blob_sha1']:
            raise ValueError(f'Public source bytes changed: {name}')
        with (out/name).open('xb') as stream:stream.write(data)
        return name,{**record,'sha256':hashlib.sha256(data).hexdigest()}
    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            artifacts=dict(pool.map(fetch,plan['artifacts'].items()))
        # C is used by the augmentation control, so screen both W and C sources.
        recipes=json.loads((ROOT/'data/plotqa-evidence-pairs-20260910-v1/recipes.json').read_text())
        training_signatures={table_signature(r['source']['chart']['categories'],r['source']['chart']['series_names'],
            r['source']['chart']['values']):r['source_table_id'] for r in recipes if r['role'] in ('W','C')}
        images=set()
        for role in ('W','C'):
            manifest=json.loads((OUTPUT/f'dataset/{role}/manifest.json').read_text())
            images.update(manifest['image_files'])
        overlaps=[]
        for name in sorted({r['source_image'] for r in plan['rows']}):
            if artifacts['png/'+name]['sha256'] in images:overlaps.append({'image':name,'kind':'exact_image_bytes'})
            table=list(csv.reader(io.StringIO((out/f'tables/{Path(name).stem}.csv').read_text(encoding='utf-8-sig'))))
            if len(table)<2 or len(table[0])<2 or any(len(row)!=len(table[0]) for row in table):
                raise ValueError(f'Cannot verify the original source table: {name}')
            signature=table_signature([row[0] for row in table[1:]],table[0][1:],
                [[row[j] for row in table[1:]] for j in range(1,len(table[0]))])
            if signature in training_signatures:overlaps.append({'image':name,'kind':'exact_normalized_labeled_table','training_source':training_signatures[signature]})
        write_new(out/'result.json',{'status':'READY_FIXED_EXTERNAL_SUBSET' if not overlaps else 'BLOCKED_SOURCE_OVERLAP',
            'plan_sha256':file_sha256(Path(plan_path)),'rows':512,'unique_images':len({r['source_image'] for r in plan['rows']}),
            'artifacts':artifacts,'overlaps':overlaps,'model_calls':0,'test_scores_computed':False,
            'overlap_limitations':['Exact byte and labeled-table screening; semantic or rounded-table equivalence is not exhaustively established.',
                'Public checkpoint pretraining data contamination is unknown. No selected chart is replaced based on model errors.']})
    except BaseException as exc:
        write_new(out/'failure.json',{'status':'INCOMPLETE_PUBLIC_ACQUISITION','error_type':type(exc).__name__,'error':str(exc)})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','acquire'))
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.plan,a.output)
    else:acquire(a.plan)
