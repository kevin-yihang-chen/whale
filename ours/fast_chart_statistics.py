"""Trace paired test/development differences to samples; bootstrap source charts."""
from collections import defaultdict
import json
from pathlib import Path
import statistics

from .chart_answer_protocol import PROTOCOL,verify
from .evidence import fingerprint
from .visual_native_evaluation import pair_inputs
from .visual_task import file_sha256


def paired_records(manifest_path,result_path):
    manifest,pairs,_=pair_inputs(manifest_path)
    if manifest.get('answer_protocol')!=PROTOCOL:raise ValueError('Do not mix historical scoring protocols')
    result=json.loads(Path(result_path).read_text())
    records={r['sample_id']:r for r in result['records']}
    expected={fingerprint({'pair_id':p.pair_id,'side':s}) for p in pairs for s in (0,1)}
    if len(records)!=len(result['records']) or set(records)!=expected:
        raise ValueError('Incomplete or repeated evaluation samples')
    output={}
    for pair in pairs:
        sides=[records[fingerprint({'pair_id':pair.pair_id,'side':s})] for s in (0,1)]
        correctness=[int(verify(row['committed_answer'],truth)) for row,truth in zip(sides,pair.answers)]
        if correctness!=[row['correct'] for row in sides]:raise ValueError('Stored rewards differ from shared verifier')
        output[pair.pair_id]={'source_id':pair.source_id,'paired':correctness[0]*correctness[1],
            'marginal':sum(correctness)/2,'generated_tokens':sum(row['generated_tokens'] for row in sides),
            'answers':[r['committed_answer'] for r in sides],'correctness':correctness}
    return output,{'manifest_sha256':file_sha256(Path(manifest_path)),
        'result_sha256':file_sha256(Path(result_path)),'role':manifest['partition']}


def matched_source_interval(before,after,*,metric='paired',replicates=20000,seed=20260915):
    """Resample source charts jointly across seeds; condition on these three models."""
    import numpy as np
    if set(before)!=set(after) or set(before)!={42,43,44}:
        raise ValueError('All three registered seeds are required')
    pair_ids=set(before[42]);deltas=[];groups={}
    for trial in (42,43,44):
        if set(before[trial])!=pair_ids or set(after[trial])!=pair_ids:
            raise ValueError('Methods/seeds have different paired coverage')
        row={}
        for ident in sorted(pair_ids):
            a,b=before[trial][ident],after[trial][ident]
            if a['source_id']!=b['source_id'] or groups.get(ident,a['source_id'])!=a['source_id']:
                raise ValueError('Source identity changed across methods/seeds')
            groups[ident]=a['source_id']
            if not (0<=a[metric]<=1 and 0<=b[metric]<=1):raise ValueError('Nonfinite or invalid accuracy')
            row[ident]=b[metric]-a[metric]
        deltas.append(row)
    sources=sorted(set(groups.values()))
    if len(sources)<2:raise ValueError('At least two independent source groups required')
    index={source:i for i,source in enumerate(sources)}
    counts=np.zeros(len(sources));sums=np.zeros(len(sources))
    for ident,source in groups.items():
        j=index[source];counts[j]+=1;sums[j]+=sum(row[ident] for row in deltas)/3
    rng=np.random.default_rng(seed);bootstrap=[]
    for start in range(0,replicates,256):
        draws=rng.integers(0,len(sources),size=(min(256,replicates-start),len(sources)))
        bootstrap.extend((sums[draws].sum(axis=1)/counts[draws].sum(axis=1)).tolist())
    per_seed={s:statistics.mean(deltas[i].values()) for i,s in enumerate((42,43,44))}
    return {'metric':metric,'mean_difference':statistics.mean(per_seed.values()),'per_seed_difference':per_seed,
        'sample_standard_deviation':statistics.stdev(per_seed.values()),
        'source_cluster_95_percentile_interval':np.quantile(bootstrap,[.025,.975]).tolist(),
        'bootstrap':{'unit':'source chart','shared_draw_across_seeds':True,'replicates':replicates,'seed':seed},
        'sources':len(sources),'pairs_per_seed':len(pair_ids),
        'limitation':'Interval reflects source sampling conditional on three trained seed pairs; seed variation is separately reported.'}


def candidate_verdict(paired,marginal,versus_marginal_gate,*,distinct_choices):
    values=list(paired['per_seed_difference'].values())
    criteria={'paired_mean_at_least_one_point':paired['mean_difference']>=.01-1e-12,
        'all_three_seeds_nonnegative':len(values)==3 and min(values)>=-1e-12,
        'at_least_two_positive_seeds':sum(v>1e-12 for v in values)>=2,
        'ordinary_mean_loss_at_most_one_point':marginal['mean_difference']>=-.01-1e-12,
        'main_source_interval_positive':paired['source_cluster_95_percentile_interval'][0]>0,
        'beyond_marginal_gate':versus_marginal_gate['mean_difference']>0 and distinct_choices>0}
    return {'status':'CANDIDATE_EVIDENCE_CRITERIA_MET' if all(criteria.values()) else 'INSUFFICIENT_METHOD_EVIDENCE',
        'criteria':criteria,'publication_acceptance_guaranteed':False}
