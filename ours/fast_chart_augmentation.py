"""Same-budget counterfactual augmentation control: replace16 of32 stage-two slots."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from .evidence import fingerprint
from .fast_chart_protocol import ROOT,OUTPUT
from .fast_chart_training import read,checkpoint,inspect_dataset
from .native_visual_service import write_new
from .visual_task import file_sha256


def replacement_map(batches,pair_ids,seed):
    if len(batches)!=4 or any(len(b)!=8 for b in batches) or len(set(x for b in batches for x in b))!=32:
        raise ValueError('Require four distinct eight-question continuation batches')
    chosen=sorted(pair_ids,key=lambda ident:fingerprint({'seed':seed,'augmentation_pair':ident}))[:8]
    if len(set(chosen))!=8:raise ValueError('Need eight distinct C pairs')
    mapping={};rows=[]
    for i,batch in enumerate(batches):
        for j,pair in enumerate(chosen[2*i:2*i+2]):
            for side in (0,1):
                old=batch[4+2*j+side];new=fingerprint({'pair_id':pair,'side':side})
                mapping[old]=new;rows.append({'batch':i+5,'original_sample_id':old,
                    'replacement_sample_id':new,'pair_id':pair,'side':side})
    return mapping,rows


def prepare(output,parent_plan,selection):
    import pandas as pd
    from omegaconf import OmegaConf
    from .visual_native_evaluation import pair_inputs
    output,parent_plan,selection=map(lambda p:Path(p).resolve(),(output,parent_plan,selection))
    if output.exists() or not output.is_relative_to(OUTPUT):raise ValueError('Require fresh augmentation output')
    parent=read(parent_plan);choice=read(selection)
    if parent['kind']!='compact_chart_rsft_stage' or parent['stage']!=1 or choice['condition']!='whale' or choice['parent_plan_sha256']!=file_sha256(parent_plan):
        raise ValueError('Augmentation must branch from the same theta1 using WHALE-selected h')
    native=checkpoint(Path(parent['output'])/'checkpoints/global_step_4',4)
    reference=inspect_dataset(OmegaConf.create(parent['config']),native)
    data=read(OUTPUT/'native-data/result.json')['partitions'];manifest,pairs,_=pair_inputs(data['C']['manifest'])
    mapping,slots=replacement_map(reference['next_four_batches'],[p.pair_id for p in pairs],parent['seed'])
    w=pd.read_parquet(data['W']['parquet']);c=pd.read_parquet(data['C']['parquet'])
    positions={ident:i for i,ident in enumerate(w['visual_sample_id'])};c_rows={r['visual_sample_id']:r for r in c.to_dict('records')}
    records=w.to_dict('records')
    for old,new in mapping.items():
        row=deepcopy(c_rows[new]);row['extra_info']={'index':positions[old]};records[positions[old]]=row
    if len({r['visual_sample_id'] for r in records})!=2048:raise ValueError('Augmentation introduced duplicate sample identities')
    output.mkdir(parents=True);parquet=output/'training.parquet'
    with parquet.open('xb') as stream:pd.DataFrame(records).to_parquet(stream,index=False)
    receipt={'kind':'compact_counterfactual_augmentation','parent_plan':str(parent_plan),'parent_plan_sha256':file_sha256(parent_plan),
        'selection':str(selection),'selection_sha256':file_sha256(selection),
        'original_training_data':data['W'],'audit_data':data['C'],'actual_sampler_reference':reference,
        'replacements':slots,'mapping':mapping,'ratio':.5,'pairs':8,'replaced_questions':16,
        'total_questions':32,'total_trajectories':256,'source_sha256':file_sha256(Path(__file__)),
        'parquet':str(parquet),'parquet_sha256':file_sha256(parquet),
        'limitation':'Same native sampler indices/state, with explicit C payload replacement; not an ordinary W-only branch.'}
    write_new(output/'manifest.json',receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('output','parent-plan','selection'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();prepare(a.output,a.parent_plan,a.selection)
