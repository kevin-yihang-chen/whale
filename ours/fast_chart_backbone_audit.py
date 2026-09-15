"""Bind every accepted E4 image to actual backbone use across four batches."""
from collections import Counter
import json
from pathlib import Path

from .fast_chart_protocol import OUTPUT
from .native_visual_service import write_new
from .visual_pixel_bootstrap import tensor_identity
from .visual_task import file_sha256


def read(path):return json.loads(Path(path).read_text())


def validate_records(expected,records,job_id):
    forwards=[r for r in records if r['kind']=='visual_backbone_input']
    gradients=[r for r in records if r['kind']=='visual_feature_gradient']
    if len(forwards)+len(gradients)!=len(records) or any(str(r['job_id'])!=str(job_id) for r in records):
        raise ValueError('Unknown backbone event or mismatched training job')
    actual=Counter(json.dumps({k:r[k] for k in ('actor_pixels','consumed_pixels','grid')},sort_keys=True) for r in forwards)
    if actual!=expected:
        raise ValueError('Actual backbone images differ from the complete native success subset')
    if any(not r['grad_enabled'] or not r['exact_after_native_dtype_cast'] for r in forwards):
        raise ValueError('Backbone input lacks training gradients or changed its native pixel cast')
    events=Counter(r['event'] for r in forwards)
    if any(n!=1 for n in events.values()) or Counter(r['event'] for r in gradients)!=events:
        raise ValueError('Missing, duplicated or unpaired image-gradient event')
    if any(not r['finite'] or r['l1']<0 for r in gradients):
        raise ValueError('Invalid visual-feature gradient')
    nonzero=sum(r['nonzero_elements']>0 for r in gradients)
    if expected and not nonzero:
        raise ValueError('Successful trajectories have no nonzero visual gradient')
    return {'accepted_examples':sum(expected.values()),'actual_backbone_forwards':len(forwards),
        'paired_gradient_events':len(gradients),'nonzero_gradient_events':nonzero}


def audit(plan_path,output):
    from .training_bootstrap import prepare_worker
    prepare_worker()
    from verl import DataProto
    import torch
    plan_path,output=Path(plan_path).resolve(),Path(output).resolve()
    plan=read(plan_path);root=Path(plan['output']);result_path=root/'execution-result.json';result=read(result_path)
    if (plan['kind']!='compact_chart_rsft_stage' or result['status']!='COMPLETE_COMPACT_NATIVE_STAGE' or
            result['plan_sha256']!=file_sha256(plan_path) or
            plan['config']['trainer']['online_rsft']['sft_epochs']!=1):
        raise ValueError('Require a completed native compact stage with one SFT epoch')
    allocation=OUTPUT/'allocations'/plan_path.stem
    terminal=read(allocation/'allocation-result.json');submission=read(allocation/'submission.json')
    if (terminal['state']!='COMPLETED' or str(terminal['job_id'])!=str(result['job_id']) or
            submission['plan_sha256']!=file_sha256(plan_path)):
        raise ValueError('Image-consumption audit requires the matching completed GPU allocation')
    evidence={str(p):file_sha256(p) for p in (plan_path,result_path,allocation/'allocation-result.json',allocation/'submission.json')}
    expected=Counter();steps=[]
    start=1 if plan['stage']==1 else 5
    if [b['step'] for b in result['batches']]!=list(range(start,start+4)):
        raise ValueError('Incomplete native four-batch execution evidence')
    for batch in result['batches']:
        step=batch['step'];count=batch['successful_trajectories']
        if not count:steps.append({'step':step,'accepted':0});continue
        path=root/f'checkpoints/audit/accepted-step{step}.pkl';receipt_path=path.with_suffix('.json');receipt=read(receipt_path)
        if (receipt['examples']!=count or file_sha256(path)!=receipt['sha256'] or
                str(receipt['job_id'])!=str(result['job_id'])):
            raise ValueError('Accepted native batch differs from execution evidence')
        accepted=DataProto.load_from_disk(path)
        if len(accepted)!=count or set(accepted.non_tensor_batch['visual_harness_sha256'])!={plan['harness_sha256']}:
            raise ValueError('Accepted batch coverage or harness identity differs')
        for image in accepted.non_tensor_batch['multi_modal_inputs']:
            pixels=image['pixel_values'].to(torch.bfloat16)
            expected[json.dumps({'actor_pixels':tensor_identity(pixels),
                'consumed_pixels':tensor_identity(pixels.float()),
                'grid':tensor_identity(image['image_grid_thw'])},sort_keys=True)]+=1
        evidence.update({str(p):file_sha256(p) for p in (path,receipt_path)})
        steps.append({'step':step,'accepted':count});del accepted
    records=[]
    for path in sorted((root/'checkpoints/audit').glob('visual-backbone-*.jsonl')):
        records.extend(json.loads(line) for line in path.read_text().splitlines())
        evidence[str(path)]=file_sha256(path)
    checked=validate_records(expected,records,result['job_id'])
    report={'status':'AUDITED_COMPACT_IMAGE_CONSUMPTION_AND_GRADIENTS',
        'job_id':result['job_id'],'plan_sha256':file_sha256(plan_path),'batches':steps,**checked,
        'evidence_sha256':evidence,'audit_source_sha256':file_sha256(Path(__file__)),
        'model_calls':0,'optimizer_steps':0,'actual_images_match_native_success_subset':True,
        'parameter_change_verified':False,'scientific_method_verified':False,
        'limitations':['Complete success-subset image consumption; full parameter/export comparison remains separate.',
            'Zero-success completed branches remain reportable; they are not omitted for having no updates.']}
    write_new(output,report);return report
