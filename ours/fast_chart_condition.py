"""Prepare explicit condition branches from the same completed native prefix."""
import argparse
from pathlib import Path

from .fast_chart_protocol import OUTPUT
from .fast_chart_training import read,prepare
from .fast_chart_augmentation import prepare as prepare_augmentation
from .native_visual_service import write_new
from .visual_search_bridge import once
from .visual_task import file_sha256
from .fast_chart_admission import verified_calibration


def second_stage(path,output,configuration,parent_plan,*,condition,search=None):
    path,output,configuration,parent_plan=map(lambda p:Path(p).resolve(),(path,output,configuration,parent_plan))
    if condition not in ('weight_only','whale','veto','marginal_gate','counterfactual_augmentation'):
        raise ValueError('Harness-only has no weight stage')
    shared,parent=verified_calibration(configuration),read(parent_plan)
    if shared['status']!='COMPLETE_SHARED_CHART_CONFIGURATION' or parent['stage']!=1 or parent['rate']!=shared['learning_rate']:
        raise ValueError('Branch configuration differs from frozen common calibration')
    if parent['harness_sha256']!=file_sha256(Path(shared['harness'])) or parent['h0_calibration']!=shared['h0_calibration']:
        raise ValueError('Common first stage used another initial harness')
    if path.exists() or output.exists() or not output.is_relative_to(OUTPUT):raise ValueError('Require fresh branch paths')
    path.parent.mkdir(parents=True,exist_ok=True)
    augmentation=None
    if condition=='weight_only':
        choice=path.with_name(path.stem+'-fixed-h0.json')
        once(choice,{'status':'COMPLETE_COMPACT_SELECTION','condition':'weight_only',
            'parent_plan_sha256':file_sha256(parent_plan),'selected_harness':shared['harness'],
            'harness_sha256':file_sha256(Path(shared['harness'])),'selection':'Fixed common h0; no search used by this control.'})
    else:
        if search is None:raise ValueError('Optimized branches require the same completed candidate archive')
        search=Path(search).resolve();search_result=read(search/'result.json');search_plan=read(search/'plan.json')
        if search_result['status']!='COMPLETE_COMPACT_SEARCH_AND_SELECTION' or search_plan['parent_plan_sha256']!=file_sha256(parent_plan):
            raise ValueError('Search did not use this common prefix')
        mode='whale' if condition=='counterfactual_augmentation' else condition
        choice=search/f'selection-{mode}.json'
        if condition=='counterfactual_augmentation':
            directory=output.with_name(output.name+'-augmentation')
            prepare_augmentation(directory,parent_plan,choice);augmentation=directory/'manifest.json'
    result=prepare(path,output,calibration=shared['h0_calibration'],seed=parent['seed'],rate=shared['learning_rate'],
        parent_plan=parent_plan,selection=choice,augmentation=augmentation)
    write_new(path.with_name(path.stem+'-condition.json'),{'condition':condition,'seed':parent['seed'],
        'plan':str(path),'plan_sha256':file_sha256(path),'shared_configuration_sha256':file_sha256(configuration),
        'common_first_stage_sha256':file_sha256(parent_plan),'status':'PREPARED_NOT_EXECUTED'})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','output','configuration','parent-plan','search'):p.add_argument('--'+name,type=Path,required=name!='search')
    p.add_argument('--condition',required=True,choices=('weight_only','whale','veto','marginal_gate','counterfactual_augmentation'))
    a=p.parse_args();second_stage(a.plan,a.output,a.configuration,a.parent_plan,condition=a.condition,search=a.search)
