"""V3 boundary tests use synthetic metadata only; they are not results."""

import json
from pathlib import Path

import pytest

from ours.fast_chart_search import validate_v3_candidate_metadata


def payload():
    mechanisms={'h1':'visual_recheck','h2':'reasoning_decomposition','h3':'response_control'}
    return {'candidates':[{'name':name,'parent':'h0','mechanism':mechanism,
        'hypothesis':'observable paired failure should decrease',
        'change':'one bounded policy change',
        'predicted_failure_addressed':'one-side failure',
        'expected_cost':{'extra_policy_calls':0,'extra_tool_calls':0}}
        for name,mechanism in mechanisms.items()]}


def test_v3_metadata_contract_accepts_only_orthogonal_fixed_slots(tmp_path):
    path=tmp_path/'search';path.mkdir();(path/'pending_eval.json').write_text(json.dumps(payload()))
    validate_v3_candidate_metadata(tmp_path,{'selection_protocol':'safety_v3'})
    for mutation in ('mechanism','tool','missing'):
        value=payload()
        if mutation=='mechanism':value['candidates'][0]['mechanism']='response_control'
        elif mutation=='tool':value['candidates'][1]['expected_cost']['extra_tool_calls']=1
        else:value['candidates'].pop()
        (path/'pending_eval.json').write_text(json.dumps(value))
        with pytest.raises(ValueError):
            validate_v3_candidate_metadata(tmp_path,{'selection_protocol':'safety_v3'})


def test_historical_protocol_does_not_require_v3_metadata(tmp_path):
    validate_v3_candidate_metadata(tmp_path,{'selection_protocol':'evidence_v2'})
