import pytest

from ours.acceptance import EvidenceConstrainedAcceptance
from ours.chart_selection_protocol import acceptance_for, selection_modes
from ours.counterfactual_safety import CounterfactualSafetyAcceptance


SETTINGS={'audit_pairs':256,'competence_epsilon':.01,'fragility_epsilon':0.,
          'familywise_alpha':.05,'candidate_budget':3}


def test_versions_preserve_old_rules_and_build_v3_explicitly():
    assert dict(selection_modes({'selection_protocol':'gate_v1'}))['veto']=='paired'
    assert dict(selection_modes({'selection_protocol':'evidence_v2'}))['veto']=='paired_rank'
    plan={'selection_protocol':'safety_v3','safety_audit':SETTINGS}
    assert dict(selection_modes(plan))['veto']=='counterfactual_safety'
    assert isinstance(acceptance_for(plan,'counterfactual_safety'),CounterfactualSafetyAcceptance)
    assert isinstance(acceptance_for(plan,'off'),EvidenceConstrainedAcceptance)


def test_v3_cannot_silently_use_missing_or_extra_settings():
    for settings in (None,{**SETTINGS,'extra':1},{k:v for k,v in SETTINGS.items() if k!='audit_pairs'}):
        with pytest.raises(ValueError):acceptance_for({'selection_protocol':'safety_v3','safety_audit':settings},
                                                      'counterfactual_safety')
    with pytest.raises(ValueError):acceptance_for({'selection_protocol':'evidence_v2','safety_audit':SETTINGS},
                                                  'counterfactual_safety')
