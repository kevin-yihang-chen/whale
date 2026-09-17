"""Constructed verifier outcomes test V3 decisions; they are not results."""

from dataclasses import replace

import pytest

from ours.acceptance import CandidateMetrics
from ours.adapter import WHALEAcceptanceAdapter
from ours.counterfactual_safety import CounterfactualSafetyAcceptance, evidence_statistics
from ours.evidence import AuditReceipt, EvaluationIdentity, fingerprint


def digest(value): return fingerprint(value)


IDENTITY = EvaluationIdentity(*(digest(x) for x in ('weights','H','C','decode','verify')),phase='v3-test')


def archive(n=16):
    candidates=tuple(CandidateMetrics(name,digest(name),accuracy,turns) for name,accuracy,turns in
        [('h0',.6,2),('safe',.8,2),('fail',.98,1),('uncertain',.99,1)])
    rows={
        'h0':((1,0),)*n,
        'safe':((1,1),)*n,
        'fail':((0,0),)*n,
        'uncertain':((1,1),)*(n//2)+((0,0),)*(n-n//2),
    }
    audits={c.name:AuditReceipt(IDENTITY,c.harness_sha256,'C',tuple(f'p{i}' for i in range(n)),rows[c.name])
            for c in candidates}
    return candidates,audits


def test_decomposition_and_uncertainty_abstention_keep_H_as_objective():
    candidates,audits=archive()
    rule=CounterfactualSafetyAcceptance(audit_pairs=16,candidate_budget=3)
    decision,diagnostics=rule.select_with_diagnostics(candidates,incoming='h0',identity=IDENTITY,audits=audits)
    assert decision.selected=='safe'
    assert decision.eligible==('h0','safe')
    assert diagnostics['verdicts']['safe']['verdict']=='PASS'
    assert diagnostics['verdicts']['fail']['verdict']=='FAIL'
    assert diagnostics['verdicts']['uncertain']['verdict']=='UNCERTAIN'
    assert diagnostics['decision_status']=='SELECTED_PASS_WITH_UNCERTAIN_EXCLUSIONS'
    for receipt in audits.values():
        stats=evidence_statistics(receipt)
        assert stats['paired_accuracy']==stats['marginal_competence']-stats['counterfactual_fragility']/2


def test_fixed_size_identity_order_and_candidate_budget_fail_closed():
    candidates,audits=archive()
    rule=CounterfactualSafetyAcceptance(audit_pairs=16,candidate_budget=3)
    with pytest.raises(ValueError,match='fixed maximum'):
        rule.select(candidates,incoming='h0',identity=IDENTITY,
                    audits={name:replace(row,pair_ids=row.pair_ids[:-1],correctness=row.correctness[:-1])
                            for name,row in audits.items()})
    changed=dict(audits);changed['safe']=replace(changed['safe'],pair_ids=tuple(reversed(changed['safe'].pair_ids)))
    with pytest.raises(ValueError,match='matched, ordered'):
        rule.select(candidates,incoming='h0',identity=IDENTITY,audits=changed)
    with pytest.raises(ValueError,match='multiplicity'):
        CounterfactualSafetyAcceptance(audit_pairs=16,candidate_budget=2).select(
            candidates,incoming='h0',identity=IDENTITY,audits=audits)


def test_adapter_binds_and_rechecks_v3_diagnostics_on_resume():
    candidates,audits=archive();rule=CounterfactualSafetyAcceptance(audit_pairs=16,candidate_budget=3)
    adapter=WHALEAcceptanceAdapter(rule)
    kwargs=dict(original_selector=lambda:'fail',incoming='h0',archive_provider=lambda:candidates,
                audit_provider=lambda c:audits[c.name],identity=IDENTITY)
    winner,receipt=adapter.decide(stage='ordinary',**kwargs)
    assert winner=='safe' and receipt['audit_diagnostics']['verdicts']['uncertain']['verdict']=='UNCERTAIN'
    assert adapter.decide(stage='resume',resume_receipt=receipt,**kwargs)[0]=='safe'
    receipt['audit_diagnostics']['verdicts']['safe']['verdict']='FAIL'
    with pytest.raises(ValueError,match='differs'):
        adapter.decide(stage='resume',resume_receipt=receipt,**kwargs)


def test_uncertain_only_archive_abstains_to_incoming():
    candidates,audits=archive()
    subset=(candidates[0],candidates[3])
    rule=CounterfactualSafetyAcceptance(audit_pairs=16,candidate_budget=3)
    decision,diagnostics=rule.select_with_diagnostics(subset,incoming='h0',identity=IDENTITY,
        audits={candidate.name:audits[candidate.name] for candidate in subset})
    assert decision.selected=='h0'
    assert diagnostics['decision_status']=='ABSTAINED_TO_INCOMING'


@pytest.mark.parametrize('field,value',[('audit_pairs',1),('candidate_budget',0),
    ('competence_epsilon',-1),('fragility_epsilon',1.1),('familywise_alpha',0)])
def test_invalid_v3_configuration(field,value):
    with pytest.raises(ValueError): CounterfactualSafetyAcceptance(**{field:value})
