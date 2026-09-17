"""Version the E2-E3 intervention without relabeling frozen gate-v1 evidence."""
PROTOCOLS = {
    'gate_v1': (('whale', 'off'), ('veto', 'paired'), ('marginal_gate', 'marginal_gate')),
    'evidence_v2': (('whale', 'off'), ('veto', 'paired_rank'),
                    ('marginal_rank_floor', 'marginal_rank_floor')),
    'safety_v3': (('whale', 'off'), ('veto', 'counterfactual_safety'),
                  ('point_gate', 'paired')),
}


def selection_modes(plan):
    version = plan.get('selection_protocol', 'gate_v1')
    if version not in PROTOCOLS:
        raise ValueError('Unknown chart selection protocol')
    return PROTOCOLS[version]


def acceptance_for(plan, mode):
    """Build a versioned selector without changing historical V1/V2 semantics."""
    from .acceptance import EvidenceConstrainedAcceptance
    if mode != 'counterfactual_safety':
        return EvidenceConstrainedAcceptance(mode)
    if plan.get('selection_protocol') != 'safety_v3':
        raise ValueError('Counterfactual safety selector requires the V3 protocol')
    from .counterfactual_safety import CounterfactualSafetyAcceptance
    settings = plan.get('safety_audit')
    expected = {'audit_pairs', 'competence_epsilon', 'fragility_epsilon',
                'familywise_alpha', 'candidate_budget'}
    if not isinstance(settings, dict) or set(settings) != expected:
        raise ValueError('V3 requires a complete frozen safety-audit configuration')
    return CounterfactualSafetyAcceptance(**settings)
