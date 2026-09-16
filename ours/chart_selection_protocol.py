"""Version the E2-E3 intervention without relabeling frozen gate-v1 evidence."""
PROTOCOLS = {
    'gate_v1': (('whale', 'off'), ('veto', 'paired'), ('marginal_gate', 'marginal_gate')),
    'evidence_v2': (('whale', 'off'), ('veto', 'paired_rank'),
                    ('marginal_rank_floor', 'marginal_rank_floor')),
}


def selection_modes(plan):
    version = plan.get('selection_protocol', 'gate_v1')
    if version not in PROTOCOLS:
        raise ValueError('Unknown chart selection protocol')
    return PROTOCOLS[version]
