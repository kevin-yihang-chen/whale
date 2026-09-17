from ours.veto_v3_data import preference_preserving_assignment


def test_pair_task_assignment_changes_the_minimum_needed_for_exact_quotas():
    recipes=[{'source_table_id':str(i)} for i in range(6)]
    preferred={'0':'read_value','1':'read_value','2':'comparison',
               '3':'comparison','4':'difference','5':'difference'}
    eligible={key:['comparison','read_value','difference'] for key in preferred}
    eligible['0']=['comparison','difference']
    assigned=preference_preserving_assignment(recipes,eligible,preferred)
    assert {task:list(assigned.values()).count(task) for task in set(assigned.values())} == {
        'comparison':2,'read_value':2,'difference':2}
    assert assigned['0']!='read_value'
    assert sum(assigned[key]!=preferred[key] for key in assigned)==2
