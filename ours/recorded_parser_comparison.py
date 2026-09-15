"""Compare parsers on fixed audited replies without inventing an ablation.

This diagnoses the E3 harness components on recorded responses. It does not
estimate the score of a new prompt/parser combination, whose generated replies
would require a separate controlled evaluation. No model or API is called.
"""
import argparse
import ast
from collections import Counter,defaultdict
import json
from pathlib import Path

from .audit_native_training_batch import require
from .native_search import write_json
from .staged_search import verify_evaluation
from .visual_task import file_sha256


def structural_changes(reference,candidate):
    def functions(text):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    a,b=functions(reference.source),functions(candidate.source)
    return {'changed_or_added_functions':sorted(k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)),
        'system_prompt_changed':reference.system_prompt!=candidate.system_prompt,
        'user_prompt_changed':reference.user_prompt!=candidate.user_prompt,
        'reference_budgets':{k:getattr(reference,k) for k in ('format_retry_budget','illegal_move_retry_budget','max_turns')},
        'candidate_budgets':{k:getattr(candidate,k) for k in ('format_retry_budget','illegal_move_retry_budget','max_turns')}}


def compare(plan_path,private,reference_path):
    from autoharness_chess_puzzle.harness import load_harness
    plan=json.loads(plan_path.read_text())
    summary,terminal=verify_evaluation(private,plan,plan_path)
    request=json.loads((private/'request.json').read_text())
    candidate_path=Path(request['kwargs']['harness_path'])
    require(reference_path.resolve()==Path(plan['incoming_harness']).resolve() and
            file_sha256(reference_path)==plan['incoming_harness_sha256'],'Reference must be the frozen incoming h0')
    reference,candidate=[load_harness(p) for p in (reference_path,candidate_path)]
    require(callable(reference.parse_action) and callable(candidate.parse_action),'Both parsers must be explicit')
    rows=[];episodes=[];artifacts={}
    for replica in (0,1):
        directory=private/f'replica-{replica}'
        generations=[json.loads(line) for line in (directory/'generations.jsonl').read_text().splitlines() if line]
        outputs=json.loads((directory/'native-outputs.json').read_text())
        ids=json.loads((directory/'ordered-ids.json').read_text())
        observed=Counter((e['raw_response'],e['parsed_action']) for o in outputs for e in o['harness_trace'] if e['actor']=='assistant')
        recomputed=Counter();flags=defaultdict(set)
        for call in generations:
            choice,=call['raw_response']['choices'];content=choice['message']['content']
            old,new=reference.parse_action(content),candidate.parse_action(content)
            recomputed[(content,new)]+=1
            observation=call['request']['messages'][-1]['content']
            old_legal=bool(reference.is_legal_action(observation,old))
            new_legal=bool(candidate.is_legal_action(observation,new))
            thinking_end=248069 in choice['token_ids']
            flags[content].add((choice['finish_reason'],thinking_end))
            rows.append({'replica':replica,'call':call['call'],'reference_action':old,'candidate_action':new,
                'same_action':old==new,'reference_legal':old_legal,'candidate_legal':new_legal,
                'same_legality':old_legal==new_legal,'finish_reason':choice['finish_reason'],
                'thinking_end_observed':thinking_end,'completion_tokens':call['raw_response']['usage']['completion_tokens']})
        require(recomputed==observed,'Candidate parser does not reproduce the audited actions')
        for key,output in zip(ids,outputs,strict=True):
            events=[e for e in output['harness_trace'] if e['actor']=='assistant']
            require(all(len(flags[e['raw_response']])==1 for e in events),'Ambiguous recorded termination flags')
            endings=[next(iter(flags[e['raw_response']])) for e in events]
            episodes.append({'puzzle_id':key,'solved':output['reward']==1.,
                'all_actions_same_with_reference_parser':all(reference.parse_action(e['raw_response'])==e['parsed_action'] for e in events),
                'all_calls_have_thinking_end':all(v[1] for v in endings),
                'any_length_stop':any(v[0]=='length' for v in endings),'stop_condition':output['stop_condition']})
        for name in ('generations.jsonl','native-outputs.json','ordered-ids.json'):
            artifacts[str(directory/name)]=file_sha256(directory/name)
    audit=json.loads((private/'audit.json').read_text())
    require(len(rows)==audit['calls'] and sum(r['completion_tokens'] for r in rows)==audit['generated_tokens'] and
            sum(e['solved'] for e in episodes)==summary['solved_examples'],'Diagnostic accounting differs')
    successes=[e for e in episodes if e['solved']]
    return {'kind':'fixed_reply_parser_comparison','status':'PASS','plan_sha256':file_sha256(plan_path),
        'evaluation_result_sha256':file_sha256(private/'result.json'),'reference_harness':str(reference_path),
        'reference_sha256':file_sha256(reference_path),'candidate_harness':str(candidate_path),
        'candidate_sha256':file_sha256(candidate_path),'structure':structural_changes(reference,candidate),
        'recorded_score':{'solved':summary['solved_examples'],'examples':summary['num_examples']},
        'calls':len(rows),'action_disagreements':sum(not r['same_action'] for r in rows),
        'legality_disagreements':sum(not r['same_legality'] for r in rows),
        'solved_with_same_reference_actions':sum(e['all_actions_same_with_reference_parser'] for e in successes),
        'solved_with_thinking_end_on_every_call':sum(e['all_calls_have_thinking_end'] for e in successes),
        'solved_with_any_length_stop':sum(e['any_length_stop'] for e in successes),
        'termination_counts':dict(Counter(e['stop_condition'] for e in episodes)),
        'rows':rows,'episodes':episodes,'allocation':terminal,'artifact_sha256':artifacts,
        'source_sha256':{n:file_sha256(Path(n)) for n in ('ours/recorded_parser_comparison.py',
            'ours/staged_search.py','upstream/WHALE/domains/chess_puzzles/autoharness_chess_puzzle/harness.py')},
        'new_model_calls':0,'new_api_calls':0,
        'limitations':['Actions are compared on fixed recorded replies, not replies from a counterfactual prompt.',
            'Observed agreement is not a proof of parser equivalence on every possible input.',
            'No heldout score, causal contribution percentage or new ablation score is inferred.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('plan','evaluation','reference','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();require(not args.output.exists(),'Preserve existing diagnostic report')
    result=compare(args.plan,args.evaluation,args.reference)
    write_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','episodes','artifact_sha256','source_sha256')}),flush=True)
