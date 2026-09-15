"""Host-owned original ChartQA open-answer scoring, separate from derived E1 data.

Protocol source: Masry et al., Findings of ACL2022, section5.1,
https://aclanthology.org/2022.findings-acl.177/ . The5% numeric rule and exact
text rule are retained. Percentages are fractions; zero requires numeric
equality. This zero convention is explicit because the Pix2Struct reference
implementation falls back to string equality for zero. No ANLS is used.
"""
from decimal import Decimal,InvalidOperation
from .chart_answer_protocol import parse_answer

PROTOCOL='chartqa_open_relaxed_v1'


def numeric(text):
    try:
        text=text.strip()
        value=Decimal(text[:-1])/100 if text.endswith('%') else Decimal(text)
        return value if value.is_finite() else None
    except (InvalidOperation,ValueError):return None


def relaxed_correct(prediction,truth):
    if not isinstance(truth,str) or not truth.strip():raise ValueError('Invalid original ChartQA answer')
    if not isinstance(prediction,str):raise TypeError('Expected shared parsed answer text')
    target,answer=numeric(truth),numeric(prediction)
    if target is not None and answer is not None:
        return abs(answer-target)<=abs(target)*Decimal('.05')
    return prediction.strip().casefold()==truth.strip().casefold()


def score_response(raw,truth):
    answer=parse_answer(raw)
    return {'raw_answer':raw,'committed_answer':answer,'correct':int(relaxed_correct(answer,truth)),
        'repository_exact_match':int(answer.strip()==truth.strip()),'protocol':PROTOCOL}


def aggregate(rows,expected_ids):
    records={r['sample_id']:r for r in rows}
    if len(records)!=len(rows) or set(records)!=set(expected_ids):
        raise ValueError('External predictions must cover every fixed sample exactly once')
    report={}
    for split in ('human','augmented'):
        selected=[r for r in rows if r['split']==split]
        if len(selected)!=256:raise ValueError('External subset split coverage differs')
        if any(type(r['correct']) is not int or r['correct'] not in (0,1) for r in selected):
            raise ValueError('Invalid shared external correctness')
        report[split]={'examples':256,'relaxed_accuracy':sum(r['correct'] for r in selected)/256,
            'repository_exact_match':sum(r['repository_exact_match'] for r in selected)/256}
    report['overall_relaxed_accuracy']=sum(report[s]['relaxed_accuracy'] for s in ('human','augmented'))/2
    report['name']='ChartQA fixed subset,256human+256augmented'
    report['official_full_benchmark_score']=False
    return report
