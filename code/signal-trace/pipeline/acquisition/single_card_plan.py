"""Acquire a count-matched cohort for one pooled success/failure direction.

Physical cells control matching and permutation, not separate direction fits.
All screened responses are retained. Matching uses declared acquisition order,
never internal traces, and creates no failure/success pairwise target.
"""
from collections import defaultdict,Counter
from fractions import Fraction
import hashlib
import math
from pipeline.acquisition.prepare_a_cases import make


def cell(case):
    p=case['parameters'];return (p['m'],p['v'],p['drop'])


def matched_members(cases,roles,order):
    ids={c['id'] for c in cases};rank={cid:i for i,cid in enumerate(order)}
    if len(ids)!=len(cases) or len(rank)!=len(order) or not ids.issubset(rank):
        raise ValueError('Matching needs distinct cases and a complete frozen order')
    groups=defaultdict(lambda:defaultdict(list))
    for c in cases:groups[cell(c)][roles.get(c['id'],'unclassified')].append(c['id'])
    kept={'failure':set(),'natural_numerical_success':set()};counts=[]
    for key,group in sorted(groups.items()):
        n=min(len(group['failure']),len(group['natural_numerical_success']))
        for role in kept:
            kept[role].update(sorted(group[role],key=rank.__getitem__)[:n])
        counts.append(dict(cell=list(key),matched_per_class=n))
    result={role:sorted(members,key=rank.__getitem__) for role,members in kept.items()}
    result['unmatched']=sorted(ids-set.union(*kept.values()),key=rank.__getitem__)
    result['cells']=counts
    return result


def deficits(cases,roles):
    groups=defaultdict(list)
    for c in cases:groups[cell(c)].append(c['id'])
    rows=[]
    for key,ids in sorted(groups.items()):
        counts=Counter(roles.get(cid,'unclassified') for cid in ids)
        f=counts['failure'];s=counts['natural_numerical_success']
        quota=max(f,s) if f and s else min(f,s)
        rows.append(dict(cell=list(key),members=sorted(ids),failures=f,successes=s,
                         other=len(ids)-f-s,matched_quota=quota,
                         need_failure=max(0,quota-f),need_success=max(0,quota-s),
                         matching_supported=bool(f and s)))
    return rows


def build_plan(cases,roles,excluded_prompts):
    if len({c['id'] for c in cases})!=len(cases):raise ValueError('Duplicate source case')
    measured=deficits(cases,roles)
    measured.sort(key=lambda r:(-Fraction(r['failures'],len(r['members'])),tuple(r['cell'])))
    losses=sorted({c['parameters']['loss'] for c in cases})
    rises=sorted({c['parameters']['rise'] for c in cases})
    if any(x<=0 or int(x)!=x for x in losses+rises):
        raise ValueError('This declared grid needs positive integer source quantities')
    loss_quantum=math.gcd(*(int(x) for x in losses))
    rise_quantum=math.gcd(*(int(x) for x in rises))
    templates=sorted({c.get('template','channel') for c in cases})
    caps={c['max_tokens'] for c in cases}
    if len(caps)!=1:raise ValueError('Source generation caps disagree')
    excluded=set(excluded_prompts)|{c['prompt'] for c in cases};candidates=[]
    for row in measured:
        m,v,drop=row['cell']
        total=Fraction(m*v*v,2)+10*m*drop
        climb_quantum=10*m*rise_quantum
        local=[]
        for loss in range(loss_quantum,math.ceil(total),loss_quantum):
            available=total-loss
            # Every positive-grid YES height and the first strictly-NO height.
            # Higher NO heights are unbounded; no arbitrary upper height is used.
            max_rise=(available//climb_quantum+1)*rise_quantum
            for rise in range(rise_quantum,max_rise+1,rise_quantum):
                for template in templates:
                    cid=f'pooled-m{m}-v{v}-d{drop}-loss{loss}-h{rise}-{template}'
                    c=make(cid,f'pooled-m{m}-v{v}-d{drop}-loss{loss}-{template}',m,v,drop,loss,rise,template)
                    c['max_tokens']=next(iter(caps))
                    if c['prompt'] not in excluded:local.append(c)
        local.sort(key=lambda c:hashlib.sha256(c['prompt'].encode()).hexdigest())
        required=Fraction(0);rates={}
        for role,count,need in [('failure',row['failures'],row['need_failure']),
                                ('natural_numerical_success',row['successes'],row['need_success'])]:
            rate=Fraction(count,len(row['members']));rates[role]=dict(numerator=count,denominator=len(row['members']))
            if need:required+=need/rate
        rate_based=math.ceil(required)
        row.update(candidate_ids=[c['id'] for c in local],historical_rates=rates,
                   rate_based_draws=rate_based,
                   planned_draws=min(len(local),rate_based),
                   zero_rate_note=None if row['matching_supported'] else
                       'A class is absent: no finite yield estimate, quota zero, no screening; originals are surplus.',
                   available_before_loss=float(total))
        candidates.extend(local)
    return dict(cells=measured,candidates=candidates,
                domain=dict(source_loss=losses,source_rise=rises,templates=templates,
                            loss_quantum=loss_quantum,rise_quantum=rise_quantum),
                initial_match_deficit=sum(r['need_failure']+r['need_success'] for r in measured),
                conditional_matched_per_class=sum(r['matched_quota'] for r in measured),
                initial_screening_ceiling=sum(r['planned_draws'] for r in measured),
                scope='Historical mixed-card counts only; recompute quotas and prefix lengths after GPU0 originals, before new candidate outcomes. Pooled directions, exact cell matching, no power claim.')
