"""Resumable, private thread triage acceptance eval.

Run: python evals/triage.py run [--id t01] [--method candidate]
     python evals/triage.py summary
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sieve.cache import NullCache
from sieve.client import build_client
from sieve.jev import JevScorer, QuestionSpec, Usage
from sieve.triage import request_blocks, score_thread
from sieve.validate import validate_response

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / 'evals/triage/labels.json'
RAW = ROOT / 'evals/raw/triage'
METHODS = ('baseline', 'whole_window', 'candidate')
DEFAULT = (0.5, 0.5, 0.5)
GRID = tuple(i / 20 for i in range(21))
SPECS = (
    QuestionSpec('items', 'Did the assistant in {ref} ask Derick a question that no later human turn answered?', 'There is an unanswered assistant question for Derick.', 'There is no unanswered assistant question for Derick.'),
    QuestionSpec('items', 'Does the latest assistant statement in {ref} say progress is waiting for Derick?', 'The latest statement says progress waits for Derick.', 'The latest statement does not say progress waits for Derick.'),
    QuestionSpec('items', 'Does the latest progress evidence in {ref} identify exactly one remaining action before the stated acceptance is met?', 'Exactly one action remains before acceptance.', 'The number of remaining actions is not exactly one or is unclear.'),
)
MARKER = re.compile(r'\b(?:choose|confirm|decide|let me know|should i|do you want|which|tell me|approve|pick|please|can you|could you|would you)\b', re.I)


def cases():
    rows = json.loads(LABELS.read_text())
    if len(rows) != 30:
        raise ValueError('expected 30 snapshots')
    return [(f't{i:02d}', row) for i, row in enumerate(rows, 1)]


def snapshot(row):
    return json.loads((ROOT / row['events_file']).read_text())


def output_tokens(payload):
    return math.ceil(len(json.dumps(payload, ensure_ascii=False, separators=(',', ':'))) / 4)


def chief_tokens(snap):
    return math.ceil(len(json.dumps(snap['events'], ensure_ascii=False)) / 4)


def baseline(snap):
    events = snap['events']
    last = next((e for e in reversed(events) if e.get('kind') == 'text'), None)
    blocked = bool(last and last.get('role') == 'assistant' and snap.get('last_status') != 'running' and
                   ('?' in last.get('text', '') or MARKER.search(last.get('text', ''))))
    return {'bucket': 'blocked_on_derick' if blocked else 'nothing', 'one_step_left': 'unknown', 'requests': [],
            'usage': {'input_tokens': 0, 'output_tokens': 0, 'requests': 0, 'cost_usd': 0}}


def window(snap, limit=5000):
    turns = [e for e in snap['events'] if e.get('role') in ('human', 'assistant') and e.get('kind') == 'text']
    selected = []
    size = 0
    for e in reversed(turns):
        width = math.ceil(len(e.get('text', '')) / 4)
        if selected and size + width > limit:
            break
        selected.append({'id': e['id'], 'role': e['role'], 'text': e['text'][-limit * 4:]})
        size += min(width, limit)
    state = {'title': snap['title'], 'first_prompt': snap['first_prompt'],
             'human_amendments': snap['human_amendments'], 'turns': list(reversed(selected))}
    while len(json.dumps(state, ensure_ascii=False)) / 4 > limit and len(state['turns']) > 1:
        state['turns'].pop(0)
    if len(json.dumps(state, ensure_ascii=False)) / 4 > limit and state['turns']:
        excess = math.ceil(len(json.dumps(state, ensure_ascii=False)) - limit * 4)
        state['turns'][0]['text'] = state['turns'][0]['text'][excess:]
    return state


async def whole_window(snap):
    client = build_client()
    scorer = JevScorer(client, cache=NullCache(), budget_usd=0.50)
    state = window(snap)
    try:
        questions = {f'q{i}': spec.question_for(0) for i, spec in enumerate(SPECS)}
        response = await scorer.client.system_one({'question': '', 'items': [state]}, questions, model=scorer.model)
        values = [validate_response(response, questions)[f'q{i}'] for i in range(3)]
        scorer.usage.add(Usage(int(response.usage.input_tokens or 0), int(response.usage.output_tokens or 0), 1))
    finally:
        await client.aclose()
    return {'model': response.model, 'probabilities': dict(zip(('unanswered', 'blocking', 'one_step_left'), values)),
            'usage': scorer.usage.as_dict(), 'budget_exhausted': any(v is None for v in values)}


async def candidate(snap):
    client = build_client()
    try:
        return await score_thread(snap['events'], {'title': snap['title'], 'first_prompt': snap['first_prompt'],
                                  'human_amendments': snap['human_amendments']}, status=snap.get('last_status'),
                                  client=client, cache=NullCache(), budget=0.50, request_threshold=0.0)
    finally:
        await client.aclose()


def candidate_bucket(result, snap, thresholds):
    request_t, answered_t, blocking_t = thresholds
    requests = [r for r in result['requests'] if r['probabilities']['request'] is not None and r['probabilities']['request'] >= request_t]
    blocked = False
    unresolved = False
    for r in requests:
        p = r['probabilities']
        answered = p['answered'] is not None and p['answered'] >= answered_t
        withdrawn = p['withdrawn'] is not None and p['withdrawn'] >= answered_t
        if answered or withdrawn:
            continue
        recent = snap.get('last_status') == 'running' and bool(r['coverage']['assistant_windows'])
        if request_blocks(p['blocking'], blocking_t, approval_required=r.get('approval_required', False),
                          recent_progress=recent) and not result['coverage']['truncated']:
            blocked = True
        else:
            unresolved = True
    one = result['buckets']['one_step_left']
    one_p = one.get('probability') if isinstance(one, dict) else None
    if blocked:
        return 'blocked_on_derick'
    if one_p is not None and one_p >= 0.5:
        return 'one_step_left'
    if result['coverage']['truncated'] or result['usage']['budget_exhausted'] or unresolved or (isinstance(one, dict) and one_p is None):
        return 'unknown'
    return 'nothing'


def whole_bucket(result, thresholds):
    request_t, _, blocking_t = thresholds
    p = result['probabilities']
    if p['unanswered'] is not None and p['blocking'] is not None and p['unanswered'] >= request_t and p['blocking'] >= blocking_t:
        return 'blocked_on_derick'
    if p['one_step_left'] is not None and p['one_step_left'] >= 0.5:
        return 'one_step_left'
    if any(v is None for v in p.values()):
        return 'unknown'
    return 'nothing'


def bucket(method, result, snap, thresholds):
    return (result['bucket'] if method == 'baseline' else whole_bucket(result, thresholds) if method == 'whole_window'
            else candidate_bucket(result, snap, thresholds))


def truth(row):
    if row['blocked_on_derick'] is True:
        return 'blocked_on_derick'
    if row['one_step_left'] is True:
        return 'one_step_left'
    if row['nothing_pending'] is True:
        return 'nothing'
    return 'unknown'


_MATCH_CACHE = {}


def matches(result, row):
    """Assign at most one scored span to each label, within its event ID."""
    cache_key = (id(result), id(row))
    if cache_key in _MATCH_CACHE:
        return _MATCH_CACHE[cache_key]
    scored = result.get('requests', [])
    options = []
    for li, label in enumerate(row['candidates']):
        for pi, request in enumerate(scored):
            if label['event_id'] == request['event_id']:
                similarity = difflib.SequenceMatcher(None, label['excerpt'], request['span_text']).ratio()
                options.append((similarity, li, pi))
    assigned = {}
    used = set()
    for _, li, pi in sorted(options, reverse=True):
        if li not in assigned and pi not in used:
            assigned[li] = scored[pi]
            used.add(pi)
    _MATCH_CACHE[cache_key] = assigned
    return assigned


def stale(method, result, row, snap, thresholds):
    stale_labels = [c for c in row['candidates'] if c['answered_by_event_id'] or c['withdrawn'] or c['optional_offer']]
    if method != 'candidate':
        n = int(bucket(method, result, snap, thresholds) == 'blocked_on_derick' and bool(stale_labels))
        return n, n
    request_t, answered_t, _ = thresholds
    n = 0
    assigned = matches(result, row)
    for c in stale_labels:
        for r in [assigned.get(row['candidates'].index(c))]:
            if r is None:
                continue
            p = r['probabilities']
            if p['request'] is not None and p['request'] >= request_t and not (p['answered'] is not None and p['answered'] >= answered_t) and not (p['withdrawn'] is not None and p['withdrawn'] >= answered_t):
                n += 1
                break
    return int(n > 0), n


def metrics(rows, method, thresholds):
    buckets = [(row, bucket(method, out, snap, thresholds), stale(method, out, row, snap, thresholds)) for _, row, snap, out in rows]
    return {'missed': sum(r['blocked_on_derick'] is True and b != 'blocked_on_derick' for r,b,_ in buckets),
            'stale_threads': sum(s[0] for _,_,s in buckets), 'stale_candidates': sum(s[1] for _,_,s in buckets),
            'unknown': sum(b == 'unknown' for _,b,_ in buckets),
            'unknown_correct': sum(b == 'unknown' and truth(r) == 'unknown' for r,b,_ in buckets),
            'unknown_overcautious': sum(b == 'unknown' and truth(r) != 'unknown' for r,b,_ in buckets),
            'accuracy': sum(b == truth(r) for r,b,_ in buckets), 'n': len(rows)}


def fit(rows, method):
    if method == 'baseline':
        return DEFAULT
    return min(((r,a,b) for r in GRID for a in (GRID if method == 'candidate' else (0.5,)) for b in GRID),
               key=lambda t: (lambda m: (3*m['missed'] + m['stale_threads'] + m['unknown_overcautious'],
                                         m['missed'], m['stale_threads'], sum(abs(x-.5) for x in t), t))(metrics(rows, method, t)))


def load():
    rows = []
    for ident, row in cases():
        snap = snapshot(row)
        for method in METHODS:
            path = RAW / method / f'{ident}.json'
            if not path.exists():
                raise SystemExit(f'missing {path}; run first')
            output = json.loads(path.read_text())
            if output['agent_id'] != row['agent_id'] or output['last_event_id'] != snap['events'][-1]['id']:
                raise SystemExit(f'changed snapshot: {ident}')
            rows.append((ident, row, snap, method, output))
    return rows


def summary():
    rows = load()
    for method in METHODS:
        tune = [(i,r,s,o) for i,r,s,m,o in rows if m == method and int(i[1:]) % 2]
        test = [(i,r,s,o) for i,r,s,m,o in rows if m == method and not int(i[1:]) % 2]
        chosen = fit(tune, method)
        print(method, 'chosen', chosen)
        for name, subset, t in (('tune chosen',tune,chosen),('test chosen',test,chosen),('tune default',tune,DEFAULT),('test default',test,DEFAULT)):
            print(name, metrics(subset,method,t))
        usage = [o['usage'] for _,_,_,o in tune+test]
        chief = sum(o['chief_tokens'] for _,_,_,o in tune+test)
        out = sum(o['output_tokens_est'] for _,_,_,o in tune+test)
        print('tokens',chief,out,chief-out,'jev_input',sum(u['input_tokens'] for u in usage),'requests',sum(u['requests'] for u in usage),'cost',sum(u['cost_usd'] for u in usage),'wall',sum(o['wall_seconds'] for _,_,_,o in tune+test))
        if method == 'candidate':
            for name, subset in (('tune',tune),('test',test)):
                for tag,t in (('chosen',chosen),('default',DEFAULT)):
                    tp=fp=fn=ans_ok=ans_n=0
                    for _,r,_,o in subset:
                        assigned=matches(o,r)
                        for ci,c in enumerate(r['candidates']):
                            hit=assigned.get(ci)
                            found=bool(hit and hit['probabilities']['request'] is not None and hit['probabilities']['request'] >= t[0])
                            tp += bool(found and c['is_request_for_human'])
                            fp += bool(found and not c['is_request_for_human'])
                            fn += bool(not found and c['is_request_for_human'])
                            if hit:
                                ans=hit['probabilities']['answered']
                                ans_ok += (ans is not None and ans >= t[1]) == (c['answered_by_event_id'] is not None)
                                ans_n += 1
                    print(name,tag,'request tp fp fn',tp,fp,fn,'answered',ans_ok,ans_n)
    print('per thread')
    for i,r,s,m,o in rows:
        print(i,m,'truth',truth(r),'default',bucket(m,o,s,DEFAULT),'output',o['output_tokens_est'],'input',o['usage']['input_tokens'],'cost',o['usage']['cost_usd'],'wall',round(o['wall_seconds'],2))


async def run(args):
    for ident, row in cases():
        if args.id and ident != args.id:
            continue
        snap = snapshot(row)
        for method in METHODS:
            if args.method and method != args.method:
                continue
            path = RAW / method / f'{ident}.json'
            if path.exists() and not args.force:
                print('skip',ident,method,flush=True)
                continue
            print('run',ident,method,flush=True)
            started=time.monotonic()
            result = baseline(snap) if method == 'baseline' else await whole_window(snap) if method == 'whole_window' else await candidate(snap)
            elapsed=time.monotonic()-started
            record={'id': ident, 'agent_id': row['agent_id'], 'last_event_id': snap['events'][-1]['id'],
                    'method': method, 'chief_tokens': chief_tokens(snap), 'output_tokens_est': output_tokens(result),
                    'wall_seconds': elapsed, 'usage': result['usage'], **result}
            path.parent.mkdir(parents=True,exist_ok=True)
            tmp=path.with_suffix('.tmp')
            tmp.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
            tmp.replace(path)
            print('done',ident,method,record['usage']['requests'],'requests',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('run');p.add_argument('--id',choices=[f't{i:02d}' for i in range(1,31)]);p.add_argument('--method',choices=METHODS);p.add_argument('--force',action='store_true')
    sub.add_parser('summary')
    args=parser.parse_args()
    asyncio.run(run(args)) if args.command=='run' else summary()


if __name__=='__main__':
    main()
