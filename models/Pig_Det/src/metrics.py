def event_metrics(truth,predictions,tolerance=2):
    """Greedy earliest chronological one-to-one within same pen/type, absolute tolerance.
    Explicit convention; label times should represent expected alert (crossing+hold).
    """
    if tolerance<0: raise ValueError('negative tolerance')
    used=set(); matches=[]
    for p in sorted(predictions,key=lambda x:x['t']):
        choices=[(abs(p['t']-g['t']),i) for i,g in enumerate(truth) if i not in used and g['pen']==p['pen'] and g['kind']==p['kind'] and abs(p['t']-g['t'])<=tolerance]
        if choices:
            _,i=min(choices); used.add(i); matches.append(p['t']-truth[i]['t'])
    tp=len(matches); fp=len(predictions)-tp; fn=len(truth)-tp
    return dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,mean_signed_delay=sum(matches)/tp if tp else None)

def count_metrics(pairs):
    if not pairs: return {'sessions':0,'exact_rate':None,'mae':None}
    return {'sessions':len(pairs),'exact_rate':sum(a==b for a,b in pairs)/len(pairs),'mae':sum(abs(a-b) for a,b in pairs)/len(pairs)}
