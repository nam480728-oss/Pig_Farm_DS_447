import argparse,csv,json,statistics
from pathlib import Path

def analyze(csv_path,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with Path(csv_path).open() as fp: rows=list(csv.DictReader(fp))
    rows=[{k.strip():float(v) for k,v in r.items() if k is not None and v.strip()} for r in rows]
    if not rows or 'metrics/mAP50-95(B)' not in rows[0]: raise ValueError('missing detection metrics')
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    best=max(rows,key=lambda r:r['metrics/mAP50-95(B)'])
    for title,cols in [('loss',[k for k in rows[0] if 'loss' in k]),('metrics',[k for k in rows[0] if k.startswith('metrics/')])]:
        fig,ax=plt.subplots(figsize=(10,5))
        for key in cols: ax.plot([r['epoch'] for r in rows],[r[key] for r in rows],label=key)
        ax.legend(); ax.grid(); ax.set_xlabel('epoch'); fig.tight_layout(); fig.savefig(out/(title+'.png'),dpi=140); plt.close(fig)
    result={'epochs':len(rows),'best_validation':best,'last_validation':rows[-1],
      'note':'Best validation row is descriptive; use actual best.pt for evaluation. No test claims or classification accuracy.'}
    (out/'analysis.json').write_text(json.dumps(result,indent=2))
    (out/'analysis.md').write_text('# Training analysis\nEpochs: '+str(len(rows))+'\n\nBest validation mAP50-95: '+str(best['metrics/mAP50-95(B)'])+'\n\nInspect loss curves, validation gap, PR/confusion plots and actual error examples. This is not test performance.\n')
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--csv',required=True); p.add_argument('--out',default='reports/training'); a=p.parse_args(); analyze(a.csv,a.out)
if __name__=='__main__': main()
