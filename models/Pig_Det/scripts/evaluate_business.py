import argparse,json,csv
from pathlib import Path
from src.metrics import event_metrics,count_metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--truth',required=True);p.add_argument('--pred',required=True);p.add_argument('--counts');p.add_argument('--tolerance',type=float,default=2);p.add_argument('--out',required=True);a=p.parse_args()
    result={'events':event_metrics(json.loads(Path(a.truth).read_text()),json.loads(Path(a.pred).read_text()),a.tolerance)}
    if a.counts:result['counts']=count_metrics([(int(r['truth']),int(r['prediction'])) for r in csv.DictReader(Path(a.counts).open())])
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
