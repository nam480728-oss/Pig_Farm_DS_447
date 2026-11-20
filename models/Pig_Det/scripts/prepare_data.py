"""Validate an existing YOLO-format dataset before training; never change splits."""
import argparse,json
from scripts.data_tools import audit

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='config/data.yaml')
    p.add_argument('--dataset',help='Optional path to pig_det dataset')
    p.add_argument('--out',default='reports/data_audit')
    a=p.parse_args()
    from scripts.dataset_layout import configure
    if a.data == 'config/data.yaml': configure(explicit=a.dataset)
    r=audit(a.data,a.out)
    print(json.dumps({'stats':r['stats'],'errors':r['errors'][:20], 'warnings_count':len(r['warnings'])},indent=2,ensure_ascii=False))
    if r['errors']: raise SystemExit(1)

if __name__=='__main__':main()
