import argparse,json,hashlib,shutil
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--weights',required=True); p.add_argument('--metrics',required=True); p.add_argument('--out',default='models/handover'); a=p.parse_args()
    w=Path(a.weights); m=json.loads(Path(a.metrics).read_text()); digest=hashlib.sha256(w.read_bytes()).hexdigest()
    if m['weights_sha256']!=digest or m['split']!='test': raise ValueError('Checkpoint hash/test split mismatch')
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(w,out/'pig_set.pt'); shutil.copy2(a.metrics,out/'metrics.json')
    card={'class_names':{0:'pig'},'sha256':digest,'metrics':m,'target_AP_definition':'Needs team confirmation; report both AP50 and AP50-95','limitations':['Occlusion','Night domain shift','No persistent animal identification']}
    (out/'model_card.json').write_text(json.dumps(card,indent=2))
if __name__=='__main__': main()
