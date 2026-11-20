import argparse,json,hashlib,csv
from pathlib import Path
from PIL import Image,ImageDraw
import yaml

def load_data(path):
    path=Path(path).resolve(); cfg=yaml.safe_load(path.read_text()); root=Path(cfg['path']).expanduser()
    if not root.is_absolute(): root=path.parent/root
    if cfg.get('names') not in ({0:'pig'},['pig']): raise ValueError('require single class pig=0')
    return cfg,root.resolve()

def audit(config,out):
    cfg,root=load_data(config); errors=[]; warnings=[]; stats={}; manifest=[]; hashes={}; sizes=[]
    for split in ('train','val','test'):
        folder=root/cfg[split]
        if not folder.is_dir(): errors.append(f'Missing image directory: {folder}')
        images=sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in ('.jpg','.png','.jpeg','.bmp')) if folder.is_dir() else []
        nboxes=0; negative=0
        expected=set()
        for p in images:
            label=root/'labels'/split/p.relative_to(folder).with_suffix('.txt'); expected.add(label.resolve())
            try:
                with Image.open(p) as im: width,height=im.size; im.verify()
            except Exception as e: errors.append(f'Corrupt image {p}: {e}'); continue
            digest=hashlib.sha256(p.read_bytes()).hexdigest()
            if digest in hashes:
                (errors if hashes[digest]!=split else warnings).append(f'Duplicate content {p}; prior split={hashes[digest]}')
            hashes[digest]=split
            manifest.append({'split':split,'image':str(p.relative_to(root)),'sha256':digest,'width':width,'height':height})
            if not label.exists(): errors.append(f'Missing label {label}'); continue
            lines=[s for s in label.read_text().splitlines() if s.strip()]
            if not lines: negative+=1
            for line in lines:
                try:
                    import math
                    vals=list(map(float,line.split())); assert len(vals)==5 and all(map(math.isfinite,vals))
                    cl,x,y,w,h=vals
                    assert cl==0 and 0<w<=1 and 0<h<=1 and x-w/2>=-1e-6 and y-h/2>=-1e-6 and x+w/2<=1.000001 and y+h/2<=1.000001
                    nboxes+=1; sizes.append({'split':split,'width_norm':w,'height_norm':h,'area_norm':w*h})
                except Exception: errors.append(f'Invalid label {label}: {line}')
        for label in (root/'labels'/split).rglob('*.txt'):
            if label.resolve() not in expected: warnings.append(f'Orphan label: {label}')
        if not images: errors.append(f'Empty split {split}')
        stats[split]={'images':len(images),'boxes':nboxes,'empty_labels':negative}
    result={'stats':stats,'errors':errors,'warnings':warnings,'manifest':manifest,'note':'Empty labels require human verification; do not auto-resplit official data.'}
    if out:
        out=Path(out);out.mkdir(parents=True,exist_ok=True)
        (out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        for filename,rows in [('manifest.csv',manifest),('bbox_distribution.csv',sizes)]:
            if rows:
                with (out/filename).open('w',newline='') as fp:
                    writer=csv.DictWriter(fp,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return result

def visualize(config,out,limit=12):
    cfg,root=load_data(config);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for split in ('train','val','test'):
        folder=root/cfg[split]; images=sorted(p for p in folder.rglob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png'))[:limit]
        for i,p in enumerate(images):
            im=Image.open(p).convert('RGB'); d=ImageDraw.Draw(im);W,H=im.size
            label=root/'labels'/split/p.relative_to(folder).with_suffix('.txt')
            for line in label.read_text().splitlines():
                if not line.strip(): continue
                _,x,y,w,h=map(float,line.split()); d.rectangle(((x-w/2)*W,(y-h/2)*H,(x+w/2)*W,(y+h/2)*H),outline='red',width=3)
            im.save(out/f'{split}_{i:04d}.jpg')

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['audit','visualize']);p.add_argument('--data',default='config/data.yaml');p.add_argument('--out',required=True);a=p.parse_args()
    if a.action=='audit':
        r=audit(a.data,a.out);print(json.dumps({'stats':r['stats'],'errors':len(r['errors'])}));raise SystemExit(bool(r['errors']))
    visualize(a.data,a.out)
if __name__=='__main__': main()
