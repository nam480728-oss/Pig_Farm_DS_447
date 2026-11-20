import argparse,csv
from pathlib import Path
import yaml

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--metadata',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    from scripts.data_tools import load_data
    cfg,root=load_data(a.data);out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    folder=(root/cfg['test']).resolve(); seen=set(); groups={'day':[],'night':[]}
    expected={p.resolve() for p in folder.rglob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp')}
    for row in csv.DictReader(Path(a.metadata).open()):
        im=(root/row['image']).resolve();tag=row['lighting']
        if tag not in groups or im not in expected or im in seen: raise ValueError('Invalid/duplicate metadata image')
        seen.add(im);groups[tag].append(str(im))
    if seen!=expected: raise ValueError('Metadata must cover official test images exactly')
    for name,images in groups.items():
        if not images: raise ValueError('Empty lighting subset')
        listing=out/(name+'.txt');listing.write_text('\n'.join(images)+'\n')
        subset=dict(cfg,path=str(root),test=str(listing))
        (out/(name+'.yaml')).write_text(yaml.safe_dump(subset))
if __name__=='__main__':main()
