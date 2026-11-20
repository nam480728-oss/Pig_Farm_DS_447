import argparse,json
from pathlib import Path
from src.engine import Engine

def run(config,frames,out):
    engine=Engine(json.loads(Path(config).read_text()),out)
    base=Path(frames).resolve().parent
    for line in Path(frames).read_text().splitlines():
        if not line.strip(): continue
        f=json.loads(line)
        if f.get('image') and not Path(f['image']).is_absolute(): f['image']=str(base/f['image'])
        engine.process(f)
    return engine

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--frames',required=True); p.add_argument('--out',required=True)
    a=p.parse_args(); run(a.config,a.frames,a.out)
if __name__=='__main__': main()
