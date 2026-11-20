import argparse,json,hashlib
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--weights',required=True); p.add_argument('--data',default='config/data.yaml'); p.add_argument('--split',choices=['val','test'],default='val'); p.add_argument('--device',default='mps'); p.add_argument('--imgsz',type=int,default=960); p.add_argument('--out',default='runs/evaluation'); a=p.parse_args()
    from ultralytics import YOLO
    w=Path(a.weights)
    if not w.is_file(): raise FileNotFoundError('Use your actual trained checkpoint')
    m=YOLO(str(w)).val(data=a.data,split=a.split,device=a.device,imgsz=a.imgsz,plots=True,project=a.out,name=a.split,exist_ok=False)
    d={'weights_sha256':hashlib.sha256(w.read_bytes()).hexdigest(),'data_yaml_sha256':hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),'split':a.split,'imgsz':a.imgsz,'precision':float(m.box.mp),'recall':float(m.box.mr),'mAP50':float(m.box.map50),'mAP50_95':float(m.box.map),'speed_ms':m.speed}
    (Path(m.save_dir)/'metrics.json').write_text(json.dumps(d,indent=2))
if __name__=='__main__': main()
