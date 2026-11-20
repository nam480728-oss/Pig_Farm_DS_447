import argparse,json,math
from pathlib import Path
from datetime import datetime,timedelta

def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--source',required=True);p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cpu');p.add_argument('--start-time',required=True);a=p.parse_args()
    import cv2
    from ultralytics import YOLO
    from src.engine import Engine
    from src.box_policy import expand_xyxy
    # Offline file only: no misleading wall-clock duration from RTSP processing speed.
    if not Path(a.source).is_file():raise ValueError('Use recorded video file; live ingestion belongs to shared pipeline')
    if not Path(a.weights).is_file():raise FileNotFoundError('Trained weights required')
    start=datetime.fromisoformat(a.start_time)
    if start.tzinfo is None:raise ValueError('start-time needs timezone')
    cap=cv2.VideoCapture(a.source);fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
    if not math.isfinite(fps) or fps<=0:raise ValueError('Invalid video FPS')
    cfg=json.loads(Path(a.config).read_text());engine=Engine(cfg,a.out)
    spool=Path(a.out)/'current_frame.jpg';index=0
    for r in YOLO(a.weights).track(source=a.source,stream=True,tracker='botsort.yaml',device=a.device,conf=cfg.get('confidence',.25),verbose=False):
        h,w=r.orig_img.shape[:2]
        if [w,h]!=cfg['size']:raise ValueError('Polygon/config size must match source video')
        if not cv2.imwrite(str(spool),r.orig_img):raise IOError('Cannot save frame evidence')
        det=[]
        if r.boxes is not None:
            boxes=r.boxes
            ids=boxes.id.cpu().tolist() if boxes.id is not None else [None]*len(boxes)
            for box,conf,cls,ident in zip(boxes.xyxy.cpu().tolist(),boxes.conf.cpu().tolist(),boxes.cls.cpu().tolist(),ids):
                if int(cls)==0:det.append(dict(bbox=expand_xyxy(box,w,h),confidence=float(conf),class_id=0,track_id=None if ident is None else int(ident)))
        t=index/fps;engine.process(dict(t=t,size=[w,h],pen=cfg['pen'],wall_time=(start+timedelta(seconds=t)).isoformat(),detections=det,image=str(spool)))
        index+=1
    if spool.exists():spool.unlink()
    if engine.counter.phase!='done':
        (Path(a.out)/'incomplete_session.json').write_text(json.dumps({'reason':'video ended before count/recount finished','phase':engine.counter.phase}))
if __name__=='__main__':main()
