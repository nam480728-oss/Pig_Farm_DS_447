from pathlib import Path
import json,uuid
from PIL import Image,ImageDraw
from .contracts import validate_frame,validate_event
from .geometry import validate_polygon,inside,outside
from .counting import CountSession
from .escape import EscapeDetector

class Engine:
    """One instance per camera/pen/session. Explicitly configure source track ownership."""
    def __init__(self,config,out):
        self.cfg=config; self.polygon=validate_polygon(config['polygon'])
        self.size=config['size']; self.pen=config['pen']; self.out=Path(out); self.out.mkdir(parents=True,exist_ok=True)
        if len(self.size)!=2 or min(self.size)<=0 or any(not (0<=x<=self.size[0] and 0<=y<=self.size[1]) for x,y in self.polygon): raise ValueError('polygon outside image')
        self.counter=CountSession(config['registered'],**config.get('count',{}))
        self.escape=EscapeDetector(**config.get('escape',{})); self.last=-1; self.second=None; self.count_sent=False
        self.events=[]

    def evidence(self,f,ids=None):
        source=f.get('image')
        if not source or not Path(source).is_file(): raise FileNotFoundError('frame evidence image missing')
        im=Image.open(source).convert('RGB')
        if list(im.size)!=self.size: raise ValueError('evidence size mismatch')
        draw=ImageDraw.Draw(im); draw.line([tuple(p) for p in self.polygon]+[tuple(self.polygon[0])],fill='blue',width=3)
        for d in f['detections']:
            color='red' if ids and d.get('track_id') in ids else 'green'
            draw.rectangle(d['bbox'],outline=color,width=4)
            draw.text((d['bbox'][0],d['bbox'][1]),str(d.get('track_id')),fill=color)
        path=self.out/'evidence'/f'{uuid.uuid4().hex}.jpg'; path.parent.mkdir(exist_ok=True)
        im.save(path); return str(path)

    def emit(self,kind,wall,image,data):
        e=validate_event(dict(loai=kind,o=self.pen,thoi_gian=wall,
             muc={'dem_du':'thong_tin','dem_thieu':'canh_bao','lac_chuong':'khan_cap'}[kind],anh=image,du_lieu=data))
        with (self.out/'events.jsonl').open('a',encoding='utf-8') as fp: fp.write(json.dumps(e,ensure_ascii=False)+'\n')
        self.events.append(e); return e

    def process(self,f):
        validate_frame(f)
        if f['pen']!=self.pen or f['size']!=self.size: raise ValueError('camera/pen/size mismatch')
        if f['t']<=self.last: raise ValueError('nonmonotonic frame')
        self.last=f['t']; detections=[d for d in f['detections'] if d['confidence']>=self.cfg.get('confidence',.25)]
        # Optional owned track IDs prevent cross-pen false assignment. Single pen default.
        owned=self.cfg.get('owned_track_ids')
        tracked=[d for d in detections if d.get('track_id') is not None and (owned is None or d['track_id'] in owned)]
        ids=[d['track_id'] for d in tracked if outside(d['bbox'],self.polygon)]
        ready=self.escape.update(self.pen,f['t'],ids)
        result=[]
        if ready:
            image=self.evidence(f,ready)
            result.append(self.emit('lac_chuong',f['wall_time'],image,{'tracker_ids':ready,'video_seconds':f['t'],'hold_seconds':self.escape.hold}))
        sec=int(f['t'])
        if sec!=self.second and self.counter.phase!='done':
            self.second=sec
            n=sum(inside(((d['bbox'][0]+d['bbox'][2])/2,(d['bbox'][1]+d['bbox'][3])/2),self.polygon) for d in detections)
            image=self.evidence(f)
            state=self.counter.sample(sec,n,image)
            if state and state['status'] in ('dem_du','dem_thieu') and not self.count_sent:
                result.append(self.emit(state['status'],f['wall_time'],state['image'],{'so_dem':state['count'],'so_dang_ky':state['registered'],'phases':self.counter.phase_history}))
                self.count_sent=True
            if state:
                (self.out/'count_summary.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
        return result
