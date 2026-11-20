from datetime import datetime
import math

def validate_frame(f):
    t=f['t']; w,h=f['size']
    if not isinstance(t,(int,float)) or not math.isfinite(t) or t<0 or w<=0 or h<=0: raise ValueError('invalid frame')
    if not isinstance(f['pen'],str) or not f['pen']: raise ValueError('invalid pen')
    dt=datetime.fromisoformat(f['wall_time'])
    if dt.tzinfo is None: raise ValueError('wall_time needs timezone')
    ids=[]
    for d in f['detections']:
        x1,y1,x2,y2=d['bbox']
        if not all(math.isfinite(float(v)) for v in d['bbox']) or not (0<=x1<x2<=w and 0<=y1<y2<=h): raise ValueError('bbox out of bounds')
        if not 0<=d['confidence']<=1 or d['class_id']!=0: raise ValueError('invalid detection')
        ident=d.get('track_id')
        if ident is not None:
            if type(ident)!=int or ident<0: raise ValueError('invalid track id')
            ids.append(ident)
    if len(set(ids))!=len(ids): raise ValueError('duplicate track IDs')
    return f

def validate_event(e):
    if set(e)!=set(['loai','o','thoi_gian','muc','anh','du_lieu']): raise ValueError('event fields')
    levels={'dem_du':'thong_tin','dem_thieu':'canh_bao','lac_chuong':'khan_cap'}
    if e['loai'] not in levels or levels[e['loai']]!=e['muc']: raise ValueError('event level')
    if not e['anh'] or not isinstance(e['du_lieu'],dict): raise ValueError('evidence required')
    if datetime.fromisoformat(e['thoi_gian']).tzinfo is None: raise ValueError('event timezone')
    return e
