import random, sys, numpy as np, cv2, pandas as pd
from pathlib import Path
from ultralytics import YOLO
random.seed(0)
m = YOLO(r"D:\Pig_Farm\models\Pig_Seg\pig_seg.pt")
goc = Path(r"D:\Pig_Farm\data\data_cat_lung_tinh_kg\RGB_9579")
rows = []
for fold in sorted(goc.iterdir()):
    for con in sorted(fold.iterdir()):
        fs = sorted(con.glob("*.png")); fs = random.sample(fs, min(15, len(fs)))
        for f in fs:
            im = cv2.imread(str(f)); h, w = im.shape[:2]
            r = m.predict(im, imgsz=640, retina_masks=True, verbose=False, conf=0.25)[0]
            n = 0 if r.masks is None else len(r.masks)
            n5 = 0 if r.masks is None else int((r.boxes.conf > 0.5).sum())
            if n:
                k = int(r.boxes.conf.argmax()); mk = r.masks.data[k].cpu().numpy() > 0.5
                ys, xs = np.nonzero(mk); cx, cy = xs.mean() / w, ys.mean() / h
                cham_bien = bool(mk[0].any() or mk[-1].any() or mk[:, 0].any() or mk[:, -1].any())
                rows.append(dict(fold=fold.name, con=con.name, kg=float(con.name.split('_')[0]), anh=f.name, n=n, n05=n5,
                                 conf=float(r.boxes.conf[k]), area=int(mk.sum()), cx=cx, cy=cy, cham_bien=cham_bien))
            else:
                rows.append(dict(fold=fold.name, con=con.name, kg=float(con.name.split('_')[0]), anh=f.name, n=0, n05=0, conf=0, area=0, cx=np.nan, cy=np.nan, cham_bien=False))
d = pd.DataFrame(rows); d.to_csv(sys.argv[1], index=False)
print("so anh", len(d), "so con", d.con.nunique())
print("khong tim thay lon:", (d.n == 0).sum())
print("so mask (conf>=0.25) moi anh:", d.n.value_counts().sort_index().to_dict())
print("so mask conf>0.5 moi anh:", d.n05.value_counts().sort_index().to_dict())
print("conf mask chinh: mean %.3f min %.3f p5 %.3f" % (d.conf.mean(), d.conf.min(), d.conf.quantile(.05)))
print("mask chinh gan tam (|cx-.5|<.2 & |cy-.5|<.25): %.3f" % (((d.cx-.5).abs()<.2)&((d.cy-.5).abs()<.25)).mean())
print("mask cham bien anh: %.3f" % d.cham_bien.mean())
g = d[d.n > 0].groupby("con").agg(kg=("kg","first"), area_med=("area","median"), cv=("area", lambda x: x.std()/x.mean()))
print("CV dien tich trong 1 con: median %.3f p90 %.3f max %.3f" % (g.cv.median(), g.cv.quantile(.9), g.cv.max()))
print("tuong quan Pearson dien tich trung vi - kg: %.3f" % g[["area_med","kg"]].corr().iloc[0,1])
print("tuong quan Spearman: %.3f" % g[["area_med","kg"]].corr("spearman").iloc[0,1])
