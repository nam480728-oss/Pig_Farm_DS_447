"""Cắt ảnh từng con lợn để làm dữ liệu phân loại tư thế ĐỨNG / KHÔNG ĐỨNG.

Nguồn:
  - PigDetect (data_goc/detection/PigDetect, có hộp vẽ tay): 2.000 con, 500 con / trại x 4 trại
    (alameer, bergamini, danuma, psota), lấy cả ảnh dev và test. Trại thiếu (bergamini chỉ có 186 ảnh)
    thì phần thiếu được bù bằng trại sau.
  - RGB_9579 (data/data_cat_lung_tinh_kg/RGB_9579): 1.500 con, chia đều 73 con lợn,
    hộp cắt lấy từ dac_trung.csv (pig_seg 2 lượt, đã nới 12 %).

Bộ lọc hộp PigDetect (vì hệ thống chỉ cân con đứng riêng rẽ, nhìn rõ):
  - không chạm mép ảnh (cách mép >= 3 px)
  - cạnh ngắn >= 4 % cạnh ngắn của ảnh (bỏ con quá nhỏ / ở xa)
  - phần bị hộp khác che <= 25 % diện tích hộp
  - tối đa 3 hộp / ảnh gốc để trải đều trên nhiều khung hình

Ảnh cắt (nới 10 % mỗi cạnh) ghi vào data/data_posture/_chua_xep/, kèm data/data_posture/danh_sach.csv.
(File code này nằm ở data/data_posture_final_ok/, dữ liệu nhãn nằm ở data/data_posture/.)
Tên file:  pd_<trại>_<tên ảnh gốc>_h<số hộp>.jpg   |   rgb_<fold>_<con>_<tên ảnh gốc>.jpg
Chạy:  python data/data_posture_final_ok/cat_anh_tu_the.py
"""
import glob
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parents[2]
PD = GOC / "data_goc" / "detection" / "PigDetect"
RGB = GOC / "data" / "data_cat_lung_tinh_kg" / "RGB_9579"
DAC_TRUNG = GOC / "data" / "data_cat_lung_tinh_kg" / "dac_trung.csv"
NHAN = GOC / "data" / "data_posture"            # nơi chứa dung/, khong_dung/ và danh_sach.csv
RA = NHAN / "_chua_xep"
SO_PD_MOI_TRAI = 500
SO_RGB = 1500
NOI = 0.10
SEED = 42


def doc_anh(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def ghi_anh(p, anh):
    ok, buf = cv2.imencode(".jpg", anh, [cv2.IMWRITE_JPEG_QUALITY, 95])
    buf.tofile(str(p))


def cat(anh, x1, y1, x2, y2, noi):
    h, w = anh.shape[:2]
    dx, dy = (x2 - x1) * noi, (y2 - y1) * noi
    a, b = int(max(0, x1 - dx)), int(max(0, y1 - dy))
    c, d = int(min(w, x2 + dx)), int(min(h, y2 + dy))
    return anh[b:d, a:c]


def hop_pigdetect(f_anh, w, h):
    nhan = Path(str(f_anh).replace("images", "labels")).with_suffix(".txt")
    hop = []
    if nhan.exists():
        for ln in nhan.read_text().split("\n"):
            p = ln.split()
            if len(p) == 5:
                cx, cy, bw, bh = map(float, p[1:])
                hop.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return hop


def bi_che(i, hop):
    """Tỉ lệ diện tích hộp i bị các hộp khác đè lên (lấy lớn nhất)."""
    x1, y1, x2, y2 = hop[i]
    s = (x2 - x1) * (y2 - y1)
    lon_nhat = 0.0
    for j, (a, b, c, d) in enumerate(hop):
        if j != i:
            g = max(0, min(x2, c) - max(x1, a)) * max(0, min(y2, d) - max(y1, b))
            lon_nhat = max(lon_nhat, g / s if s else 0)
    return lon_nhat


def main():
    random.seed(SEED)
    RA.mkdir(exist_ok=True)
    dong = []

    # ---------- PigDetect ----------
    thieu = 0   # trại nào không đủ 500 con thì phần thiếu chuyển sang trại sau (danuma nhiều ảnh nhất, xếp cuối)
    for trai in ["bergamini", "alameer", "psota", "danuma"]:
        muc_tieu = SO_PD_MOI_TRAI + thieu
        anh_trai = sorted(glob.glob(str(PD / "dev" / "images" / f"{trai}_*"))) + \
                   sorted(glob.glob(str(PD / "test" / "images" / f"{trai}_*")))
        random.shuffle(anh_trai)
        da_lay = 0
        for f in anh_trai:
            if da_lay >= muc_tieu:
                break
            anh = doc_anh(f)
            h, w = anh.shape[:2]
            hop = hop_pigdetect(f, w, h)
            hop_ok = [i for i, (x1, y1, x2, y2) in enumerate(hop)
                      if x1 >= 3 and y1 >= 3 and x2 <= w - 3 and y2 <= h - 3
                      and min(x2 - x1, y2 - y1) >= 0.04 * min(w, h)
                      and bi_che(i, hop) <= 0.25]
            for i in random.sample(hop_ok, min(3, len(hop_ok))):
                if da_lay >= muc_tieu:
                    break
                ten = f"pd_{trai}_{Path(f).stem}_h{i:02d}.jpg"
                ghi_anh(RA / ten, cat(anh, *hop[i], NOI))
                dong.append({"file": ten, "nguon": "pigdetect", "trai": trai, "anh_goc": Path(f).name,
                             "tap_goc": Path(f).parents[1].name, "hop": i})
                da_lay += 1
        thieu = muc_tieu - da_lay
        print(f"PigDetect {trai}: {da_lay} con")

    # ---------- RGB_9579 ----------
    d = pd.read_csv(DAC_TRUNG).dropna(subset=["x1"])
    so_moi_con = SO_RGB // d.con.nunique()          # 20 con/lợn x 73 = 1.460
    chon = d.groupby("con", group_keys=False).apply(lambda g: g.sample(min(len(g), so_moi_con), random_state=SEED))
    con_lai = d.drop(chon.index).sample(SO_RGB - len(chon), random_state=SEED)   # bù cho đủ 1.500
    chon = pd.concat([chon, con_lai])
    for _, r in chon.iterrows():
        anh = doc_anh(RGB / r.anh)
        fold, con, ten_goc = r.anh.split("/")
        ten = f"rgb_{fold}_{con}_{Path(ten_goc).stem}.jpg"
        ghi_anh(RA / ten, cat(anh, r.x1, r.y1, r.x2, r.y2, 0.0))   # hộp đã nới 12 % sẵn
        dong.append({"file": ten, "nguon": "rgb9579", "trai": "rgb9579", "anh_goc": r.anh, "tap_goc": fold, "hop": 0})
    print(f"RGB_9579: {len(chon)} con")

    pd.DataFrame(dong).to_csv(NHAN / "danh_sach.csv", index=False)
    print("Tong:", len(dong))


if __name__ == "__main__":
    main()
