"""Chia dữ liệu tư thế đứng / không đứng thành train / val / test = 70 / 20 / 10 và chuẩn hóa ảnh.

Đọc (KHÔNG sửa): data/data_posture/dung/, khong_dung/ (nhãn = tên thư mục, đã kiểm tra thủ công)
Ghi:             data/data_posture_final/{train,val,test}/{dung,khong_dung}/*.jpg  + split.csv

1. Chọn ảnh: toàn bộ ảnh PigDetect + SO_RGB ảnh RGB_9579 (lấy đều theo con lợn) để 2 lớp cân bằng
   và ảnh RGB_9579 (1 trại, nền xi măng) không chiếm đa số lớp "dung".
2. Chia THEO NHÓM để ảnh gần giống nhau không nằm ở 2 tập (tránh test "học thuộc"):
     alameer   -> cùng video (phần tên trước "--")
     bergamini -> cùng clip  (PIGSddmmyy__xxxxxx)
     danuma, psota -> khối 25 số thứ tự khung liên tiếp (tên chỉ có số thứ tự)
     RGB_9579  -> cùng con lợn
   Mỗi trại chia riêng 70/20/10 -> tập nào cũng có đủ 5 nguồn.
3. Chuẩn hóa ảnh (tu_the_chung.py, cùng thư mục):
     - RGB_9579 thu nhỏ trước cho cạnh dài = 256 px (cỡ ảnh PigDetect) -> không học "ảnh nét = đứng"
     - xoay cho thân nằm ngang -> không học "ảnh dọc = đứng"
     - thêm viền xám thành 224 x 224 (giữ trọn con lợn, không cắt chân)
Chạy: python data/data_posture_final/chia_du_lieu.py
"""
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))   # tu_the_chung.py nằm cùng thư mục
from tu_the_chung import chuan_hoa_anh  # noqa: E402

THU_MUC = GOC / "data" / "data_posture"          # nguồn: dung/, khong_dung/, danh_sach*.csv (không sửa)
RA = Path(__file__).resolve().parent             # đích: data/data_posture_final/{train,val,test}
LOP = ["dung", "khong_dung"]
TI_LE = {"train": 0.7, "val": 0.2, "test": 0.1}
SO_RGB = 350                 # số ảnh RGB_9579 dùng (lớp dung)
CANH_DAI_RGB = 256           # thu nhỏ RGB_9579 về cỡ ảnh PigDetect
KHOI_KHUNG = 25              # danuma / psota: 25 số thứ tự liên tiếp = 1 nhóm
SEED = 42


def doc_anh(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def ghi_anh(p, anh):
    cv2.imencode(".jpg", anh, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(str(p))


def nhom(trai, anh_goc):
    """Khóa nhóm: các ảnh cùng khóa luôn vào cùng 1 tập."""
    if trai == "rgb9579":
        return "rgb_" + anh_goc.split("/")[1]                      # fold/con/anh -> con
    if trai == "alameer":
        return anh_goc.split("--")[0]                               # cùng video
    if trai == "bergamini":
        m = re.search(r"(PIGS\d+__\d+)", anh_goc)
        return m.group(1) if m else anh_goc                         # cùng clip
    so = int(re.search(r"(\d+)\.\w+$", anh_goc).group(1))           # danuma_1185.jpg, psota_000000379.jpg
    return f"{trai}_{so // KHOI_KHUNG}"


def chia_nhom(bang, rng):
    """Gán từng nhóm vào tập đang thiếu nhiều nhất so với mục tiêu (tính theo số ảnh)."""
    dem = bang.groupby("nhom").size()
    nhoms = dem.index.tolist()
    rng.shuffle(nhoms)
    tong = dem.sum()
    da = {k: 0 for k in TI_LE}
    gan = {}
    for g in sorted(nhoms, key=lambda g: -dem[g]):                 # nhóm lớn xếp trước cho đều
        k = max(TI_LE, key=lambda t: TI_LE[t] * tong - da[t])
        gan[g] = k
        da[k] += dem[g]
    return bang.nhom.map(gan)


def main():
    rng = np.random.default_rng(SEED)
    ds = pd.concat([pd.read_csv(THU_MUC / "danh_sach.csv"),
                    pd.read_csv(THU_MUC / "danh_sach_ung_vien_dung.csv")]).drop_duplicates("file")
    ds = ds.set_index("file")

    dong = []
    for lop in LOP:
        for f in sorted((THU_MUC / lop).glob("*.jpg")):
            r = ds.loc[f.name]
            dong.append({"file": f.name, "lop": lop, "trai": r.trai, "anh_goc": r.anh_goc})
    bang = pd.DataFrame(dong)

    # Chọn SO_RGB ảnh RGB_9579 lấy đều theo con lợn
    rgb = bang[bang.trai == "rgb9579"].copy()
    rgb["con"] = rgb.anh_goc.str.split("/").str[1]
    rgb = rgb.sample(frac=1, random_state=SEED)
    rgb["thu_tu"] = rgb.groupby("con").cumcount()
    rgb_chon = rgb.sort_values(["thu_tu", "con"]).head(SO_RGB)
    bang = pd.concat([bang[bang.trai != "rgb9579"], rgb_chon[bang.columns]], ignore_index=True)

    bang["nhom"] = [nhom(t, a) for t, a in zip(bang.trai, bang.anh_goc)]
    bang["tap"] = ""
    for trai in bang.trai.unique():
        m = bang.trai == trai
        bang.loc[m, "tap"] = chia_nhom(bang[m], rng).values

    # Kiểm tra: không nhóm nào nằm ở 2 tập
    assert (bang.groupby("nhom").tap.nunique() == 1).all()

    for tap in TI_LE:                              # chỉ xóa 3 thư mục ảnh cũ, KHÔNG xóa code trong RA
        if (RA / tap).exists():
            shutil.rmtree(RA / tap)
    for tap in TI_LE:
        for lop in LOP:
            (RA / tap / lop).mkdir(parents=True)
    for r in bang.itertuples():
        anh = doc_anh(THU_MUC / r.lop / r.file)
        if r.trai == "rgb9579":
            h, w = anh.shape[:2]
            s = CANH_DAI_RGB / max(h, w)
            anh = cv2.resize(anh, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        ghi_anh(RA / r.tap / r.lop / r.file, chuan_hoa_anh(anh))
    bang.to_csv(RA / "split.csv", index=False)

    print(pd.crosstab([bang.tap, bang.lop], bang.trai, margins=True))
    print("\nTi le anh moi tap:", (bang.tap.value_counts(normalize=True) * 100).round(1).to_dict())


if __name__ == "__main__":
    main()
