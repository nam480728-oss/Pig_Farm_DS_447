"""Kiểm tra và hiển thị dữ liệu train: pig_det, pig_seg, RGB_9579.

Chạy ở bất kỳ đâu:  python data/kiem_tra_du_lieu.py
Tùy chọn:           --so-mau 9      số ảnh mẫu mỗi bộ (mặc định 6)
                    --khong-mo      chỉ lưu ảnh mẫu, không tự mở
Ảnh mẫu lưu trong data/kiem_tra_anh/.
"""
import argparse
import os
import random
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw

GOC = Path(__file__).resolve().parent
DET = GOC / "data_phat_hien_pig" / "pig_det"
SEG = GOC / "data_cat_lung_tinh_kg" / "pig_seg"
CAN_NANG = GOC / "data_cat_lung_tinh_kg" / "RGB_9579"
THU_MUC_ANH = GOC / "kiem_tra_anh"

DUOI_ANH = {".jpg", ".jpeg", ".png"}
TEN_SEG = re.compile(r"^(.+?_\d+)_(.+kg_\d+)$")   # <kg>_<so anh>_<kg>kg_<stt>


def ds_anh(thu_muc):
    return sorted(f for f in thu_muc.glob("*") if f.suffix.lower() in DUOI_ANH)


def hop_le_det(dong):
    p = dong.split()
    try:
        return len(p) == 5 and p[0] == "0" and all(0 <= float(v) <= 1 for v in p[1:]) \
            and float(p[3]) > 0 and float(p[4]) > 0
    except ValueError:
        return False


def hop_le_seg(dong):
    p = dong.split()
    try:
        v = [float(t) for t in p[1:]]
    except ValueError:
        return False
    return len(p) >= 7 and p[0] == "0" and len(v) % 2 == 0 and all(0 <= t <= 1 for t in v)


def check_pig_det():
    print("=== KIỂM TRA DỮ LIỆU PIG_DET ===")
    tap = {}
    for split in ["train", "val", "test"]:
        anh = ds_anh(DET / "images" / split)
        nhan = {f.stem: f for f in (DET / "labels" / split).glob("*.txt")}
        tap[split] = {f.name for f in anh}
        so_hop = loi = 0
        for f in nhan.values():
            for dong in f.read_text().splitlines():
                if dong.strip():
                    so_hop += 1
                    loi += not hop_le_det(dong)
        thieu = len({f.stem for f in anh} - set(nhan))
        thua = len(set(nhan) - {f.stem for f in anh})
        print(f"Tập {split.upper():5s}: {len(anh):5d} ảnh | {len(nhan):5d} file nhãn | {so_hop:6d} hộp "
              f"| thiếu nhãn {thieu} | nhãn thừa {thua} | dòng lỗi {loi}")
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        print(f"Trùng lặp {a}-{b}: {len(tap[a] & tap[b])} ảnh.")


def check_pig_seg():
    print("\n=== KIỂM TRA DỮ LIỆU PIG_SEG ===")
    con = {}
    for split in ["train", "val"]:
        anh = ds_anh(SEG / "images" / split)
        nhan = {f.stem: f for f in (SEG / "labels" / split).glob("*.txt")}
        loi = 0
        for f in nhan.values():
            dong = [d for d in f.read_text().splitlines() if d.strip()]
            loi += not (len(dong) == 1 and hop_le_seg(dong[0]))
        con[split] = {m.group(1) for f in anh if (m := TEN_SEG.match(f.stem))}
        thieu = len({f.stem for f in anh} - set(nhan))
        print(f"Tập {split.upper():5s}: {len(anh):5d} ảnh | {len(nhan):5d} file nhãn YOLO-seg "
              f"| {len(con[split]):2d} con lợn | thiếu nhãn {thieu} | nhãn lỗi {loi}")
    print(f"Con lợn nằm ở cả train và val: {len(con['train'] & con['val'])}")


def check_rgb_9579():
    print("\n=== KIỂM TRA DỮ LIỆU RGB_9579 (cân nặng) ===")
    tong_anh = tong_con = 0
    kg_tat_ca = []
    for fold in sorted(f for f in CAN_NANG.iterdir() if f.is_dir()):
        so_anh = so_con = 0
        kg = []
        for con in sorted(f for f in fold.iterdir() if f.is_dir()):
            so_con += 1
            so_anh += len(ds_anh(con))
            kg.append(float(con.name.split("_")[0]))
        tong_anh, tong_con = tong_anh + so_anh, tong_con + so_con
        kg_tat_ca += kg
        print(f"{fold.name}: {so_con:2d} con | {so_anh:5d} ảnh | cân nặng {min(kg):6.2f} - {max(kg):6.2f} kg")
    print(f"Tổng: {tong_con} con | {tong_anh} ảnh | cân nặng {min(kg_tat_ca):.2f} - {max(kg_tat_ca):.2f} kg")


# ---------------------------------------------------------------- hiển thị ảnh mẫu
def ghep_luoi(o, so_cot, rong, cao, ten_file):
    so_hang = (len(o) + so_cot - 1) // so_cot
    luoi = Image.new("RGB", (so_cot * rong, so_hang * cao), (30, 30, 30))
    for i, anh in enumerate(o):
        anh.thumbnail((rong, cao))
        luoi.paste(anh, ((i % so_cot) * rong, (i // so_cot) * cao))
    THU_MUC_ANH.mkdir(exist_ok=True)
    duong_dan = THU_MUC_ANH / ten_file
    luoi.save(duong_dan, quality=90)
    return duong_dan


def mau_det(n):
    anh = [(s, f) for s in ["train", "val", "test"] for f in ds_anh(DET / "images" / s)]
    o = []
    for s, f in random.sample(anh, min(n, len(anh))):
        im = Image.open(f).convert("RGB")
        w, h = im.size
        ve = ImageDraw.Draw(im)
        for dong in (DET / "labels" / s / f"{f.stem}.txt").read_text().splitlines():
            if hop_le_det(dong):
                _, cx, cy, bw, bh = map(float, dong.split())
                ve.rectangle([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h],
                             outline=(255, 0, 0), width=max(2, w // 400))
        ve.text((8, 8), s, fill=(255, 255, 0))
        o.append(im)
    return ghep_luoi(o, 3, 480, 270, "pig_det_mau.jpg")


def mau_seg(n):
    anh = [(s, f) for s in ["train", "val"] for f in ds_anh(SEG / "images" / s)]
    o = []
    for s, f in random.sample(anh, min(n, len(anh))):
        im = Image.open(f).convert("RGB")
        w, h = im.size
        dong = (SEG / "labels" / s / f"{f.stem}.txt").read_text().split()
        v = [float(t) for t in dong[1:]]
        ImageDraw.Draw(im).polygon([(v[i] * w, v[i + 1] * h) for i in range(0, len(v), 2)],
                                   outline=(255, 0, 0))
        o.append(im)
    return ghep_luoi(o, 3, 480, 360, "pig_seg_mau.jpg")


def mau_can_nang(n):
    anh = [f for f in CAN_NANG.rglob("*") if f.suffix.lower() in DUOI_ANH]
    o = []
    for f in random.sample(anh, min(n, len(anh))):
        im = Image.open(f).convert("RGB")
        ImageDraw.Draw(im).text((10, 10), f"{f.parent.parent.name} | {f.parent.name} | {f.name}",
                                fill=(255, 255, 0))
        o.append(im)
    return ghep_luoi(o, 3, 480, 270, "rgb_9579_mau.jpg")


def main():
    sys.stdout.reconfigure(encoding="utf-8")   # console Windows mặc định không in được tiếng Việt
    ap = argparse.ArgumentParser()
    ap.add_argument("--so-mau", type=int, default=6)
    ap.add_argument("--khong-mo", action="store_true")
    a = ap.parse_args()

    check_pig_det()
    check_pig_seg()
    check_rgb_9579()

    random.seed(0)
    print("\n=== ẢNH MẪU ===")
    for ham in (mau_det, mau_seg, mau_can_nang):
        duong_dan = ham(a.so_mau)
        print(f"Đã lưu: {duong_dan}")
        if not a.khong_mo:
            try:
                os.startfile(duong_dan)
            except (AttributeError, OSError):
                pass


if __name__ == "__main__":
    main()
