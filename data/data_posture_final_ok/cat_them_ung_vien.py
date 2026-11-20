"""Cắt THÊM ảnh lợn từ PigDetect (những hộp chưa từng cắt) để tìm thêm con ĐỨNG.

Dùng lại hàm và bộ lọc của cat_anh_tu_the.py (không chạm mép, không quá nhỏ, bị che <= 25 %, nới 10 %).
Bỏ qua mọi hộp đã có trong danh_sach.csv -> không trùng ảnh đã gán nhãn.
Ảnh cắt ghi vào thư mục tạm (tham số --ra); sau khi xem, chỉ những ảnh nghi là ĐỨNG được chuyển vào khong_ro/
để người kiểm tra tự phân loại.
Chạy:  python data/data_posture_final_ok/cat_them_ung_vien.py --ra <thư mục tạm>
"""
import argparse
import glob
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cat_anh_tu_the import NOI, PD, bi_che, cat, doc_anh, ghi_anh, hop_pigdetect  # noqa: E402

# Số ảnh cần cắt thêm cho mỗi trại (danuma, psota nhiều cảnh lợn đứng / đi lại hơn)
CHI_TIEU = {"danuma": 900, "psota": 700, "alameer": 500, "bergamini": 400}
TOI_DA_MOI_ANH = 4
SEED = 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ra", required=True)
    a = ap.parse_args()
    ra = Path(a.ra)
    ra.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)

    cu = pd.read_csv(Path(__file__).resolve().parents[1] / "data_posture" / "danh_sach.csv")
    da_cat = set(zip(cu.anh_goc, cu.hop))
    dong = []
    for trai, chi_tieu in CHI_TIEU.items():
        anh_trai = sorted(glob.glob(str(PD / "dev" / "images" / f"{trai}_*"))) + \
                   sorted(glob.glob(str(PD / "test" / "images" / f"{trai}_*")))
        random.shuffle(anh_trai)
        lay = 0
        for f in anh_trai:
            if lay >= chi_tieu:
                break
            anh = doc_anh(f)
            h, w = anh.shape[:2]
            hop = hop_pigdetect(f, w, h)
            ok = [i for i, (x1, y1, x2, y2) in enumerate(hop)
                  if (Path(f).name, i) not in da_cat
                  and x1 >= 3 and y1 >= 3 and x2 <= w - 3 and y2 <= h - 3
                  and min(x2 - x1, y2 - y1) >= 0.04 * min(w, h)
                  and bi_che(i, hop) <= 0.25]
            for i in random.sample(ok, min(TOI_DA_MOI_ANH, len(ok))):
                if lay >= chi_tieu:
                    break
                ten = f"pd_{trai}_{Path(f).stem}_h{i:02d}.jpg"
                ghi_anh(ra / ten, cat(anh, *hop[i], NOI))
                dong.append({"file": ten, "nguon": "pigdetect", "trai": trai, "anh_goc": Path(f).name,
                             "tap_goc": Path(f).parents[1].name, "hop": i})
                lay += 1
        print(f"{trai}: {lay}")
    pd.DataFrame(dong).to_csv(ra / "danh_sach_them.csv", index=False)
    print("Tong:", len(dong))


if __name__ == "__main__":
    main()
