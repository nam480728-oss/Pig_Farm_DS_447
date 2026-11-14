"""Test thực tế mô hình cân nặng: ảnh -> pig_seg (mask) -> 5 đặc trưng -> XGBoost -> kg. Hiện ảnh, không lưu.

Cách chạy:
    python scripts/test_kg.py "data/data_cat_lung_tinh_kg/RGB_9579/fold1/101.74_127/101.74kg_1.png"
    python scripts/test_kg.py anh1.png anh2.png                # nhiều ảnh, bấm phím bất kỳ để sang ảnh sau
    python scripts/test_kg.py anh_da_cat.jpg --da-cat           # ảnh đã cắt sẵn quanh 1 con (bỏ bước tự cắt)

Quy trình (giống hệt lúc tạo dữ liệu train trong lay_dac_trung.py):
    1. Ảnh nguyên khung -> pig_seg lượt 1 tìm con chính -> nới hộp 12 % -> cắt   (bước này sau sẽ do pig_det làm)
    2. Ảnh cắt -> pig_seg lượt 2 -> mask lưng
    3. Phép mở bỏ tai/chân/đuôi -> 5 đặc trưng (pixel) -> in ra terminal
    4. XGBoost (weight.json) -> kg dự đoán -> in ra terminal + ghi lên ảnh

Trên cửa sổ ảnh: đỏ = mask thô, xanh lá = mask sau khi bỏ tai/chân/đuôi, vàng = hình chữ nhật dài × rộng,
tím = ellipse (độ lệch tâm), xanh dương = hộp cắt. Phím bất kỳ: ảnh tiếp, ESC/q: thoát.

LƯU Ý: đặc trưng tính bằng PIXEL, mô hình học trên ảnh 960x540 chụp từ độ cao 1,88 m (RGB_9579).
Ảnh khác độ phân giải / độ cao camera -> số kg sẽ lệch, cần hiệu chỉnh trước khi tin con số tuyệt đối.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# BẮT BUỘC import torch (qua ultralytics) TRƯỚC xgboost: trên máy này import xgboost trước sẽ làm torch
# báo lỗi "WinError 1114 ... c10.dll" (xung đột thư viện DLL trên Windows).
from ultralytics import YOLO
from xgboost import XGBRegressor

from lay_dac_trung import (bo_tai_chan_duoi, cat_quanh_lon, cham_bien, dac_trung, mask_con_chinh,
                           vien_lon_nhat)

GOC = Path(__file__).resolve().parents[1]
MODEL_SEG = GOC / "models" / "Pig_Seg" / "pig_seg.pt"
MODEL_KG = GOC / "models" / "Pig_Kg" / "weight.json"
THONG_TIN = GOC / "models" / "Pig_Kg" / "thong_tin_mo_hinh.json"
KICH_THUOC_TRAIN = (960, 540)   # (rộng, cao) ảnh RGB_9579 mà mô hình đã học


def doc_tham_so():
    ap = argparse.ArgumentParser(description="Test mô hình cân nặng lợn trên ảnh thật")
    ap.add_argument("anh", nargs="+", help="đường dẫn 1 hoặc nhiều ảnh")
    ap.add_argument("--model-seg", default=str(MODEL_SEG))
    ap.add_argument("--model-kg", default=str(MODEL_KG))
    ap.add_argument("--da-cat", action="store_true", help="ảnh đã cắt sẵn quanh 1 con -> bỏ bước tự cắt (lượt 1)")
    return ap.parse_args()


def ve_ket_qua(anh, hop, mask, ti_le_kernel, kg, dt):
    """Vẽ mask, các đặc trưng và số kg lên ảnh gốc. Mask nằm trong hệ tọa độ ảnh cắt -> dịch theo góc hộp cắt."""
    ra = anh.copy()
    x1, y1, x2, y2 = hop
    cv2.rectangle(ra, (x1, y1), (x2, y2), (255, 0, 0), 2)
    lech = np.array([x1, y1])
    mask_mo, _ = bo_tai_chan_duoi(mask, ti_le_kernel)
    tho, mo = vien_lon_nhat(mask), vien_lon_nhat(mask_mo)

    lop = ra.copy()                                         # tô mờ phần thân dùng để đo
    if mo is not None:
        cv2.drawContours(lop, [mo + lech], -1, (0, 255, 0), -1)
    ra = cv2.addWeighted(ra, 0.75, lop, 0.25, 0)
    if tho is not None:
        cv2.drawContours(ra, [tho + lech], -1, (0, 0, 255), 2)
    if mo is not None and len(mo) >= 5:
        cv2.drawContours(ra, [mo + lech], -1, (0, 255, 0), 2)
        hcn = cv2.minAreaRect(mo)
        cv2.drawContours(ra, [np.intp(cv2.boxPoints(hcn)) + lech], -1, (0, 255, 255), 1)
        (cx, cy), truc, goc = cv2.fitEllipse(mo)
        cv2.ellipse(ra, ((cx + x1, cy + y1), truc, goc), (255, 0, 255), 1)

    chu = [f"{kg:.1f} kg", f"DT {dt['dien_tich']:.0f}  CV {dt['chu_vi']:.0f}",
           f"D {dt['dai']:.0f}  R {dt['rong']:.0f}  e {dt['lech_tam']:.3f}"]
    for i, c in enumerate(chu):
        co = 1.1 if i == 0 else 0.6
        y = 40 + i * 30
        cv2.putText(ra, c, (12, y), cv2.FONT_HERSHEY_SIMPLEX, co, (0, 0, 0), 4)       # viền đen cho dễ đọc
        cv2.putText(ra, c, (12, y), cv2.FONT_HERSHEY_SIMPLEX, co, (255, 255, 255), 2)
    return ra


def vua_man_hinh(anh, rong_toi_da=1400, cao_toi_da=850):
    h, w = anh.shape[:2]
    tl = min(rong_toi_da / w, cao_toi_da / h, 1.0)
    return cv2.resize(anh, (int(w * tl), int(h * tl))) if tl < 1 else anh


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    a = doc_tham_so()
    for p in (a.model_seg, a.model_kg, THONG_TIN):
        if not Path(p).exists():
            sys.exit(f"Không thấy {p}. Cần train pig_seg (train_cat_lung.py) và cân nặng (train_can_nang.py) trước.")

    # Thông tin lúc train: thứ tự đặc trưng + tỉ lệ kernel phải GIỐNG HỆT để số đo khớp với mô hình
    tt = json.loads(Path(THONG_TIN).read_text(encoding="utf-8"))
    thu_tu, ti_le_kernel = tt["dac_trung_theo_thu_tu"], tt["ti_le_kernel_phep_mo"]
    kg_min, kg_max = tt["khoang_kg_da_hoc"]
    seg = YOLO(a.model_seg)
    kg_model = XGBRegressor()
    kg_model.load_model(a.model_kg)

    for duong_dan in a.anh:
        # imdecode + fromfile: đọc được đường dẫn có dấu tiếng Việt trên Windows
        anh = cv2.imdecode(np.fromfile(duong_dan, np.uint8), cv2.IMREAD_COLOR) if Path(duong_dan).exists() else None
        if anh is None:
            print(f"[Bỏ qua] Không đọc được ảnh: {duong_dan}")
            continue
        h, w = anh.shape[:2]
        print(f"\n{duong_dan}  ({w}x{h})")

        # Bước 1: cắt quanh con chính (hoặc dùng nguyên ảnh nếu đã cắt sẵn)
        if a.da_cat:
            anh_cat, hop = anh, (0, 0, w, h)
        else:
            if (w, h) != KICH_THUOC_TRAIN:
                print(f"  [Cảnh báo] Ảnh {w}x{h} khác ảnh train {KICH_THUOC_TRAIN[0]}x{KICH_THUOC_TRAIN[1]} "
                      f"-> đặc trưng pixel đổi tỉ lệ, kg có thể lệch nhiều.")
            cat = cat_quanh_lon(seg, anh)
            if cat is None:
                print("  Không tìm thấy con lợn nào.")
                continue
            anh_cat, hop, _, so_con = cat
            if so_con > 1:
                print(f"  Thấy {so_con} con, chọn con chính (độ tin cậy cao, gần tâm ảnh).")

        # Bước 2: mask lưng trên ảnh cắt
        mask, conf = mask_con_chinh(seg, anh_cat)
        if mask is None:
            print("  Không tìm thấy con lợn nào.")
            continue

        # Bước 3: 5 đặc trưng (cùng hàm, cùng kernel với lúc train)
        dt = dac_trung(mask, ti_le_kernel=ti_le_kernel)
        if any(np.isnan(dt[k]) for k in thu_tu):
            print("  Mask quá nhỏ, không đo được đặc trưng.")
            continue

        # Bước 4: dự đoán kg (đưa đặc trưng theo ĐÚNG thứ tự lúc train)
        x = np.array([[dt[k] for k in thu_tu]])
        kg = float(kg_model.predict(x)[0])

        print(f"  Độ tin cậy mask : {conf:.3f}")
        print(f"  Diện tích       : {dt['dien_tich']:10.1f} px²")
        print(f"  Chu vi          : {dt['chu_vi']:10.1f} px")
        print(f"  Chiều dài thân  : {dt['dai']:10.1f} px")
        print(f"  Chiều rộng thân : {dt['rong']:10.1f} px")
        print(f"  Độ lệch tâm     : {dt['lech_tam']:10.4f}")
        print(f"  ==> CÂN NẶNG DỰ ĐOÁN: {kg:.1f} kg")
        if cham_bien(mask):
            print("  [Cảnh báo] Mask chạm mép ảnh -> lợn có thể bị cắt cụt, kg dễ bị thấp.")
        if conf < 0.5:
            print("  [Cảnh báo] Độ tin cậy mask thấp (< 0.5).")
        if not kg_min <= kg <= kg_max:
            print(f"  [Cảnh báo] Ngoài khoảng cân nặng đã học ({kg_min:.0f}-{kg_max:.0f} kg) -> kém tin cậy.")

        cv2.imshow("Can nang lon - phim bat ky: anh tiep, ESC/q: thoat",
                   vua_man_hinh(ve_ket_qua(anh, hop, mask, ti_le_kernel, kg, dt)))
        if cv2.waitKey(0) & 0xFF in (27, ord("q")):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
