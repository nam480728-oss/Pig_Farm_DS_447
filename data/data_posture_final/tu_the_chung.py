"""Hàm xử lý ảnh DÙNG CHUNG cho model tư thế đứng / không đứng.

Nằm trong data/data_posture_final/ (phần xử lý dữ liệu). Phải dùng đúng hàm này ở mọi chỗ
để ảnh lúc train và lúc chạy thật giống nhau:
  - data/data_posture_final/chia_du_lieu.py (tạo dữ liệu train/val/test)
  - code test / pipeline camera (ảnh cắt từ hộp pig_det, nới 10 % mỗi cạnh): import từ đây
    sys.path.insert(0, "<gốc dự án>/data/data_posture_final"); from tu_the_chung import chuan_hoa_anh
"""
import cv2
import numpy as np

KICH_THUOC = 224          # ảnh vuông đưa vào YOLO-cls
MAU_VIEN = (114, 114, 114)  # xám, giống letterbox của Ultralytics


def xoay_nam_ngang(anh):
    """Xoay 90° nếu ảnh dọc -> thân lợn luôn nằm ngang, hướng ảnh không còn mang thông tin về nhãn."""
    h, w = anh.shape[:2]
    return cv2.rotate(anh, cv2.ROTATE_90_CLOCKWISE) if h > w else anh


def them_vien_vuong(anh, kich_thuoc=KICH_THUOC):
    """Thu/phóng giữ tỉ lệ cho cạnh dài = kich_thuoc rồi thêm viền xám thành ảnh vuông (không cắt, không méo)."""
    h, w = anh.shape[:2]
    s = kich_thuoc / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    nho = cv2.resize(anh, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    ra = np.full((kich_thuoc, kich_thuoc, 3), MAU_VIEN, np.uint8)
    y, x = (kich_thuoc - nh) // 2, (kich_thuoc - nw) // 2
    ra[y:y + nh, x:x + nw] = nho
    return ra


def chuan_hoa_anh(anh, kich_thuoc=KICH_THUOC):
    """Ảnh cắt 1 con lợn (BGR) -> ảnh vuông kich_thuoc x kich_thuoc, thân nằm ngang."""
    return them_vien_vuong(xoay_nam_ngang(anh), kich_thuoc)
