"""Test thực tế mô hình cắt lưng lợn: đưa ảnh vào -> hiện mask lên màn hình (không lưu file).

Cách chạy:
    python scripts/test_cat_lung.py "D:/anh/lon1.jpg"
    python scripts/test_cat_lung.py anh1.jpg anh2.jpg              # nhiều ảnh, bấm phím bất kỳ để sang ảnh sau
    python scripts/test_cat_lung.py anh.jpg --conf 0.5             # chỉ hiện mask có độ tin cậy >= 0.5
    python scripts/test_cat_lung.py anh.jpg --model models/Pig_Seg/train/weights/last.pt

Trên cửa sổ ảnh: bấm phím bất kỳ = ảnh tiếp theo, ESC hoặc q = thoát.

Lưu ý: mô hình được train trên ảnh ĐÃ CẮT quanh 1 con lợn (nới hộp 12%). Ảnh test giống vậy sẽ cho kết quả
đúng nhất. Ảnh toàn chuồng nhiều con vẫn chạy được nhưng có thể kém hơn (bình thường sẽ do pig_det cắt trước).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

GOC = Path(__file__).resolve().parents[1]
MODEL_MAC_DINH = GOC / "models" / "Pig_Seg" / "pig_seg.pt"

# Màu (B, G, R) cho từng con nếu ảnh có nhiều con
BANG_MAU = [(0, 0, 255), (0, 200, 0), (255, 120, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0)]


def doc_tham_so():
    ap = argparse.ArgumentParser(description="Test mô hình cắt lưng lợn trên ảnh thật")
    ap.add_argument("anh", nargs="+", help="đường dẫn 1 hoặc nhiều ảnh")
    ap.add_argument("--model", default=str(MODEL_MAC_DINH))
    ap.add_argument("--conf", type=float, default=0.25, help="ngưỡng độ tin cậy tối thiểu")
    ap.add_argument("--imgsz", type=int, default=640, help="giống lúc train")
    return ap.parse_args()


def ve_ket_qua(anh, kq):
    """Tô mask bán trong suốt + viền + ghi độ tin cậy và diện tích (pixel) của từng con."""
    ra = anh.copy()
    if kq.masks is None:
        cv2.putText(ra, "Khong tim thay lon", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        return ra, []

    thong_tin = []
    lop_mau = ra.copy()
    for i, (m, conf) in enumerate(zip(kq.masks.data.cpu().numpy(), kq.boxes.conf.cpu().numpy())):
        m = (m > 0.5).astype(np.uint8)          # mask xác suất -> nhị phân 0/1
        mau = BANG_MAU[i % len(BANG_MAU)]
        lop_mau[m == 1] = mau
        vien, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ra, vien, -1, mau, 3)
        dien_tich = int(m.sum())                  # số pixel thuộc lưng lợn (sau này dùng tính cân nặng)
        thong_tin.append((i + 1, float(conf), dien_tich))
        if vien:
            x, y, _, _ = cv2.boundingRect(max(vien, key=cv2.contourArea))
            cv2.putText(ra, f"#{i + 1} {conf:.2f}", (x, max(y - 8, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, mau, 2)
    # Trộn 60% ảnh gốc + 40% lớp màu -> mask trong suốt, vẫn nhìn thấy lưng lợn bên dưới
    ra = cv2.addWeighted(ra, 0.6, lop_mau, 0.4, 0)
    return ra, thong_tin


def vua_man_hinh(anh, rong_toi_da=1400, cao_toi_da=850):
    """Thu nhỏ ảnh quá to để cửa sổ không tràn màn hình (chỉ để xem, không đổi kết quả)."""
    h, w = anh.shape[:2]
    ti_le = min(rong_toi_da / w, cao_toi_da / h, 1.0)
    return cv2.resize(anh, (int(w * ti_le), int(h * ti_le))) if ti_le < 1 else anh


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    a = doc_tham_so()
    if not Path(a.model).exists():
        sys.exit(f"Không thấy model: {a.model}. Hãy chạy scripts/train_cat_lung.py trước.")

    from ultralytics import YOLO
    model = YOLO(a.model)

    for duong_dan in a.anh:
        # imdecode + fromfile thay cho cv2.imread để đọc được đường dẫn có dấu tiếng Việt trên Windows
        anh = cv2.imdecode(np.fromfile(duong_dan, np.uint8), cv2.IMREAD_COLOR) if Path(duong_dan).exists() else None
        if anh is None:
            print(f"[Bỏ qua] Không đọc được ảnh: {duong_dan}")
            continue

        # retina_masks=True: mask có đúng kích thước ảnh gốc để vẽ chồng lên ảnh
        kq = model.predict(anh, imgsz=a.imgsz, conf=a.conf, retina_masks=True, verbose=False)[0]
        ra, thong_tin = ve_ket_qua(anh, kq)

        print(f"\n{duong_dan}  ({anh.shape[1]}x{anh.shape[0]}, {kq.speed['inference']:.0f} ms)")
        if not thong_tin:
            print("  Không tìm thấy con lợn nào.")
        for so, conf, dt in thong_tin:
            print(f"  Con #{so}: độ tin cậy {conf:.3f} | diện tích mask {dt} pixel")

        ten_cua_so = "Cat lung lon - phim bat ky: anh tiep, ESC/q: thoat"
        cv2.imshow(ten_cua_so, vua_man_hinh(ra))
        phim = cv2.waitKey(0) & 0xFF
        if phim in (27, ord("q")):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
