"""Test mô hình PHÁT HIỆN LỢN (pig_det.pt, YOLO11l nhóm khác train trên PigDetect): đưa ảnh vào -> hiện hộp
từng con lên màn hình (không lưu file).

Cách chạy (đứng ở thư mục gốc dự án):
    python scripts/test_pig_det.py "D:/anh/chuong.jpg"
    python scripts/test_pig_det.py anh1.jpg anh2.jpg                       # nhiều ảnh
    python scripts/test_pig_det.py data/data_phat_hien_pig/pig_det/images/test   # cả thư mục ảnh
    python scripts/test_pig_det.py anh.jpg --conf 0.5                      # chỉ giữ hộp tin cậy >= 0.5

Trên cửa sổ: phím bất kỳ = ảnh tiếp, ESC / q = thoát.
Nếu ảnh có file nhãn YOLO đi kèm (vd. ảnh PigDetect: .../images/x.jpg <-> .../labels/x.txt), script vẽ thêm hộp
nhãn thật (màu trắng) và in số con tìm đúng / bỏ sót / thừa (khớp khi IoU >= 0.5).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

GOC = Path(__file__).resolve().parents[1]
MODEL_MAC_DINH = GOC / "models" / "Pig_Det" / "models" / "pig_det.pt"
DUOI_ANH = {".jpg", ".jpeg", ".png", ".bmp"}


def doc_tham_so():
    ap = argparse.ArgumentParser(description="Test mô hình phát hiện lợn pig_det trên ảnh")
    ap.add_argument("anh", nargs="+", help="đường dẫn ảnh hoặc thư mục ảnh")
    ap.add_argument("--model", default=str(MODEL_MAC_DINH))
    ap.add_argument("--conf", type=float, default=0.25, help="ngưỡng độ tin cậy tối thiểu")
    ap.add_argument("--imgsz", type=int, default=960, help="giống lúc train (960)")
    return ap.parse_args()


def kiem_tra_model(p):
    """Dừng nếu không có model hoặc chỉ là con trỏ Git LFS (~130 byte) chứ chưa phải file model thật."""
    p = Path(p)
    if not p.exists():
        sys.exit(f"Không thấy model: {p}")
    if p.stat().st_size < 1_000_000:
        dau = p.read_bytes()[:200].decode("utf-8", "ignore")
        if "git-lfs" in dau:
            sys.exit(f"{p} chỉ {p.stat().st_size} byte: đây là CON TRỎ Git LFS, chưa phải model thật.\n"
                     f"Lấy file thật (~51 MB) bằng 'git lfs pull' trong repo gốc, hoặc nhờ nhóm train gửi trực tiếp file .pt.")
        sys.exit(f"{p} quá nhỏ ({p.stat().st_size} byte), không phải model YOLO.")


def liet_ke_anh(ds):
    """Nhận cả file lẫn thư mục -> danh sách file ảnh."""
    ra = []
    for x in ds:
        x = Path(x)
        if x.is_dir():
            ra += sorted(f for f in x.iterdir() if f.suffix.lower() in DUOI_ANH)
        else:
            ra.append(x)
    return ra


def doc_anh(p):
    """imdecode + fromfile: đọc được đường dẫn có dấu tiếng Việt trên Windows."""
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR) if Path(p).exists() else None


def hop_nhan_that(f, w, h):
    """Đọc file nhãn YOLO đi kèm (images -> labels, .txt). Không có -> None."""
    parts = list(Path(f).parts)
    if "images" not in parts:
        return None
    parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
    nhan = Path(*parts).with_suffix(".txt")
    if not nhan.exists():
        return None
    hop = []
    for ln in nhan.read_text().split("\n"):
        p = ln.split()
        if len(p) == 5:
            cx, cy, bw, bh = map(float, p[1:])
            hop.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return np.array(hop).reshape(-1, 4)


def iou(a, b):
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    giao = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return giao / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - giao + 1e-9)


def so_khop(du, that, nguong=0.5):
    """Ghép tham lam theo IoU -> (đúng, bỏ sót, thừa)."""
    if len(that) == 0 or len(du) == 0:
        return 0, len(that), len(du)
    da = set()
    dung = 0
    for b in du:
        v = iou(b, that)
        v[list(da)] = 0
        j = int(v.argmax())
        if v[j] >= nguong:
            da.add(j)
            dung += 1
    return dung, len(that) - dung, len(du) - dung


def vua_man_hinh(anh, rong_toi_da=1400, cao_toi_da=850):
    h, w = anh.shape[:2]
    tl = min(rong_toi_da / w, cao_toi_da / h, 1.0)
    return cv2.resize(anh, (int(w * tl), int(h * tl))) if tl < 1 else anh


def ghi_chu(anh, chu, xy, mau, co=0.6, day=2):
    cv2.putText(anh, chu, xy, cv2.FONT_HERSHEY_SIMPLEX, co, (0, 0, 0), day + 3)
    cv2.putText(anh, chu, xy, cv2.FONT_HERSHEY_SIMPLEX, co, mau, day)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")      # thông báo lỗi (sys.exit) có dấu tiếng Việt
    a = doc_tham_so()
    kiem_tra_model(a.model)
    from ultralytics import YOLO
    model = YOLO(a.model)
    print("Lớp của model:", model.names)

    tong = [0, 0, 0]            # đúng, bỏ sót, thừa (chỉ ảnh có nhãn)
    for f in liet_ke_anh(a.anh):
        anh = doc_anh(f)
        if anh is None:
            print(f"[Bỏ qua] Không đọc được ảnh: {f}")
            continue
        h, w = anh.shape[:2]
        r = model.predict(anh, conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
        hop = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()

        ra = anh.copy()
        that = hop_nhan_that(f, w, h)
        if that is not None:
            for x1, y1, x2, y2 in that.astype(int):
                cv2.rectangle(ra, (x1, y1), (x2, y2), (255, 255, 255), 1)
        for (x1, y1, x2, y2), c in zip(hop.astype(int), conf):
            cv2.rectangle(ra, (x1, y1), (x2, y2), (0, 220, 0), 2)
            ghi_chu(ra, f"{c:.2f}", (x1 + 2, max(14, y1 + 14)), (0, 255, 0), 0.45, 1)

        dong = f"{len(hop)} con"
        print(f"\n{f}  ({w}x{h}, {r.speed['inference']:.0f} ms)")
        print(f"  Tìm thấy {len(hop)} con lợn" + (f" | độ tin cậy TB {conf.mean():.3f}, thấp nhất {conf.min():.3f}"
                                               if len(hop) else ""))
        if that is not None:
            d, s, t = so_khop(hop, that)
            tong = [tong[0] + d, tong[1] + s, tong[2] + t]
            print(f"  Nhãn thật {len(that)} con -> đúng {d}, bỏ sót {s}, thừa {t}")
            dong += f" | that {len(that)}: dung {d}, sot {s}, thua {t}"
        ghi_chu(ra, dong, (10, h - 15), (255, 255, 255), 0.7, 2)

        cv2.imshow("Pig_det - phim bat ky: anh tiep, ESC/q: thoat", vua_man_hinh(ra))
        if cv2.waitKey(0) & 0xFF in (27, ord("q")):
            break
    cv2.destroyAllWindows()

    if sum(tong):
        d, s, t = tong
        print(f"\nTỔNG (ảnh có nhãn): đúng {d}, bỏ sót {s}, thừa {t} | "
              f"recall {d / max(1, d + s):.3f}, precision {d / max(1, d + t):.3f}")


if __name__ == "__main__":
    main()
