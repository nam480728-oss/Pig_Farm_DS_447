"""Test mô hình CÂN NẶNG (chỉ cân nặng): ảnh / video -> pig_seg (mask) -> 10 đặc trưng -> XGBoost -> kg.
Không lưu gì: in ra terminal + hiện ảnh. Video: phát bình thường, chỉ dự đoán khung nào bấm CHỤP.

Cách chạy (đứng ở thư mục gốc dự án):
    python scripts/test_kg.py "data/data_cat_lung_tinh_kg/RGB_9579/fold1/101.74_127/101.74kg_1.png"
    python scripts/test_kg.py anh1.png anh2.png          # nhiều ảnh: phím bất kỳ = ảnh tiếp, ESC/q = thoát
    python scripts/test_kg.py "D:/video/lon.mp4"         # video: phát bình thường, bấm nút CHỤP (hoặc phím C)
                                                         #   -> dự đoán khung đang xem, hiện như đầu vào là ảnh
    python scripts/test_kg.py anh_da_cat.jpg --da-cat    # ảnh đã cắt sẵn quanh 1 con (bỏ bước tự cắt)

Quy trình (GIỐNG HỆT lúc tạo dữ liệu train trong lay_dac_trung.py):
    0. Đổi khung về rộng 960 px (giữ tỉ lệ) = thang pixel của ảnh train RGB_9579 (960x540)
    1. pig_seg lượt 1 trên cả khung -> con CHÍNH (tin cậy cao, gần tâm) -> nới hộp 12 % -> cắt
       (không dùng pig_det: file này chỉ test cân nặng; pipeline đầy đủ xem test_all_pro.py)
    2. Ảnh cắt -> pig_seg lượt 2 -> mask lưng
    3. Bỏ tai/chân/đuôi -> 10 đặc trưng (dac_trung của lay_dac_trung.py) -> in ra terminal
    4. XGBoost (weight.json) -> kg

Trên ảnh: xanh dương = hộp cắt, đỏ = mask thô, xanh lá (tô mờ) = thân sau khi bỏ tai/chân/đuôi,
vàng = 2 trục SVD (dài × rộng), xanh ngọc = đường sống lưng, hồng = đoạn nối 2 đầu, đỏ đậm = độ lệch lớn nhất.

LƯU Ý: đặc trưng tính bằng PIXEL, mô hình học trên ảnh 960x540 chụp thẳng đứng từ độ cao 1,88 m (RGB_9579).
Camera khác độ cao / góc nhìn -> số kg lệch, cần hiệu chỉnh trước khi tin con số tuyệt đối.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# BẮT BUỘC import torch (qua ultralytics) TRƯỚC xgboost: trên máy này import xgboost trước sẽ làm torch
# báo lỗi "WinError 1114 ... c10.dll" (xung đột thư viện DLL trên Windows).
from ultralytics import YOLO
from xgboost import XGBRegressor

from lay_dac_trung import cat_quanh_lon, cham_bien, dac_trung, mask_con_chinh, vien_lon_nhat

GOC = Path(__file__).resolve().parents[1]
MODEL_SEG = GOC / "models" / "Pig_Seg" / "pig_seg.pt"
MODEL_KG = GOC / "models" / "Pig_Kg" / "weight.json"
THONG_TIN = GOC / "models" / "Pig_Kg" / "thong_tin_mo_hinh.json"
RONG_TRAIN = 960                     # ảnh RGB_9579 rộng 960 px (960x540) -> mọi khung đổi về rộng này
DUOI_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v", ".mpg", ".mpeg"}
CUA_SO = "Can nang lon - phim bat ky: anh tiep, SPACE: dung video, ESC/q: thoat"


# =============================================================================== dùng chung (test_all_pro.py gọi lại)
def nap_model_kg():
    """weight.json + thông tin lúc train (thứ tự đặc trưng, tỉ lệ kernel, khoảng kg). Thiếu file -> dừng."""
    for p in (MODEL_SEG, MODEL_KG, THONG_TIN):
        if not p.exists():
            sys.exit(f"Không thấy {p}. Cần train pig_seg (train_cat_lung.py) và cân nặng (train_can_nang.py) trước.")
    tt = json.loads(THONG_TIN.read_text(encoding="utf-8"))
    m = XGBRegressor()
    m.load_model(str(MODEL_KG))
    return m, tt


def doi_ve_thang_train(anh, da_bao=[False]):
    """Đổi khung về rộng 960 px, GIỮ tỉ lệ (không méo hình) -> pixel cùng thang với ảnh train.

    Khung không phải 16:9 -> cao khác 540, in cảnh báo 1 lần (thang pixel theo chiều rộng vẫn khớp)."""
    h, w = anh.shape[:2]
    if not da_bao[0] and abs(w / h - 16 / 9) > 0.02:
        print(f"  [Cảnh báo] Khung {w}x{h} không phải 16:9 như ảnh train (960x540) -> đổi về rộng 960, cao "
              f"{round(h * RONG_TRAIN / w)}; x_cam/y_cam và thang pixel có thể lệch so với lúc train.")
        da_bao[0] = True
    if w == RONG_TRAIN:
        return anh
    return cv2.resize(anh, (RONG_TRAIN, round(h * RONG_TRAIN / w)),
                      interpolation=cv2.INTER_AREA if w > RONG_TRAIN else cv2.INTER_LINEAR)


def du_doan_kg(model_kg, tt, ds_dac_trung):
    """List dict 10 đặc trưng -> mảng kg (đưa cột theo ĐÚNG thứ tự lúc train)."""
    if not ds_dac_trung:
        return np.zeros(0)
    x = np.array([[d[k] for k in tt["dac_trung_theo_thu_tu"]] for d in ds_dac_trung])
    return model_kg.predict(x)


def ve_hinh_hoc(ra, mask, hh, goc, mau_vien=(0, 255, 0), day_du=True):
    """Vẽ mask + hình học đo đặc trưng (lấy từ dac_trung(..., chi_tiet=True)) lên ảnh ra.

    mask, hh nằm trong tọa độ ẢNH CẮT -> cộng goc = (x1, y1) để về tọa độ khung.
    day_du=False: chỉ tô thân + viền (chuồng đông, tránh rối)."""
    lech = np.array(goc, dtype=np.int32)
    lop = ra.copy()
    cv2.drawContours(lop, [hh["vien"] + lech], -1, mau_vien, -1)                  # tô mờ thân dùng để đo
    ra[:] = cv2.addWeighted(ra, 0.72, lop, 0.28, 0)
    if day_du:
        tho = vien_lon_nhat(mask)
        if tho is not None:
            cv2.drawContours(ra, [tho + lech], -1, (0, 0, 255), 1)                 # mask thô
    cv2.drawContours(ra, [hh["vien"] + lech], -1, mau_vien, 2)
    if not day_du:
        return
    tam, v1, v2 = hh["tam"] + np.array(goc, float), hh["v1"], hh["v2"]
    diem = lambda uu, ww: tuple(np.intp(np.round(tam + uu * v1 + ww * v2)))    # (u, w) -> pixel khung
    cv2.line(ra, diem(hh["u_min"], 0), diem(hh["u_max"], 0), (0, 255, 255), 1)   # trục chính = dài
    cv2.line(ra, diem(0, hh["w_min"]), diem(0, hh["w_max"]), (0, 255, 255), 1)   # trục phụ = rộng
    cv2.circle(ra, diem(0, 0), 4, (255, 0, 0), -1)                               # trọng tâm
    sl = hh["song_lung"]
    if len(sl) >= 3:
        pts = np.array([diem(a, b) for a, b in sl])
        cv2.polylines(ra, [pts], False, (255, 255, 0), 2)                        # đường sống lưng
        cv2.line(ra, tuple(pts[0]), tuple(pts[-1]), (180, 105, 255), 1)          # đoạn nối 2 đầu
        a, b, p = sl[0], sl[-1], sl[hh["j_lech"]]
        t = np.dot(p - a, b - a) / max(np.dot(b - a, b - a), 1e-9)
        cv2.line(ra, diem(*p), diem(*(a + t * (b - a))), (0, 0, 200), 2)          # độ lệch lớn nhất


def ghi_chu(anh, chu, xy, mau=(255, 255, 255), co=0.6, day=2):
    """Chữ trên NỀN TỐI (OpenCV 5: viền đen dày + chữ mỏng bị lệch nhau -> chữ nhòe, nên dùng nền)."""
    (w, h), d = cv2.getTextSize(chu, cv2.FONT_HERSHEY_SIMPLEX, co, day)
    x, y = xy
    x1, y1, x2, y2 = max(0, x - 3), max(0, y - h - 4), min(anh.shape[1], x + w + 3), min(anh.shape[0], y + d + 2)
    if x2 > x1 and y2 > y1:
        anh[y1:y2, x1:x2] = (anh[y1:y2, x1:x2] * 0.35).astype(anh.dtype)
    cv2.putText(anh, chu, xy, cv2.FONT_HERSHEY_SIMPLEX, co, mau, day)


def dong_dac_trung(dt):
    """10 đặc trưng gọn trên 1 dòng (để in terminal / ghi lên ảnh)."""
    return (f"DT {dt['dien_tich']:.0f} CV {dt['chu_vi']:.0f} D {dt['dai']:.0f} R {dt['rong']:.0f} "
            f"D/R {dt['dai_rong']:.2f} | x_cam {dt['x_cam']:+.2f} y_cam {dt['y_cam']:+.2f} | "
            f"x_tam {dt['x_tam']:.3f} dac {dt['do_dac']:.3f} cong {dt['do_cong']:.3f}")


def in_dac_trung(dt):
    """In đủ 10 đặc trưng, mỗi đặc trưng 1 dòng."""
    ten = [("dien_tich", "Diện tích (px²)", ".1f"), ("chu_vi", "Chu vi (px)", ".1f"), ("dai", "Dài SVD (px)", ".1f"),
           ("rong", "Rộng SVD (px)", ".1f"), ("dai_rong", "Dài / rộng", ".3f"), ("x_cam", "x_cam", "+.3f"),
           ("y_cam", "y_cam", "+.3f"), ("x_tam", "x_tam", ".4f"), ("do_dac", "Độ đặc", ".4f"),
           ("do_cong", "Độ cong sống lưng", ".4f")]
    for k, nhan, f in ten:
        print(f"    {nhan:18s}: {format(dt[k], f):>10s}")


def canh_bao_kg(kg, tt, mask=None, conf=None, tien_to="  "):
    """Các cảnh báo độ tin cậy -> list chuỗi."""
    kg_min, kg_max = tt["khoang_kg_da_hoc"]
    cb = []
    if mask is not None and cham_bien(mask):
        cb.append("mask chạm mép ảnh cắt -> lợn có thể bị cắt cụt, kg dễ thấp")
    if conf is not None and conf < 0.5:
        cb.append(f"độ tin cậy mask thấp ({conf:.2f} < 0.5)")
    if not kg_min <= kg <= kg_max:
        cb.append(f"ngoài khoảng kg đã học ({kg_min:.0f}-{kg_max:.0f} kg) -> kém tin cậy")
    return [f"{tien_to}[Cảnh báo] {c}" for c in cb]


def vua_man_hinh(anh, rong_toi_da=1400, cao_toi_da=850):
    h, w = anh.shape[:2]
    tl = min(rong_toi_da / w, cao_toi_da / h, 1.0)
    return cv2.resize(anh, (int(w * tl), int(h * tl))) if tl < 1 else anh


def doc_anh(p):
    """imdecode + fromfile: đọc được đường dẫn có dấu tiếng Việt trên Windows."""
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR) if Path(p).exists() else None


CUA_SO_VIDEO = "Video - bam CHUP (hoac phim C) de du doan khung dang xem"
CAO_THANH_NUT = 56                                   # thanh chứa nút CHỤP nằm dưới khung video (px)


def hien_cac_cua_so(ds):
    """ds = list (tên cửa sổ, ảnh hoặc None). None -> đóng cửa sổ đó (nếu đang mở)."""
    for ten, anh in ds:
        if anh is not None:
            cv2.imshow(ten, vua_man_hinh(anh))
            continue
        try:
            cv2.destroyWindow(ten)
        except cv2.error:
            pass


def ve_thanh_nut(khung, giay, tong_giay, dang_dung):
    """Khung video (đã thu nhỏ vừa màn hình) + thanh dưới có nút CHỤP. Trả về (ảnh ghép, hộp nút x1,y1,x2,y2)."""
    h, w = khung.shape[:2]
    thanh = np.full((CAO_THANH_NUT, w, 3), 40, np.uint8)
    nut = (10, 8, 10 + 190, CAO_THANH_NUT - 8)                  # tọa độ trong thanh
    cv2.rectangle(thanh, nut[:2], nut[2:], (0, 150, 255), -1)
    cv2.rectangle(thanh, nut[:2], nut[2:], (255, 255, 255), 1)
    cv2.putText(thanh, "CHUP (C)", (nut[0] + 32, nut[3] - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    trang_thai = "TAM DUNG" if dang_dung else "dang phat"
    cv2.putText(thanh, f"{giay:5.1f}s / {tong_giay:.1f}s  {trang_thai}  |  SPACE: dung/chay  ESC/q: thoat",
                (nut[2] + 14, CAO_THANH_NUT // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    return np.vstack([khung, thanh]), (nut[0], h + nut[1], nut[2], h + nut[3])


def phat_video(duong_dan, du_doan):
    """Phát video ĐÚNG tốc độ thật, KHÔNG dự đoán liên tục. Dưới khung có nút CHỤP:
    bấm nút (hoặc phím C) -> du_doan(khung đang xem) -> in terminal + hiện kết quả y như đầu vào là 1 ảnh.

    du_doan(khung) -> list (tên cửa sổ, ảnh hoặc None) để hiện. Video vẫn chạy tiếp khi đang xem kết quả;
    thời gian dự đoán được trừ ra nên video không nhảy cóc. SPACE = tạm dừng (lúc dừng vẫn chụp được).
    Trả về True nếu bấm ESC/q (thoát hẳn)."""
    cap = cv2.VideoCapture(str(duong_dan))
    if not cap.isOpened():
        print(f"[Bỏ qua] Không mở được video: {duong_dan}")
        return False
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1 or fps > 240:             # video không ghi fps -> coi là 30
        fps = 30.0
    tong = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n{duong_dan}  (video {w}x{h}, {fps:.1f} fps, {tong / fps:.1f} s)")
    print("  Bấm nút CHỤP dưới video (hoặc phím C) để dự đoán khung đang xem | SPACE = dừng/chạy | ESC/q = thoát")
    du_doan(np.zeros((h or 540, w or 960, 3), np.uint8))       # nạp model lên GPU trước -> lần chụp đầu không chậm

    tt = {"chup": False, "nut": (0, 0, 0, 0)}

    def bam_chuot(su_kien, x, y, *_):
        x1, y1, x2, y2 = tt["nut"]
        if su_kien == cv2.EVENT_LBUTTONDOWN and x1 <= x <= x2 and y1 <= y <= y2:
            tt["chup"] = True

    cv2.namedWindow(CUA_SO_VIDEO, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(CUA_SO_VIDEO, bam_chuot)

    def hien_video(khung, giay, dang_dung):
        anh, tt["nut"] = ve_thanh_nut(vua_man_hinh(khung, 1280, 720), giay, tong / fps, dang_dung)
        cv2.imshow(CUA_SO_VIDEO, anh)

    def chup(khung, giay):
        """Dự đoán khung đang xem -> in + hiện. Trả về thời gian đã tốn (để bù cho đồng hồ video)."""
        bat_dau = time.perf_counter()
        print(f"\n=== CHỤP tại {giay:.2f}s (khung {w}x{h}) ===")
        hien_cac_cua_so(du_doan(khung.copy()))
        return time.perf_counter() - bat_dau

    so_doc = so_chup = 0
    t0, thoat, khung, giay = None, False, None, 0.0
    while True:
        ok, moi = cap.read()
        if not ok:
            break
        khung, giay = moi, so_doc / fps             # vị trí khung này trong video (giây)
        so_doc += 1
        if t0 is None:
            t0 = time.perf_counter()
        tre = (time.perf_counter() - t0) - giay
        if tre > 2.0 / fps:                          # hiển thị không kịp -> bỏ khung cho đúng thời gian thật
            continue
        hien_video(khung, giay, False)

        cho = giay + 1.0 / fps - (time.perf_counter() - t0)   # chờ cho đúng tốc độ thật
        phim = cv2.waitKey(max(1, int(cho * 1000))) & 0xFF
        if phim in (ord("c"), ord("C")) or tt["chup"]:
            tt["chup"] = False
            t0 += chup(khung, giay)                  # bù thời gian dự đoán -> video chạy tiếp đúng chỗ
            so_chup += 1
        elif phim == ord(" "):                       # tạm dừng: vẫn bấm CHỤP được, SPACE để chạy tiếp
            t_dung = time.perf_counter()
            hien_video(khung, giay, True)
            while True:
                phim = cv2.waitKey(30) & 0xFF
                if phim in (ord("c"), ord("C")) or tt["chup"]:
                    tt["chup"] = False
                    chup(khung, giay)
                    so_chup += 1
                elif phim in (ord(" "), 27, ord("q")) or cv2.getWindowProperty(CUA_SO_VIDEO, cv2.WND_PROP_VISIBLE) < 1:
                    break
            t0 += time.perf_counter() - t_dung       # bù thời gian dừng
        if phim in (27, ord("q")) or cv2.getWindowProperty(CUA_SO_VIDEO, cv2.WND_PROP_VISIBLE) < 1:
            thoat = True
            break
    cap.release()
    if not thoat and khung is not None:              # hết video: giữ khung cuối để còn xem / chụp
        print(f"  Hết video ({so_chup} lần chụp). Vẫn bấm CHỤP được khung cuối; phím bất kỳ để tiếp tục.")
        hien_video(khung, giay, True)
        while True:
            phim = cv2.waitKey(30) & 0xFF
            if phim in (ord("c"), ord("C")) or tt["chup"]:
                tt["chup"] = False
                chup(khung, giay)
            elif phim != 255 or cv2.getWindowProperty(CUA_SO_VIDEO, cv2.WND_PROP_VISIBLE) < 1:
                thoat = phim in (27, ord("q"))
                break
    try:
        cv2.destroyWindow(CUA_SO_VIDEO)
    except cv2.error:
        pass
    return thoat


# =============================================================================== riêng test_kg
def xu_ly_mot_khung(anh_goc, seg, model_kg, tt, da_cat=False):
    """1 khung -> (ảnh đã vẽ, kết quả dict hoặc None, thông báo nếu không đo được)."""
    anh = anh_goc if da_cat else doi_ve_thang_train(anh_goc)
    H, W = anh.shape[:2]
    ra = anh.copy()
    if da_cat:
        anh_cat, hop, so_con = anh, (0, 0, W, H), 1
    else:
        cat = cat_quanh_lon(seg, anh)                # bước 1: pig_seg lượt 1 -> con chính -> cắt nới 12 %
        if cat is None:
            ghi_chu(ra, "Khong tim thay lon", (12, 40), (0, 0, 255), 1.0, 2)
            return ra, None, "Không tìm thấy con lợn nào."
        anh_cat, hop, _, so_con = cat
    mask, conf = mask_con_chinh(seg, anh_cat)        # bước 2: mask lưng trên ảnh cắt
    if mask is None:
        ghi_chu(ra, "Khong tim thay lon", (12, 40), (0, 0, 255), 1.0, 2)
        return ra, None, "Không tìm thấy con lợn nào (lượt 2)."
    # bước 3: 10 đặc trưng - cùng hàm, cùng kernel với lúc train; x_cam/y_cam cần góc cắt + kích thước khung
    dt, hh = dac_trung(mask, goc_cat=hop[:2], khung_wh=(W, H), ti_le_kernel=tt["ti_le_kernel_phep_mo"],
                       chi_tiet=True)
    if hh is None:
        return ra, None, "Mask quá nhỏ, không đo được đặc trưng."
    kg = float(du_doan_kg(model_kg, tt, [dt])[0])    # bước 4

    x1, y1, x2, y2 = map(int, hop)
    cv2.rectangle(ra, (x1, y1), (x2, y2), (255, 0, 0), 2)
    ve_hinh_hoc(ra, mask, hh, (x1, y1))
    ghi_chu(ra, f"{kg:.1f} kg", (12, 42), (255, 255, 255), 1.2, 3)
    ghi_chu(ra, f"DT {dt['dien_tich']:.0f}  CV {dt['chu_vi']:.0f}  D {dt['dai']:.0f}  R {dt['rong']:.0f}  "
                f"D/R {dt['dai_rong']:.2f}", (12, 72), (255, 255, 255), 0.55, 1)
    ghi_chu(ra, f"x_cam {dt['x_cam']:+.2f}  y_cam {dt['y_cam']:+.2f}  x_tam {dt['x_tam']:.3f}  "
                f"dac {dt['do_dac']:.3f}  cong {dt['do_cong']:.3f}", (12, 96), (255, 255, 255), 0.55, 1)
    return ra, {"kg": kg, "dt": dt, "conf": conf, "mask": mask, "so_con": so_con, "khung": (W, H)}, None


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Test mô hình cân nặng lợn trên ảnh / video (10 đặc trưng)")
    ap.add_argument("anh", nargs="+", help="đường dẫn ảnh hoặc video (nhiều file được)")
    ap.add_argument("--da-cat", action="store_true", help="ảnh đã cắt sẵn quanh 1 con -> bỏ bước tự cắt (lượt 1)")
    a = ap.parse_args()

    model_kg, tt = nap_model_kg()
    seg = YOLO(str(MODEL_SEG))
    print(f"Mô hình kg: hạng {tt.get('hang_tham_so')}, {len(tt['dac_trung_theo_thu_tu'])} đặc trưng, "
          f"MAE CV {tt['danh_gia_5fold_CV']['MAE_anh_TB_kg']:.2f} kg/ảnh")
    if a.da_cat:
        print("[Lưu ý] --da-cat: không biết vị trí con lợn trong khung gốc -> x_cam, y_cam tính như con nằm giữa "
              "ảnh cắt (kém chính xác hơn ảnh nguyên khung).")

    def du_doan_va_in(anh, da_cat=False, im_lang=False):
        """1 ảnh (hoặc 1 khung chụp từ video) -> dự đoán, in terminal -> list cửa sổ để hiện."""
        ra, kq, loi = xu_ly_mot_khung(anh, seg, model_kg, tt, da_cat)
        if im_lang:                                  # lần chạy mồi (nạp model) -> không in
            return []
        if kq is None:
            print(f"  {loi}")
        else:
            if kq["so_con"] > 1:
                print(f"  Thấy {kq['so_con']} con, chọn con chính (độ tin cậy cao, gần tâm ảnh).")
            print(f"  Khung đo: {kq['khung'][0]}x{kq['khung'][1]} | độ tin cậy mask: {kq['conf']:.3f}")
            in_dac_trung(kq["dt"])
            print(f"  ==> CÂN NẶNG DỰ ĐOÁN: {kq['kg']:.1f} kg")
            for c in canh_bao_kg(kq["kg"], tt, kq["mask"], kq["conf"]):
                print(c)
        return [(CUA_SO, ra)]

    for duong_dan in a.anh:
        if Path(duong_dan).suffix.lower() in DUOI_VIDEO:
            lan_dau = [True]

            def du_doan_khung(khung):
                im = lan_dau[0]                      # lần gọi đầu là chạy mồi trên ảnh đen
                lan_dau[0] = False
                return du_doan_va_in(khung, im_lang=im)
            if phat_video(duong_dan, du_doan_khung):
                break
            continue

        anh = doc_anh(duong_dan)
        if anh is None:
            print(f"[Bỏ qua] Không đọc được ảnh: {duong_dan}")
            continue
        print(f"\n{duong_dan}  ({anh.shape[1]}x{anh.shape[0]})")
        hien_cac_cua_so(du_doan_va_in(anh, a.da_cat))
        if cv2.waitKey(0) & 0xFF in (27, ord("q")):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
