"""Test thực tế mô hình tư thế ĐỨNG / KHÔNG ĐỨNG: đưa ảnh vào -> hiện kết quả lên màn hình (không lưu file).

Cách chạy (đứng ở thư mục gốc dự án):
    python scripts/test_tu_the.py "D:/anh/chuong.jpg"          # ảnh NGUYÊN KHUNG: tự tìm từng con rồi phân loại
    python scripts/test_tu_the.py anh1.jpg anh2.jpg            # nhiều ảnh, phím bất kỳ = ảnh tiếp, ESC/q = thoát
    python scripts/test_tu_the.py "D:/video/chuong.mp4"        # VIDEO: phát realtime, SPACE = dừng, ESC/q = thoát
    python scripts/test_tu_the.py chuong.jpg --hop 120 80 400 260   # ảnh nguyên khung + hộp con lợn (x1 y1 x2 y2) tự nhập
    python scripts/test_tu_the.py lon.jpg --da-cat             # ảnh ĐÃ CẮT sẵn quanh 1 con (vd. data/data_posture/dung/*.jpg)
    python scripts/test_tu_the.py data/data_posture_final_ok/test/dung/xxx.jpg --da-xu-ly   # ảnh 224 đã xử lý
    python scripts/test_tu_the.py anh.jpg --nguong 0.9         # khắt khe hơn khi coi là "đứng"

4 chế độ:
  - mặc định (ảnh nguyên khung): tìm lợn bằng models/Pig_Det/models/pig_det.pt; CHƯA có (hoặc chỉ là con trỏ
    Git LFS) thì tạm dùng pig_seg.pt
    (pig_seg được train trên ảnh 1 con -> ảnh chuồng đông / PigDetect thường KHÔNG tìm ra, khi đó dùng --hop).
    Mỗi hộp -> cắt vuông -> phân loại.
  - --hop x1 y1 x2 y2: ảnh nguyên khung + hộp tự nhập (pixel) -> chỉ phân loại con đó.
  - --da-cat  : ảnh đã cắt quanh 1 con. Nếu là ảnh trong data/data_posture (có trong danh_sach*.csv) thì tự tìm lại
                ảnh gốc + hộp -> cắt ĐÚNG như lúc train. Ảnh cắt ngoài thì coi cả ảnh là hộp; ảnh cắt dài/hẹp phải
                đệm phản chiếu nhiều -> kết quả kém tin cậy hơn (nên dùng ảnh nguyên khung + --hop).
  - --da-xu-ly: ảnh vuông 224 lấy từ train/ val/ test/ -> đưa thẳng vào model.

Ảnh luôn được cắt bằng ĐÚNG hàm cat_vuong() của phần xử lý dữ liệu (data/data_posture_final_ok/tu_the_chung.py)
-> giống hệt lúc train. Kết quả "dung" chỉ khi xác suất đứng >= --nguong (mặc định 0.8, xem danh_gia_tu_the.md mục 5).
Trên ảnh: xanh lá = đứng (được cân), đỏ = không đứng (bỏ qua), cam = đứng nhưng dưới ngưỡng (không chắc).
Ảnh nguyên khung / --hop: hiện 2 cửa sổ cùng lúc: (1) ảnh gốc có hộp từng con, (2) lưới ảnh cắt từng con (đúng ảnh
224 model nhìn thấy) + số thứ tự + kết quả. Không lưu file nào.
Video (.mp4 .avi .mov .mkv ...): phát đúng tốc độ thật, mỗi khung hiển thị đều tìm lợn + phân loại lại; máy xử lý
không kịp thì tự bỏ bớt khung (video vẫn chạy đúng thời gian, chỉ giật hơn). Terminal in tóm tắt mỗi 1 giây.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

GOC = Path(__file__).resolve().parents[1]
MODEL_TU_THE = GOC / "models" / "Pig_Posture" / "pig_posture.pt"
MODEL_DET = GOC / "models" / "Pig_Det" / "models" / "pig_det.pt"   # YOLO11l nhóm khác train trên PigDetect, ảnh 960
MODEL_SEG = GOC / "models" / "Pig_Seg" / "pig_seg.pt"

# Gọi lại hàm cắt ảnh của phần xử lý dữ liệu (không viết lại) -> ảnh test giống hệt ảnh train
sys.path.insert(0, str(GOC / "data" / "data_posture_final_ok"))
from tu_the_chung import KICH_THUOC, cat_vuong  # noqa: E402

XANH, DO, CAM = (0, 200, 0), (0, 0, 255), (0, 165, 255)
DUOI_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v", ".mpg", ".mpeg"}
CUA_SO_CHINH = "Tu the lon - phim bat ky: anh tiep, ESC/q: thoat"
CUA_SO_LUOI = "Tung con (anh dua vao model)"


def doc_tham_so():
    ap = argparse.ArgumentParser(description="Test mô hình tư thế đứng / không đứng trên ảnh thật")
    ap.add_argument("anh", nargs="+", help="đường dẫn 1 hoặc nhiều ảnh")
    ap.add_argument("--model", default=str(MODEL_TU_THE))
    ap.add_argument("--nguong", type=float, default=0.8, help="xác suất đứng tối thiểu để coi là 'dung'")
    ap.add_argument("--hop", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"), help="hộp con lợn trong ảnh nguyên khung")
    ap.add_argument("--da-cat", action="store_true", help="ảnh đã cắt sẵn quanh 1 con")
    ap.add_argument("--da-xu-ly", action="store_true", help="ảnh vuông 224 đã xử lý (train/val/test)")
    ap.add_argument("--conf-tim", type=float, default=0.25, help="ngưỡng tin cậy khi tìm lợn trong ảnh nguyên khung")
    return ap.parse_args()


def doc_anh(p):
    """imdecode + fromfile: đọc được đường dẫn có dấu tiếng Việt trên Windows."""
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR) if Path(p).exists() else None


def phan_loai(model, anh_vuong, nguong):
    """Ảnh vuông 224 -> (nhãn, xác suất đứng). Nhãn 'dung' chỉ khi xác suất đứng >= ngưỡng."""
    r = model.predict(anh_vuong, imgsz=KICH_THUOC, verbose=False)[0]
    p_dung = float(r.probs.data[0])                  # lớp 0 = dung, 1 = khong_dung (theo tên thư mục lúc train)
    return ("dung" if p_dung >= nguong else "khong_dung"), p_dung


def mau_nhan(nhan, p_dung):
    if nhan == "dung":
        return XANH
    return CAM if p_dung >= 0.5 else DO             # đứng theo model nhưng chưa đủ ngưỡng -> cam


def ghi_chu(anh, chu, xy, mau, co=0.7, day=2):
    cv2.putText(anh, chu, xy, cv2.FONT_HERSHEY_SIMPLEX, co, (0, 0, 0), day + 3)
    cv2.putText(anh, chu, xy, cv2.FONT_HERSHEY_SIMPLEX, co, mau, day)


def pig_det_dung_duoc():
    """pig_det.pt có thật không. File Git LFS chưa tải chỉ là 1 file chữ ~130 byte bắt đầu bằng 'version https://git-lfs'."""
    if not MODEL_DET.exists():
        return False
    if MODEL_DET.stat().st_size < 1_000_000:
        print(f"[Lưu ý] {MODEL_DET.relative_to(GOC)} chỉ {MODEL_DET.stat().st_size} byte -> là con trỏ Git LFS, chưa phải "
              f"model thật (cần 'git lfs pull' hoặc chép file ~51 MB).")
        return False
    return True


def tim_lon(anh, conf):
    """Tìm hộp từng con trong ảnh nguyên khung bằng pig_det (YOLO11l, ảnh 960).

    Chưa có pig_det thật thì tạm dùng pig_seg (train trên ảnh 1 con -> chuồng đông thường không tìm ra)."""
    from ultralytics import YOLO

    if not hasattr(tim_lon, "model"):
        duong_dan = MODEL_DET if pig_det_dung_duoc() else MODEL_SEG
        tim_lon.model = YOLO(str(duong_dan))
        tim_lon.la_det = duong_dan == MODEL_DET
        print(f"[Tìm lợn] dùng {duong_dan.relative_to(GOC)}")
        if not tim_lon.la_det:
            print("[Lưu ý] Tạm tìm lợn bằng pig_seg.pt (chuồng đông có thể sót con) -> có thể dùng --hop.")
    imgsz = 960 if tim_lon.la_det else 640
    r = tim_lon.model.predict(anh, conf=conf, imgsz=imgsz, verbose=False)[0]
    return r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))


def tim_anh_goc(ten_file):
    """Ảnh cắt trong data/data_posture -> (khung gốc, hộp, nới) để cắt lại đúng như lúc train; không thấy -> None."""
    import pandas as pd

    if not hasattr(tim_anh_goc, "bang"):
        ds = [GOC / "data" / "data_posture" / f for f in ["danh_sach.csv", "danh_sach_ung_vien_dung.csv"]]
        tim_anh_goc.bang = (pd.concat([pd.read_csv(f) for f in ds if f.exists()]).drop_duplicates("file")
                            .set_index("file") if any(f.exists() for f in ds) else None)
    b = tim_anh_goc.bang
    if b is None or ten_file not in b.index:
        return None
    r = b.loc[ten_file]
    if r.trai == "rgb9579":
        import pandas as pd
        hop = pd.read_csv(GOC / "data" / "data_cat_lung_tinh_kg" / "dac_trung.csv").set_index("anh").loc[
            r.anh_goc, ["x1", "y1", "x2", "y2"]].values
        return doc_anh(GOC / "data" / "data_cat_lung_tinh_kg" / "RGB_9579" / r.anh_goc), hop, 0.0
    from cat_anh_tu_the import PD, hop_pigdetect
    f = PD / r.tap_goc / "images" / r.anh_goc
    khung = doc_anh(f)
    return khung, hop_pigdetect(f, khung.shape[1], khung.shape[0])[int(r.hop)], 0.10


def ket_qua_mot_con(anh_vuong, nhan, p_dung, nguong):
    """Ảnh 224 đưa vào model + nhãn ghi lên trên (để xem model đã 'nhìn' gì)."""
    ra = cv2.resize(anh_vuong, (448, 448), interpolation=cv2.INTER_NEAREST)
    mau = mau_nhan(nhan, p_dung)
    cv2.rectangle(ra, (0, 0), (447, 447), mau, 6)
    ghi_chu(ra, f"{'DUNG' if nhan == 'dung' else 'KHONG DUNG'}", (12, 40), mau, 1.1, 3)
    ghi_chu(ra, f"P(dung) = {p_dung:.3f}  (nguong {nguong})", (12, 430), (255, 255, 255), 0.6, 2)
    return ra


def luoi_anh_cat(ds_o, so_cot=6):
    """Ghép ảnh cắt từng con (ảnh 224 model nhìn thấy) thành 1 lưới, mỗi ô ghi số thứ tự + kết quả.

    ds_o: list (anh_vuong, so_thu_tu, nhan, p_dung). Viền ô cùng màu với hộp trên ảnh gốc."""
    o = []
    for vuong, i, nhan, p in ds_o:
        mau = mau_nhan(nhan, p)
        v = vuong.copy()
        cv2.rectangle(v, (0, 0), (KICH_THUOC - 1, KICH_THUOC - 1), mau, 4)
        ghi_chu(v, f"#{i} {'DUNG' if nhan == 'dung' else 'KHONG'}", (8, 26), mau, 0.7, 2)
        ghi_chu(v, f"P={p:.2f}", (8, KICH_THUOC - 10), (255, 255, 255), 0.6, 2)
        o.append(v)
    so_cot = min(so_cot, len(o))
    while len(o) % so_cot:                           # hàng cuối thiếu -> thêm ô đen cho đủ
        o.append(np.zeros_like(o[0]))
    return np.vstack([np.hstack(o[k:k + so_cot]) for k in range(0, len(o), so_cot)])


def vua_man_hinh(anh, rong_toi_da=1400, cao_toi_da=850):
    """Thu nhỏ ảnh quá to để cửa sổ không tràn màn hình (chỉ để xem)."""
    h, w = anh.shape[:2]
    tl = min(rong_toi_da / w, cao_toi_da / h, 1.0)
    return cv2.resize(anh, (int(w * tl), int(h * tl))) if tl < 1 else anh


def xu_ly_khung(anh, model, a, hop=None):
    """1 ảnh nguyên khung -> tìm lợn (hoặc dùng hop có sẵn) -> cắt vuông -> phân loại cả lô 1 lần.

    Trả về (ảnh đã vẽ hộp, ds_o = list (ảnh vuông, số thứ tự, nhãn, xác suất đứng)). Dùng chung cho ảnh và video."""
    if hop is None:
        hop = tim_lon(anh, a.conf_tim)
    ra = anh.copy()
    h = anh.shape[0]
    if len(hop) == 0:
        ghi_chu(ra, "Khong tim thay lon", (10, 40), DO, 1.2, 3)
        return ra, []
    ds_vuong = [cat_vuong(anh, b) for b in hop]
    kq = model.predict(ds_vuong, imgsz=KICH_THUOC, verbose=False)   # cả lô 1 lần -> nhanh hơn từng con (video)
    ds_o = []
    for i, (b, vuong, r) in enumerate(zip(hop, ds_vuong, kq), 1):
        p = float(r.probs.data[0])                   # lớp 0 = dung
        nhan = "dung" if p >= a.nguong else "khong_dung"
        ds_o.append((vuong, i, nhan, p))
        mau = mau_nhan(nhan, p)
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(ra, (x1, y1), (x2, y2), mau, 3)
        ghi_chu(ra, f"#{i} {'DUNG' if nhan == 'dung' else 'KHONG'} {p:.2f}", (x1 + 4, max(24, y1 + 24)), mau)
    so_dung = sum(n == "dung" for _, _, n, _ in ds_o)
    ghi_chu(ra, f"Dung: {so_dung}/{len(ds_o)} con", (10, h - 15), (255, 255, 255), 0.9, 2)
    return ra, ds_o


def hien_luoi(ds_o):
    """Cửa sổ thứ 2: lưới ảnh cắt từng con. Không có con nào -> đóng cửa sổ lưới cũ (nếu đang mở)."""
    if ds_o:
        cv2.imshow(CUA_SO_LUOI, vua_man_hinh(luoi_anh_cat(ds_o)))
        return
    try:
        cv2.destroyWindow(CUA_SO_LUOI)
    except cv2.error:
        pass


def chay_video(duong_dan, model, a):
    """Phát video ĐÚNG tốc độ thật: máy xử lý chậm hơn video thì bỏ bớt khung để không bị trễ.

    Mỗi khung hiển thị đều được tìm lợn + phân loại lại. Trả về True nếu người dùng bấm ESC/q (thoát hẳn)."""
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
    print("  Phím: SPACE = tạm dừng / chạy tiếp, ESC/q = thoát")

    # Chạy thử 1 lần trên ảnh đen để nạp model lên GPU trước -> khung đầu không bị giật
    xu_ly_khung(np.zeros((h or 480, w or 640, 3), np.uint8), model, a)

    so_doc = so_xu_ly = 0
    moc_in = -1.0                                    # in tóm tắt ra terminal mỗi 1 giây video
    t0 = None                                        # thời điểm (đồng hồ thật) ứng với giây 0 của video
    thoat = False
    while True:
        ok, khung = cap.read()
        if not ok:
            break
        giay = so_doc / fps                          # vị trí khung này trong video (giây)
        so_doc += 1
        if t0 is None:
            t0 = time.perf_counter()
        # Máy chậm hơn video -> bỏ qua khung đã trễ (chỉ grab, không giải mã) cho kịp thời gian thật
        tre = (time.perf_counter() - t0) - giay
        if tre > 1.0 / fps:
            so_bo = int(tre * fps)
            for _ in range(so_bo):
                if not cap.grab():
                    break
            so_doc += so_bo
            continue

        ra, ds_o = xu_ly_khung(khung, model, a)
        so_xu_ly += 1
        ghi_chu(ra, f"{giay:5.1f}s / {tong / fps:.1f}s", (10, 30), (255, 255, 255), 0.8, 2)
        cv2.imshow(CUA_SO_CHINH, vua_man_hinh(ra))
        hien_luoi(ds_o)
        if giay - moc_in >= 1.0:
            moc_in = giay
            so_dung = sum(n == "dung" for _, _, n, _ in ds_o)
            print(f"  [{giay:6.1f}s] {len(ds_o):2d} con, {so_dung:2d} con ĐỨNG")

        # Máy nhanh hơn video -> chờ cho đúng tốc độ thật
        cho = giay + 1.0 / fps - (time.perf_counter() - t0)
        phim = cv2.waitKey(max(1, int(cho * 1000))) & 0xFF
        if phim == ord(" "):                         # tạm dừng
            t_dung = time.perf_counter()
            phim = cv2.waitKey(0) & 0xFF
            t0 += time.perf_counter() - t_dung       # bù thời gian dừng để không bị coi là trễ
        if phim in (27, ord("q")):
            thoat = True
            break
    cap.release()
    print(f"  Hết video: đọc {so_doc} khung, xử lý {so_xu_ly} khung "
          f"({so_xu_ly / max(1, so_doc) * 100:.0f}%, phần còn lại bỏ qua để giữ tốc độ thật)")
    return thoat


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    a = doc_tham_so()
    if not Path(a.model).exists():
        sys.exit(f"Không thấy model: {a.model}. Hãy chạy scripts/train_tu_the.py trước.")
    from ultralytics import YOLO
    model = YOLO(a.model)

    for duong_dan in a.anh:
        if Path(duong_dan).suffix.lower() in DUOI_VIDEO:       # video -> phát realtime
            if chay_video(duong_dan, model, a):
                break                                         # bấm ESC/q trong video -> thoát hẳn
            continue
        anh = doc_anh(duong_dan)
        if anh is None:
            print(f"[Bỏ qua] Không đọc được ảnh: {duong_dan}")
            continue
        h, w = anh.shape[:2]
        print(f"\n{duong_dan}  ({w}x{h})")

        if a.da_xu_ly or a.da_cat:
            if a.da_xu_ly:
                if (w, h) != (KICH_THUOC, KICH_THUOC):
                    print(f"  [Cảnh báo] Ảnh {w}x{h} không phải {KICH_THUOC}x{KICH_THUOC} -> có thể chưa xử lý, "
                          f"nên dùng --da-cat.")
                vuong = cv2.resize(anh, (KICH_THUOC, KICH_THUOC))
            else:
                goc = tim_anh_goc(Path(duong_dan).name)
                if goc is not None:
                    khung, hop, noi = goc
                    print("  Tìm thấy ảnh gốc trong bộ dữ liệu -> cắt lại đúng như lúc train.")
                    vuong = cat_vuong(khung, hop, noi=noi, canh_trung_gian=280 if noi == 0.0 else None)
                else:
                    # Ảnh cắt ngoài bộ dữ liệu (đã nới sẵn) -> cả ảnh là hộp, không nới thêm
                    if max(w, h) / min(w, h) > 1.6:
                        print("  [Cảnh báo] Ảnh cắt dài/hẹp -> phải đệm phản chiếu nhiều, khác ảnh lúc train. "
                              "Nên dùng ảnh nguyên khung + --hop.")
                    vuong = cat_vuong(anh, (0, 0, w, h), noi=0.0)
            nhan, p = phan_loai(model, vuong, a.nguong)
            print(f"  ==> {nhan.upper()}   (xác suất đứng {p:.3f}, ngưỡng {a.nguong})")
            ra = ket_qua_mot_con(vuong, nhan, p, a.nguong)
        else:
            hop = np.array([a.hop]) if a.hop else None
            ra, ds_o = xu_ly_khung(anh, model, a, hop)
            if not ds_o:
                print("  Không tìm thấy con lợn nào. Thử nhập hộp bằng --hop x1 y1 x2 y2.")
            for _, i, nhan, p in ds_o:
                print(f"  Con #{i}: {nhan:10s} xác suất đứng {p:.3f}")
            if ds_o:
                so_dung = sum(n == "dung" for _, _, n, _ in ds_o)
                print(f"  ==> {so_dung}/{len(ds_o)} con ĐỨNG (được đưa vào cân), ngưỡng {a.nguong}")
            hien_luoi(ds_o)

        cv2.imshow(CUA_SO_CHINH, vua_man_hinh(ra))
        if cv2.waitKey(0) & 0xFF in (27, ord("q")):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
