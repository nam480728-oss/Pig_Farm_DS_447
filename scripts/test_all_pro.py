"""TEST TOÀN BỘ DỰ ÁN trong 1 lần: ảnh / video -> pig_det -> tư thế -> pig_seg + 10 đặc trưng -> cân nặng.
Không lưu gì: in ra terminal + hiện ảnh. Video: phát bình thường, chỉ dự đoán khung nào bấm CHỤP.

Cách chạy (đứng ở thư mục gốc dự án):
    python scripts/test_all_pro.py "D:/anh/chuong.jpg"            # ảnh: phím bất kỳ = ảnh tiếp, ESC/q = thoát
    python scripts/test_all_pro.py anh1.jpg anh2.png video.mp4    # trộn ảnh + video
    python scripts/test_all_pro.py "D:/video/chuong.mp4"          # video: phát bình thường, bấm nút CHỤP (phím C)
                                                                  #   -> dự đoán khung đang xem như 1 ảnh
    python scripts/test_all_pro.py anh.jpg --nguong 0.9           # khắt khe hơn khi coi là "đứng"

Quy trình mỗi khung hình:
    1. pig_det.pt         -> hộp từng con lợn trên khung GỐC          (models/Pig_Det/models/pig_det.pt)
    2. pig_posture.pt     -> xác suất ĐỨNG từng con (cắt vuông trên khung gốc, như lúc train tư thế)
       chỉ con có P(đứng) >= --nguong (mặc định 0.8) mới được cân; con nằm / không chắc -> bỏ qua
    -- từ đây đổi khung về rộng 960 px (giữ tỉ lệ) = thang pixel ảnh train cân nặng (RGB_9579 960x540) --
    3. pig_seg.pt         -> mask lưng trên ảnh cắt quanh hộp (nới 12 %, giống lúc tạo dữ liệu cân nặng)
    4. 10 đặc trưng       -> lay_dac_trung.dac_trung (cùng hàm, cùng kernel với lúc train)
    5. weight.json        -> kg từng con

Cửa sổ 1 (khung hình): xanh lá = được cân (+ kg), cam = đứng nhưng dưới ngưỡng (không cân),
                       đỏ = không đứng (không cân), tím = đứng nhưng không đo được mask.
Cửa sổ 2 (từng con được cân): ảnh cắt + mask, trục SVD, đường sống lưng, kg, xác suất đứng.
Video: bấm CHỤP nhiều lần được, mỗi lần kết quả mới thay kết quả cũ; SPACE = dừng (lúc dừng vẫn chụp được).

LƯU Ý: kg chỉ đáng tin với camera giống dữ liệu train (thẳng đứng, ~1,88 m, ảnh 16:9). Ảnh PigDetect / camera
khác góc, độ cao -> pipeline vẫn chạy để kiểm tra từng bước, nhưng con số kg tuyệt đối có thể lệch nhiều.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

import test_kg as K                     # import ultralytics trước xgboost (xem test_kg.py) + các hàm cân nặng
import test_tu_the as TT                # tìm lợn (pig_det) + cắt vuông, phân loại tư thế
from lay_dac_trung import CONF_TOI_THIEU, IMGSZ, NOI_HOP, chon_con_chinh, dac_trung

CUA_SO = "Toan bo - pig_det > tu the > can nang | phim bat ky: anh tiep, SPACE: dung, ESC/q: thoat"
CUA_SO_LUOI = "Tung con duoc can"
XANH, CAM, DO, TIM = (0, 200, 0), (0, 165, 255), (0, 0, 255), (200, 0, 200)


def doc_tham_so():
    ap = argparse.ArgumentParser(description="Test toàn bộ: pig_det -> tư thế -> pig_seg + đặc trưng -> cân nặng")
    ap.add_argument("anh", nargs="+", help="đường dẫn ảnh hoặc video (nhiều file được)")
    ap.add_argument("--nguong", type=float, default=0.8, help="xác suất ĐỨNG tối thiểu để được cân")
    ap.add_argument("--conf-tim", type=float, default=0.25, help="ngưỡng tin cậy của pig_det khi tìm lợn")
    return ap.parse_args()


def hop_cat(b, W, H):
    """Hộp pig_det -> hộp cắt nới 12 % mỗi phía (giống cat_quanh_lon lúc tạo dữ liệu cân nặng)."""
    x1, y1, x2, y2 = map(float, b)
    pw, ph = int((x2 - x1) * NOI_HOP), int((y2 - y1) * NOI_HOP)
    return max(0, int(x1) - pw), max(0, int(y1) - ph), min(W, int(x2) + pw), min(H, int(y2) + ph)


class ToanBo:
    """Giữ 3 model + model kg, xử lý 1 khung hình -> kết quả từng con + ảnh để hiện."""

    def __init__(self, a):
        from ultralytics import YOLO
        self.a = a
        self.kg, self.tt = K.nap_model_kg()
        if not TT.MODEL_TU_THE.exists():
            sys.exit(f"Không thấy {TT.MODEL_TU_THE}. Hãy chạy scripts/train_tu_the.py trước.")
        self.tu_the = YOLO(str(TT.MODEL_TU_THE))
        self.seg = YOLO(str(K.MODEL_SEG))
        if not TT.pig_det_dung_duoc():
            print("[Lưu ý] Chưa có pig_det.pt thật -> tạm tìm lợn bằng pig_seg (chuồng đông có thể sót con).")
        print(f"Ngưỡng đứng: {a.nguong} | mô hình kg hạng {self.tt.get('hang_tham_so')}, "
              f"MAE CV {self.tt['danh_gia_5fold_CV']['MAE_anh_TB_kg']:.2f} kg/ảnh, "
              f"{self.tt['danh_gia_5fold_CV']['MAE_con_TB_kg']:.2f} kg/con (trung vị nhiều khung)")

    def xu_ly(self, anh_goc):
        """1 khung -> (ảnh khung đã vẽ, ảnh lưới từng con được cân hoặc None, list kết quả từng con)."""
        anh = K.doi_ve_thang_train(anh_goc)          # khung rộng 960: CHỈ dùng cho phần cân nặng (bước 3-5)
        H, W = anh.shape[:2]
        tl = W / anh_goc.shape[1]                    # hệ số đổi tọa độ khung gốc -> khung 960
        ra = anh.copy()

        # 1. pig_det trên KHUNG GỐC (như test_pig_det.py) -> hộp từng con
        hop_goc = TT.tim_lon(anh_goc, self.a.conf_tim)
        if len(hop_goc) == 0:
            K.ghi_chu(ra, "Khong tim thay lon", (12, 40), DO, 1.0, 2)
            return ra, None, []

        # 2. Tư thế trên KHUNG GỐC (giống lúc train tư thế + test_tu_the.py): cắt vuông -> phân loại cả lô 1 lần
        kq_tt = self.tu_the.predict([TT.cat_vuong(anh_goc, b) for b in hop_goc], imgsz=TT.KICH_THUOC, verbose=False)
        hop = hop_goc * tl                           # hộp đổi sang khung 960 để cắt cho pig_seg + vẽ
        ds = [{"so": i, "hop": b, "p_dung": float(r.probs.data[0])} for i, (b, r) in enumerate(zip(hop, kq_tt), 1)]
        dung = [c for c in ds if c["p_dung"] >= self.a.nguong]

        # 3-4. Con đứng: cắt nới 12 % -> pig_seg (cả lô) -> chọn con chính trong ảnh cắt -> 10 đặc trưng
        if dung:
            for c in dung:
                c["cat"] = hop_cat(c["hop"], W, H)
            anh_cat = [anh[y1:y2, x1:x2] for x1, y1, x2, y2 in (c["cat"] for c in dung)]
            kq_seg = self.seg.predict(anh_cat, imgsz=IMGSZ, conf=CONF_TOI_THIEU, retina_masks=True, verbose=False)
            co_dt = []
            for c, ac, r in zip(dung, anh_cat, kq_seg):
                mask, conf = chon_con_chinh(r)
                if mask is None:
                    continue
                dt, hh = dac_trung(mask, goc_cat=c["cat"][:2], khung_wh=(W, H),
                                   ti_le_kernel=self.tt["ti_le_kernel_phep_mo"], chi_tiet=True)
                if hh is None:
                    continue
                c.update(mask=mask, conf_seg=conf, dt=dt, hh=hh, anh_cat=ac)
                co_dt.append(c)
            # 5. Cân nặng (cả lô)
            for c, kg in zip(co_dt, K.du_doan_kg(self.kg, self.tt, [c["dt"] for c in co_dt])):
                c["kg"] = float(kg)

        # Vẽ khung
        for c in ds:
            x1, y1, x2, y2 = map(int, c["hop"])
            if "kg" in c:
                mau, nhan = XANH, f"#{c['so']} {c['kg']:.1f}kg"
                K.ve_hinh_hoc(ra, c["mask"], c["hh"], c["cat"][:2], mau_vien=XANH, day_du=False)
            elif c["p_dung"] >= self.a.nguong:
                mau, nhan = TIM, f"#{c['so']} khong do duoc"
            elif c["p_dung"] >= 0.5:
                mau, nhan = CAM, f"#{c['so']} dung? {c['p_dung']:.2f}"
            else:
                mau, nhan = DO, f"#{c['so']} khong dung"
            cv2.rectangle(ra, (x1, y1), (x2, y2), mau, 2)
            K.ghi_chu(ra, nhan, (x1 + 3, max(18, y1 + 18)), mau if mau != XANH else (255, 255, 255), 0.5, 1)
        so_can = sum("kg" in c for c in ds)
        K.ghi_chu(ra, f"{len(ds)} con | dung {len(dung)} (P>={self.a.nguong}) | can duoc {so_can}", (10, 28),
                  (255, 255, 255), 0.7, 2)
        return ra, self.luoi(ds), ds

    def luoi(self, ds, o=300, so_cot=4):
        """Lưới ảnh cắt các con ĐƯỢC CÂN: mask + trục SVD + sống lưng + kg + xác suất đứng."""
        can = [c for c in ds if "kg" in c]
        if not can:
            return None
        ra = []
        for c in can:
            v = c["anh_cat"].copy()
            K.ve_hinh_hoc(v, c["mask"], c["hh"], (0, 0))
            tl = min(o / v.shape[1], (o - 40) / v.shape[0])
            v = cv2.resize(v, (max(1, int(v.shape[1] * tl)), max(1, int(v.shape[0] * tl))))
            khung = np.zeros((o, o, 3), np.uint8)
            khung[:v.shape[0], :v.shape[1]] = v
            K.ghi_chu(khung, f"#{c['so']}  {c['kg']:.1f} kg", (6, 24), (255, 255, 255), 0.7, 2)
            K.ghi_chu(khung, f"P(dung) {c['p_dung']:.2f}  dac {c['dt']['do_dac']:.2f}  cong {c['dt']['do_cong']:.3f}",
                      (6, o - 12), (255, 255, 255), 0.45, 1)
            cv2.rectangle(khung, (0, 0), (o - 1, o - 1), XANH, 2)
            ra.append(khung)
        so_cot = min(so_cot, len(ra))
        while len(ra) % so_cot:
            ra.append(np.zeros_like(ra[0]))
        return np.vstack([np.hstack(ra[k:k + so_cot]) for k in range(0, len(ra), so_cot)])


def in_chi_tiet(ds, tt, nguong):
    """Ảnh: in từng con ra terminal."""
    for c in ds:
        if "kg" in c:
            print(f"  #{c['so']:<3d} ĐỨNG {c['p_dung']:.3f} -> {c['kg']:6.1f} kg | {K.dong_dac_trung(c['dt'])}")
            for cb in K.canh_bao_kg(c["kg"], tt, c["mask"], c["conf_seg"], tien_to="        "):
                print(cb)
        elif c["p_dung"] >= nguong:
            print(f"  #{c['so']:<3d} ĐỨNG {c['p_dung']:.3f} -> không đo được mask, không cân")
        else:
            tt_ = "đứng nhưng dưới ngưỡng" if c["p_dung"] >= 0.5 else "không đứng"
            print(f"  #{c['so']:<3d} {tt_} (P đứng {c['p_dung']:.3f}) -> không cân")
    can = [c["kg"] for c in ds if "kg" in c]
    print(f"  ==> {len(ds)} con | {sum(c['p_dung'] >= nguong for c in ds)} con đứng | cân được {len(can)} con"
          + (f" | TB {np.mean(can):.1f} kg (min {min(can):.1f}, max {max(can):.1f})" if can else ""))


def du_doan_va_in(tb, anh, nguong, im_lang=False):
    """1 ảnh (hoặc 1 khung CHỤP từ video) -> chạy toàn bộ pipeline, in terminal -> list cửa sổ để hiện.

    Cửa sổ lưới = None khi không con nào được cân -> K.hien_cac_cua_so sẽ đóng lưới cũ."""
    ra, luoi, ds = tb.xu_ly(anh)
    if im_lang:                                      # lần chạy mồi (nạp model lên GPU) -> không in
        return []
    if not ds:
        print("  Không tìm thấy con lợn nào.")
    in_chi_tiet(ds, tb.tt, nguong)
    return [(CUA_SO, ra), (CUA_SO_LUOI, luoi)]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    a = doc_tham_so()
    tb = ToanBo(a)

    for duong_dan in a.anh:
        if Path(duong_dan).suffix.lower() in K.DUOI_VIDEO:
            # Video phát bình thường; chỉ khi bấm CHỤP (hoặc phím C) mới đưa khung đang xem vào dự đoán
            lan_dau = [True]

            def du_doan_khung(khung):
                im = lan_dau[0]                      # lần gọi đầu là chạy mồi trên ảnh đen
                lan_dau[0] = False
                return du_doan_va_in(tb, khung, a.nguong, im_lang=im)
            if K.phat_video(duong_dan, du_doan_khung):
                break
            continue

        anh = K.doc_anh(duong_dan)
        if anh is None:
            print(f"[Bỏ qua] Không đọc được ảnh: {duong_dan}")
            continue
        print(f"\n{duong_dan}  ({anh.shape[1]}x{anh.shape[0]})")
        K.hien_cac_cua_so(du_doan_va_in(tb, anh, a.nguong))
        if cv2.waitKey(0) & 0xFF in (27, ord("q")):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
