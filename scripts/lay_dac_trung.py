"""Lấy 5 đặc trưng hình dạng lưng lợn từ mask của pig_seg.pt -> data/data_cat_lung_tinh_kg/dac_trung.csv

5 đặc trưng (đơn vị PIXEL, đo bằng OpenCV, không cần train):
    dien_tich  diện tích bên trong đường viền lưng          cv2.contourArea
    chu_vi     độ dài đường viền                             cv2.arcLength
    dai        cạnh dài của hình chữ nhật xoay ôm sát thân   cv2.minAreaRect
    rong       cạnh ngắn của hình chữ nhật đó                cv2.minAreaRect
    lech_tam   độ lệch tâm ellipse khớp thân (0 tròn, 1 dài) cv2.fitEllipse

File này dùng theo 2 cách:
  1) Chạy trực tiếp (1 lần) để tạo CSV cho train cân nặng:
         python scripts/lay_dac_trung.py                  # toàn bộ 9.578 ảnh RGB_9579 (~45 phút trên RTX 4060)
         python scripts/lay_dac_trung.py --gioi-han 50    # chạy thử 50 ảnh, ghi ra file _thu.csv
         python scripts/lay_dac_trung.py --mau 12         # thêm ảnh mẫu để kiểm tra bằng mắt
  2) Import trong pipeline lúc chạy thật (ảnh đã được pig_det cắt sẵn):
         from lay_dac_trung import mask_con_chinh, dac_trung
         mask, conf = mask_con_chinh(model_seg, anh_cat)
         x = dac_trung(mask)                               # dict 5 đặc trưng -> đưa vào weight.json
"""
import argparse
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

GOC = Path(__file__).resolve().parents[1]
MODEL_SEG = GOC / "models" / "Pig_Seg" / "pig_seg.pt"
THU_MUC_ANH = GOC / "data" / "data_cat_lung_tinh_kg" / "RGB_9579"
CSV_RA = GOC / "data" / "data_cat_lung_tinh_kg" / "dac_trung.csv"
THU_MUC_MAU = GOC / "data" / "kiem_tra_anh"

NOI_HOP = 0.12        # nới hộp 12 % mỗi cạnh: GIỐNG HỆT lúc tạo dữ liệu train pig_seg và lúc cắt theo pig_det
TI_LE_KERNEL = 0.50   # kernel phép mở = 50 % bề rộng thân -> xóa tai, chân, đuôi; chọn bằng thử nghiệm 0..0.8 (xem docstring bo_tai_chan_duoi)
CONF_TOI_THIEU = 0.25 # mask có độ tin cậy thấp hơn bị bỏ qua hoàn toàn
IMGSZ = 640           # giống lúc train pig_seg

TEN_5_DAC_TRUNG = ["dien_tich", "chu_vi", "dai", "rong", "lech_tam"]


# =============================================================================== phần dùng chung
def chon_con_chinh(kq):
    """Trong các mask tìm được, chọn con lợn CHÍNH: độ tin cậy cao và nằm gần tâm ảnh.

    điểm = conf × (1 − khoảng cách từ tâm mask tới tâm ảnh, chuẩn hóa 0..1)
    -> con lợn ở mép ảnh (bị cắt một phần) dù conf cao vẫn thua con nằm giữa.
    Trả về (mask nhị phân uint8 0/1, conf) hoặc (None, 0.0) nếu không có con nào.
    """
    if kq.masks is None or len(kq.masks) == 0:
        return None, 0.0
    h, w = kq.orig_shape
    tot_nhat, diem_max = None, -1.0
    for m, conf in zip(kq.masks.data.cpu().numpy(), kq.boxes.conf.cpu().numpy()):
        m = (m > 0.5).astype(np.uint8)
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            continue
        # khoảng cách tâm mask -> tâm ảnh, chia cho nửa đường chéo để nằm trong 0..1
        d = math.hypot(xs.mean() - w / 2, ys.mean() - h / 2) / math.hypot(w / 2, h / 2)
        diem = float(conf) * (1 - d)
        if diem > diem_max:
            tot_nhat, diem_max, conf_chon = m, diem, float(conf)
    return (tot_nhat, conf_chon) if tot_nhat is not None else (None, 0.0)


def mask_con_chinh(model, anh):
    """Chạy pig_seg trên 1 ảnh (đã cắt quanh 1 con) -> (mask 0/1 cùng kích thước ảnh, conf)."""
    # retina_masks=True: mask đúng kích thước ảnh đưa vào, đo pixel mới chính xác
    kq = model.predict(anh, imgsz=IMGSZ, conf=CONF_TOI_THIEU, retina_masks=True, verbose=False)[0]
    return chon_con_chinh(kq)


def vien_lon_nhat(mask):
    """Đường viền ngoài lớn nhất của mask (bỏ các mảnh vụn nhỏ rời rạc)."""
    vien, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(vien, key=cv2.contourArea) if vien else None


def do_5_dac_trung(vien):
    """Đo 5 đặc trưng từ 1 đường viền. Trả về dict (giá trị NaN nếu viền quá nhỏ)."""
    if vien is None or len(vien) < 5:          # fitEllipse cần tối thiểu 5 điểm
        return {k: float("nan") for k in TEN_5_DAC_TRUNG}
    (_, _), (w, h), _ = cv2.minAreaRect(vien)  # hình chữ nhật xoay nhỏ nhất ôm sát thân
    (_, _), (t1, t2), _ = cv2.fitEllipse(vien) # t1, t2 = độ dài 2 trục ellipse (đường kính)
    a, b = max(t1, t2) / 2, min(t1, t2) / 2    # bán trục dài a, bán trục ngắn b
    return {
        "dien_tich": float(cv2.contourArea(vien)),
        "chu_vi": float(cv2.arcLength(vien, True)),
        "dai": float(max(w, h)),
        "rong": float(min(w, h)),
        "lech_tam": float(math.sqrt(1 - (b / a) ** 2)) if a > 0 else float("nan"),
    }


def bo_tai_chan_duoi(mask, ti_le_kernel=TI_LE_KERNEL):
    """Phép MỞ hình thái = co (erode) rồi giãn (dilate) với kernel hình ellipse.

    Phần nào HẸP hơn kernel (tai, chân, đuôi) bị xóa khi co và không mọc lại khi giãn;
    phần thân to hơn kernel thì co rồi giãn lại gần như nguyên hình.
    Kernel tỉ lệ theo bề rộng thân -> lợn to hay nhỏ, gần hay xa camera đều cắt tương đương.
    Trả về (mask đã mở, kích thước kernel px).

    Chọn tỉ lệ 0.5 bằng thử nghiệm trên 711 ảnh RGB_9579 (10 ảnh/con, 73 con; hồi quy tuyến tính 5 đặc trưng,
    chia 5 fold theo con -> MAE chỉ để so sánh tương đối giữa các tỉ lệ):
        tỉ lệ   bỏ DT   CV DT 1 con   r(DT,kg)   MAE     ảnh bị xóa sạch
        0 (thô)  0 %      6.34 %       0.977     5.29 kg   0
        0.2      0.6 %    6.05 %       0.977     4.71 kg   0
        0.4      2.3 %    5.64 %       0.980     4.64 kg   0
        0.5      3.4 %    5.70 %       0.982     4.60 kg   0     <- chọn: tốt nhất mà không mất ảnh nào
        0.6      5.6 %    5.33 %       0.985     4.83 kg   2
        0.7      9.9 %    4.13 %       0.991     4.70 kg   10    (bắt đầu ăn vào thân)
        0.8     18.1 %   23.18 %       0.969     6.54 kg   64
    """
    vien = vien_lon_nhat(mask)
    if vien is None:
        return mask, 0
    rong = min(cv2.minAreaRect(vien)[1])
    k = max(3, int(round(rong * ti_le_kernel)) | 1)   # | 1 -> luôn là số lẻ (kernel có tâm)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel), k


def dac_trung(mask, ti_le_kernel=TI_LE_KERNEL, ca_tho=False):
    """HÀM CHÍNH: mask lưng lợn -> 5 đặc trưng (sau khi bỏ tai, chân, đuôi).

    ca_tho=True: trả thêm 5 đặc trưng của mask thô (chưa mở), tên có hậu tố _tho, và kích thước kernel.
    Lúc train và lúc chạy thật PHẢI gọi cùng hàm này với cùng ti_le_kernel.
    """
    mask_mo, k = bo_tai_chan_duoi(mask, ti_le_kernel)
    kq = do_5_dac_trung(vien_lon_nhat(mask_mo))
    if ca_tho:
        kq.update({f"{t}_tho": v for t, v in do_5_dac_trung(vien_lon_nhat(mask)).items()})
        kq["kernel_px"] = k
    return kq


# =============================================================================== phần tạo CSV từ RGB_9579
def cat_quanh_lon(model, anh):
    """Ảnh nguyên khung (chưa có pig_det) -> cắt quanh con chính, nới 12 %.

    Lượt 1: pig_seg trên ảnh nguyên khung để tìm con chính -> lấy hộp bao của mask.
    Đây là bước THAY THẾ pig_det; lúc chạy thật pig_det sẽ đưa ra hộp này.
    Trả về (ảnh cắt, (x1, y1, x2, y2), conf lượt 1, số con thấy ở lượt 1) hoặc None.
    """
    kq = model.predict(anh, imgsz=IMGSZ, conf=CONF_TOI_THIEU, retina_masks=True, verbose=False)[0]
    mask, conf = chon_con_chinh(kq)
    if mask is None:
        return None
    x, y, w, h = cv2.boundingRect(mask)
    H, W = anh.shape[:2]
    pw, ph = int(w * NOI_HOP), int(h * NOI_HOP)
    x1, y1 = max(0, x - pw), max(0, y - ph)
    x2, y2 = min(W, x + w + pw), min(H, y + h + ph)
    return anh[y1:y2, x1:x2], (x1, y1, x2, y2), conf, len(kq.masks)


def cham_bien(mask):
    """Mask có chạm mép ảnh cắt không -> có thể con lợn bị cắt cụt (ra ngoài khung hình)."""
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def do_net(anh):
    """Độ nét = phương sai Laplacian của ảnh xám. Thấp = ảnh nhòe (lợn đang chạy)."""
    return float(cv2.Laplacian(cv2.cvtColor(anh, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def xu_ly_1_anh(model, duong_dan):
    """Toàn bộ quy trình cho 1 ảnh RGB_9579 -> 1 dòng CSV (+ dữ liệu để vẽ ảnh mẫu)."""
    con = duong_dan.parent.name                       # vd 101.74_127  (kg_số ảnh)
    dong = {
        "anh": duong_dan.relative_to(THU_MUC_ANH).as_posix(),
        "fold": duong_dan.parent.parent.name,
        "con": con,
        "kg": float(con.split("_")[0]),               # cân nặng thật lấy từ tên thư mục
    }
    anh = cv2.imread(str(duong_dan))
    cat = cat_quanh_lon(model, anh) if anh is not None else None
    if cat is None:                                   # không tìm thấy lợn -> giữ dòng, đánh dấu
        dong.update(khong_thay_lon=True, conf=0.0, conf_luot1=0.0, so_lon_luot1=0, cham_bien=False,
                    do_net=float("nan"))
        return dong, None

    anh_cat, hop, conf1, so_lon = cat
    mask, conf = mask_con_chinh(model, anh_cat)       # lượt 2: giống hệt lúc chạy thật
    if mask is None:
        dong.update(khong_thay_lon=True, conf=0.0, conf_luot1=conf1, so_lon_luot1=so_lon, cham_bien=False,
                    do_net=do_net(anh_cat))
        return dong, None

    dong.update(khong_thay_lon=False, conf=conf, conf_luot1=conf1, so_lon_luot1=so_lon,
                cham_bien=cham_bien(mask), do_net=do_net(anh_cat),
                x1=hop[0], y1=hop[1], x2=hop[2], y2=hop[3])
    dong.update(dac_trung(mask, ca_tho=True))
    return dong, (anh_cat, mask)


def ve_mau(anh_cat, mask, dong):
    """Ảnh kiểm tra: đỏ = mask thô, xanh lá = sau khi bỏ tai/chân/đuôi, vàng = hình chữ nhật dài × rộng."""
    ve = anh_cat.copy()
    mask_mo, _ = bo_tai_chan_duoi(mask)
    for m, mau in ((mask, (0, 0, 255)), (mask_mo, (0, 255, 0))):
        v = vien_lon_nhat(m)
        if v is not None:
            cv2.drawContours(ve, [v], -1, mau, 2)
    v = vien_lon_nhat(mask_mo)
    if v is not None and len(v) >= 5:
        cv2.drawContours(ve, [np.intp(cv2.boxPoints(cv2.minAreaRect(v)))], -1, (0, 255, 255), 1)
        cv2.ellipse(ve, cv2.fitEllipse(v), (255, 0, 255), 1)
    # co ảnh vừa ô 400 x 270 (giữ tỉ lệ), chừa 30 px phía dưới để ghi chữ
    tl = min(400 / ve.shape[1], 270 / ve.shape[0])
    ve = cv2.resize(ve, (int(ve.shape[1] * tl), int(ve.shape[0] * tl)))
    o = np.zeros((300, 400, 3), np.uint8)
    o[:ve.shape[0], :ve.shape[1]] = ve
    cv2.putText(o, f"{dong['kg']}kg DT {dong['dien_tich']:.0f} (tho {dong['dien_tich_tho']:.0f})",
                (5, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return o


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Lấy 5 đặc trưng từ ảnh RGB_9579 -> dac_trung.csv")
    ap.add_argument("--gioi-han", type=int, default=0, help="chỉ chạy N ảnh ngẫu nhiên (để thử), ghi ra *_thu.csv")
    ap.add_argument("--mau", type=int, default=0, help="lưu N ảnh mẫu vào data/kiem_tra_anh/dac_trung_mau.jpg")
    a = ap.parse_args()

    if not MODEL_SEG.exists():
        sys.exit(f"Không thấy {MODEL_SEG}. Hãy train cắt lưng trước (scripts/train_cat_lung.py).")
    import pandas as pd
    from ultralytics import YOLO
    model = YOLO(str(MODEL_SEG))

    ds = sorted(THU_MUC_ANH.rglob("*.png"))
    random.seed(0)
    if a.gioi_han:
        ds = sorted(random.sample(ds, min(a.gioi_han, len(ds))))
    chon_mau = set(random.sample(range(len(ds)), min(a.mau, len(ds))))
    print(f"Model : {MODEL_SEG}\nẢnh   : {len(ds)} ảnh trong {THU_MUC_ANH}")

    dong_csv, mau, bat_dau = [], [], time.time()
    for i, f in enumerate(ds):
        dong, ve = xu_ly_1_anh(model, f)
        dong_csv.append(dong)
        if i in chon_mau and ve is not None:
            mau.append(ve_mau(*ve, dong))
        if (i + 1) % 200 == 0 or i + 1 == len(ds):
            da = time.time() - bat_dau
            print(f"  {i + 1}/{len(ds)} ảnh | {da / 60:.1f} phút | còn khoảng {da / (i + 1) * (len(ds) - i - 1) / 60:.1f} phút")

    bang = pd.DataFrame(dong_csv)
    # sắp cột: thông tin ảnh -> cờ chất lượng -> 5 đặc trưng chính -> 5 đặc trưng thô -> hộp cắt
    cot = ["anh", "fold", "con", "kg", "khong_thay_lon", "conf", "conf_luot1", "so_lon_luot1", "cham_bien",
           "do_net", "kernel_px"] + TEN_5_DAC_TRUNG + [f"{t}_tho" for t in TEN_5_DAC_TRUNG] + ["x1", "y1", "x2", "y2"]
    bang = bang.reindex(columns=cot)
    ra = CSV_RA.with_name("dac_trung_thu.csv") if a.gioi_han else CSV_RA
    bang.to_csv(ra, index=False, encoding="utf-8-sig")  # utf-8-sig: mở bằng Excel không lỗi font

    if mau:
        while len(mau) % 4:
            mau.append(np.zeros_like(mau[0]))
        THU_MUC_MAU.mkdir(exist_ok=True)
        cv2.imwrite(str(THU_MUC_MAU / "dac_trung_mau.jpg"),
                    np.vstack([np.hstack(mau[j:j + 4]) for j in range(0, len(mau), 4)]))
        print(f"Ảnh mẫu: {THU_MUC_MAU / 'dac_trung_mau.jpg'}")

    # ------------------------------------------------------------------ tóm tắt để kiểm tra nhanh
    tot = bang[~bang.khong_thay_lon.astype(bool)]
    con = tot.groupby("con").agg(kg=("kg", "first"), dt=("dien_tich", "median"), dt_tho=("dien_tich_tho", "median"))
    print(f"\nĐã lưu: {ra}  ({len(bang)} dòng)")
    print(f"Không thấy lợn       : {int(bang.khong_thay_lon.sum())}")
    print(f"conf < 0.5           : {int((tot.conf < 0.5).sum())}")
    print(f"Mask chạm mép ảnh cắt: {int(tot.cham_bien.sum())}")
    print(f"Diện tích bị bỏ (tai/chân/đuôi) trung bình: {(1 - tot.dien_tich / tot.dien_tich_tho).mean() * 100:.1f} %")
    if len(con) > 2:
        print(f"Tương quan diện tích (trung vị mỗi con) với kg: sau mở r = {con[['dt', 'kg']].corr().iloc[0, 1]:.3f}"
              f" | thô r = {con[['dt_tho', 'kg']].corr().iloc[0, 1]:.3f}  ({len(con)} con)")


if __name__ == "__main__":
    main()
