"""Lấy 10 đặc trưng hình dạng lưng lợn từ mask của pig_seg.pt -> data/data_cat_lung_tinh_kg/dac_trung.csv

Tất cả đo trên MASK ĐÃ BỎ TAI, CHÂN, ĐUÔI (phép mở, xem bo_tai_chan_duoi). Đơn vị pixel trừ khi ghi khác.
    Kích thước
      dien_tich  diện tích mask (số pixel bên trong đường viền)            cv2.contourArea
      chu_vi     độ dài đường viền                                          cv2.arcLength
      dai        chiều dài thân dọc TRỤC CHÍNH SVD (xa nhất 2 phía)
      rong       chiều rộng thân dọc TRỤC PHỤ SVD
      dai_rong   dai / rong (thân thon dài hay tròn)
    Vị trí trong khung hình (bù méo ống kính / góc nhìn: con ở mép khung trông khác con ở giữa)
      x_cam      (x trọng tâm mask trong khung / rộng khung) - 0.5    -> -0.5 mép trái .. 0 giữa .. 0.5 mép phải
      y_cam      (y trọng tâm mask trong khung / cao khung) - 0.5     -> -0.5 mép trên .. 0 giữa .. 0.5 mép dưới
    Hình dáng (cho mô hình biết số đo kích thước đang bị HỤT bao nhiêu để bù kg)
      x_tam      |vị trí trọng tâm dọc trục chính (0 = mút này, 1 = mút kia) - 0.5|
                 -> khối thịt dồn về đầu/mông. Lấy trị tuyệt đối vì SVD không biết đâu là đầu, đâu là đuôi
      do_dac     diện tích mask / diện tích bao lồi (0..1; thân thẳng gọn ~0.95, cong/lõm thấp hơn)
      do_cong    độ lệch lớn nhất của đường sống lưng so với đoạn thẳng nối 2 đầu / dai (0 = thẳng)

File này dùng theo 2 cách:
  1) Chạy trực tiếp (1 lần) để tạo CSV cho train cân nặng:
         python scripts/lay_dac_trung.py                  # toàn bộ 9.578 ảnh RGB_9579 (~45 phút trên RTX 4060)
         python scripts/lay_dac_trung.py --gioi-han 50    # chạy thử 50 ảnh, ghi ra file _thu.csv
         python scripts/lay_dac_trung.py --mau 12         # thêm ảnh mẫu để kiểm tra bằng mắt
  2) Import trong pipeline lúc chạy thật (ảnh đã được pig_det cắt sẵn):
         from lay_dac_trung import mask_con_chinh, dac_trung
         mask, conf = mask_con_chinh(model_seg, anh_cat)
         x = dac_trung(mask, goc_cat=(x1, y1), khung_wh=(W, H))   # dict 10 đặc trưng -> đưa vào weight.json
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

# Độ cong sống lưng
SO_LAT = 20           # số lát cắt ngang thân dọc trục chính -> 20 điểm giữa tạo thành đường sống lưng
BO_DAU = 0.05         # bỏ 5 % chiều dài ở MỖI đầu (mõm, mông nhọn -> lát cắt ít pixel, nhiễu)

TEN_DAC_TRUNG = ["dien_tich", "chu_vi", "dai", "rong", "dai_rong", "x_cam", "y_cam", "x_tam", "do_dac", "do_cong"]


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


def be_day_than(mask):
    """Bề dày thân (px) = đường kính hình tròn LỚN NHẤT lọt trong mask = 2 × khoảng cách xa nhất từ mép vào trong.

    Distance transform: mỗi pixel trong mask = khoảng cách tới mép gần nhất; điểm lớn nhất nằm giữa chỗ thân
    dày nhất (bụng). Tai, chân, đuôi mỏng nên không ảnh hưởng; thân thẳng hay cong đều cho cùng 1 kết quả.
    Mask không có gì -> 0."""
    vien = vien_lon_nhat(mask)
    if vien is None:
        return 0.0
    m = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(m, [vien], -1, 1, cv2.FILLED)    # chỉ mảng lớn nhất, lấp lỗ bên trong
    m = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)   # mask chạm mép ảnh vẫn đo đúng
    return 2.0 * float(cv2.distanceTransform(m, cv2.DIST_L2, 5).max())


def bo_tai_chan_duoi(mask, ti_le_kernel=TI_LE_KERNEL):
    """Phép MỞ hình thái = co (erode) rồi giãn (dilate) với kernel hình ellipse.

    Phần nào HẸP hơn kernel (tai, chân, đuôi) bị xóa khi co và không mọc lại khi giãn;
    phần thân to hơn kernel thì co rồi giãn lại gần như nguyên hình.
    Kernel = ti_le_kernel × BỀ DÀY THÂN (be_day_than) -> lợn to hay nhỏ, gần hay xa camera đều cắt tương đương.
    Trả về (mask đã mở, kích thước kernel px).

    Bản trước lấy bề rộng = cạnh ngắn hình chữ nhật xoay (minAreaRect). Con CONG MẠNH thì hình chữ nhật phải ôm
    cả chữ C nên cạnh ngắn rất to -> kernel quá lớn -> xóa mất thân, chỉ còn 1 hình tròn (vd. 74.4kg_66.png mất 71 %
    diện tích). Bề dày thân đo bằng distance transform không phụ thuộc thân thẳng hay cong.

    Chọn tỉ lệ 0.5 bằng thử nghiệm trên 711 ảnh RGB_9579 (10 ảnh/con, 73 con; hồi quy tuyến tính,
    chia 5 fold theo con -> MAE chỉ để so sánh tương đối giữa các tỉ lệ; đo với cách tính bề rộng CŨ - con thẳng
    thì 2 cách gần bằng nhau nên tỉ lệ vẫn dùng được):
        tỉ lệ   bỏ DT   CV DT 1 con   r(DT,kg)   MAE     ảnh bị xóa sạch
        0 (thô)  0 %      6.34 %       0.977     5.29 kg   0
        0.2      0.6 %    6.05 %       0.977     4.71 kg   0
        0.4      2.3 %    5.64 %       0.980     4.64 kg   0
        0.5      3.4 %    5.70 %       0.982     4.60 kg   0     <- chọn: tốt nhất mà không mất ảnh nào
        0.6      5.6 %    5.33 %       0.985     4.83 kg   2
        0.7      9.9 %    4.13 %       0.991     4.70 kg   10    (bắt đầu ăn vào thân)
        0.8     18.1 %   23.18 %       0.969     6.54 kg   64
    """
    rong = be_day_than(mask)
    if rong == 0:
        return mask, 0
    k = max(3, int(round(rong * ti_le_kernel)) | 1)   # | 1 -> luôn là số lẻ (kernel có tâm)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel), k


def mask_than(mask_mo):
    """Chỉ giữ mảng thân lớn nhất (phép mở đôi khi để lại mảnh vụn rời) -> mask 0/1 + đường viền."""
    vien = vien_lon_nhat(mask_mo)
    if vien is None:
        return None, None
    m = np.zeros(mask_mo.shape, np.uint8)
    cv2.drawContours(m, [vien], -1, 1, cv2.FILLED)
    return m, vien


def truc_svd(m):
    """SVD trên tọa độ các pixel của mask -> trục chính (dọc sống lưng) và trục phụ (ngang thân).

    Trả về tâm (cx, cy) = trọng tâm mask, v1, v2 = 2 vector đơn vị, u, w = tọa độ từng pixel chiếu lên v1, v2
    (gốc tại trọng tâm, nên trung bình u = 0)."""
    ys, xs = np.nonzero(m)
    p = np.column_stack([xs, ys]).astype(np.float64)
    tam = p.mean(axis=0)
    q = p - tam
    _, _, vt = np.linalg.svd(q, full_matrices=False)   # hàng 0 = hướng trải rộng nhất = trục chính
    v1, v2 = vt[0], vt[1]
    return tam, v1, v2, q @ v1, q @ v2


def duong_song_lung(u, w, u_min, u_max):
    """Cắt thân thành SO_LAT lát dọc trục chính (đã bỏ BO_DAU mỗi đầu); mỗi lát lấy điểm giữa mép trên/mép dưới.

    Trả về mảng (n, 2) các điểm (u, w) của đường sống lưng, đã làm mượt trung bình 3 điểm liền kề."""
    L = u_max - u_min
    bien = np.linspace(u_min + BO_DAU * L, u_max - BO_DAU * L, SO_LAT + 1)
    lat = np.digitize(u, bien) - 1                       # lát số mấy (ngoài khoảng -> -1 hoặc SO_LAT)
    diem = []
    for i in range(SO_LAT):
        chon = lat == i
        if chon.sum() < 3:                               # lát gần như rỗng -> bỏ
            continue
        wi = w[chon]
        diem.append(((bien[i] + bien[i + 1]) / 2, (wi.min() + wi.max()) / 2))
    diem = np.array(diem)
    if len(diem) >= 3:                                   # làm mượt: mỗi điểm = TB với 2 điểm bên cạnh
        mu = diem[:, 1].copy()
        mu[1:-1] = (diem[:-2, 1] + diem[1:-1, 1] + diem[2:, 1]) / 3
        diem[:, 1] = mu
    return diem


def do_lech_lon_nhat(diem):
    """Khoảng cách VUÔNG GÓC lớn nhất từ các điểm sống lưng tới đoạn thẳng nối điểm đầu - điểm cuối.

    Trả về (độ lệch px, chỉ số điểm lệch xa nhất)."""
    a, b = diem[0], diem[-1]
    ab = b - a
    dai_ab = np.hypot(*ab)
    if dai_ab == 0:
        return 0.0, 0
    # |tích có hướng| / độ dài đoạn = khoảng cách vuông góc từ điểm tới đường thẳng
    kc = np.abs(ab[0] * (diem[:, 1] - a[1]) - ab[1] * (diem[:, 0] - a[0])) / dai_ab
    j = int(kc.argmax())
    return float(kc[j]), j


def dac_trung(mask, goc_cat=None, khung_wh=None, ti_le_kernel=TI_LE_KERNEL, chi_tiet=False):
    """HÀM CHÍNH: mask lưng lợn (trong ảnh cắt) -> dict 10 đặc trưng (đo trên mask đã bỏ tai, chân, đuôi).

    goc_cat  = (x1, y1): góc trên-trái của ảnh cắt trong khung hình gốc   } cần cho x_cam, y_cam;
    khung_wh = (W, H)  : kích thước khung hình gốc                        } thiếu -> 2 cột này NaN
    chi_tiet=True: trả thêm (dict, hình học) để vẽ ảnh kiểm tra.
    Lúc train và lúc chạy thật PHẢI gọi cùng hàm này với cùng ti_le_kernel.
    """
    nan = {k: float("nan") for k in TEN_DAC_TRUNG}
    mask_mo, k = bo_tai_chan_duoi(mask, ti_le_kernel)
    m, vien = mask_than(mask_mo)
    if m is None or len(vien) < 5 or m.sum() < 50:      # mask quá nhỏ / bị xóa sạch
        return (nan, None) if chi_tiet else nan

    # --- Kích thước: diện tích, chu vi trên viền; dài, rộng theo SVD
    dien_tich = float(cv2.contourArea(vien))
    tam, v1, v2, u, w = truc_svd(m)
    u_min, u_max, w_min, w_max = u.min(), u.max(), w.min(), w.max()
    dai, rong = float(u_max - u_min + 1), float(w_max - w_min + 1)   # +1: tính cả pixel ở 2 mút

    # --- Vị trí trong khung: trọng tâm mask đổi sang tọa độ khung gốc
    if goc_cat is not None and khung_wh is not None:
        x_cam = (goc_cat[0] + tam[0]) / khung_wh[0] - 0.5
        y_cam = (goc_cat[1] + tam[1]) / khung_wh[1] - 0.5
    else:
        x_cam = y_cam = float("nan")

    # --- Hình dáng
    x_tam = abs((0.0 - u_min) / (u_max - u_min) - 0.5)                # trọng tâm có u = 0
    bao_loi = cv2.convexHull(vien)
    do_dac = dien_tich / max(cv2.contourArea(bao_loi), 1e-9)
    song_lung = duong_song_lung(u, w, u_min, u_max)
    if len(song_lung) >= 3:
        lech, j = do_lech_lon_nhat(song_lung)
        do_cong = lech / dai
    else:
        lech, j, do_cong = float("nan"), 0, float("nan")

    kq = {"dien_tich": dien_tich, "chu_vi": float(cv2.arcLength(vien, True)), "dai": dai, "rong": rong,
          "dai_rong": dai / rong, "x_cam": float(x_cam), "y_cam": float(y_cam), "x_tam": float(x_tam),
          "do_dac": float(do_dac), "do_cong": float(do_cong)}
    if not chi_tiet:
        return kq
    hh = dict(mask_mo=m, vien=vien, bao_loi=bao_loi, tam=tam, v1=v1, v2=v2, u_min=u_min, u_max=u_max,
              w_min=w_min, w_max=w_max, song_lung=song_lung, j_lech=j, lech=lech, kernel_px=k)
    return kq, hh


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


def xu_ly_1_anh(model, duong_dan, can_ve=False):
    """Toàn bộ quy trình cho 1 ảnh RGB_9579 -> 1 dòng CSV (+ dữ liệu để vẽ ảnh mẫu nếu can_ve)."""
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

    H, W = anh.shape[:2]
    dong.update(khong_thay_lon=False, conf=conf, conf_luot1=conf1, so_lon_luot1=so_lon,
                cham_bien=cham_bien(mask), do_net=do_net(anh_cat),
                x1=hop[0], y1=hop[1], x2=hop[2], y2=hop[3], khung_rong=W, khung_cao=H)
    dt, hh = dac_trung(mask, goc_cat=hop[:2], khung_wh=(W, H), chi_tiet=True)
    dong.update(dt)
    v_tho = vien_lon_nhat(mask)                       # chỉ để thống kê % diện tích bị bỏ (tai/chân/đuôi),
    dong["dien_tich_tho"] = float(cv2.contourArea(v_tho)) if v_tho is not None else float("nan")  # cùng cách đo
    dong["kernel_px"] = hh["kernel_px"] if hh else float("nan")
    return dong, ((anh, anh_cat, hop, mask, hh) if can_ve and hh else None)


# =============================================================================== ảnh kiểm tra bằng mắt
def ve_mau(anh, anh_cat, hop, mask, hh, dong):
    """1 ô kiểm tra = [khung gốc thu nhỏ (vị trí x_cam, y_cam)] + [ảnh cắt có vẽ hình học] + 3 dòng số.

    Ảnh cắt: đỏ = mask thô, xanh lá = thân sau khi bỏ tai/chân/đuôi, tím nhạt = bao lồi (độ đặc),
    vàng = 2 trục SVD (dài × rộng), xanh dương = trọng tâm, trắng = mút trục chính (đo x_tam),
    xanh ngọc = đường sống lưng, hồng = đoạn thẳng nối 2 đầu, đỏ đậm = độ lệch lớn nhất (độ cong)."""
    ve = anh_cat.copy()
    v_tho = vien_lon_nhat(mask)
    cv2.drawContours(ve, [v_tho], -1, (0, 0, 255), 1)
    cv2.drawContours(ve, [hh["bao_loi"]], -1, (255, 160, 255), 1)
    cv2.drawContours(ve, [hh["vien"]], -1, (0, 255, 0), 2)

    tam, v1, v2 = hh["tam"], hh["v1"], hh["v2"]
    diem = lambda uu, ww: tuple(np.intp(np.round(tam + uu * v1 + ww * v2)))   # (u, w) -> pixel ảnh cắt
    cv2.line(ve, diem(hh["u_min"], 0), diem(hh["u_max"], 0), (0, 255, 255), 1)    # trục chính (dài)
    cv2.line(ve, diem(0, hh["w_min"]), diem(0, hh["w_max"]), (0, 255, 255), 1)    # trục phụ (rộng)
    for uu in (hh["u_min"], hh["u_max"]):
        cv2.circle(ve, diem(uu, 0), 4, (255, 255, 255), -1)                       # 2 mút trục chính
    cv2.circle(ve, diem(0, 0), 5, (255, 0, 0), -1)                                # trọng tâm

    sl = hh["song_lung"]
    if len(sl) >= 3:
        pts = np.array([diem(a, b) for a, b in sl])
        cv2.polylines(ve, [pts], False, (255, 255, 0), 2)                         # đường sống lưng
        cv2.line(ve, tuple(pts[0]), tuple(pts[-1]), (180, 105, 255), 2)           # đoạn nối 2 đầu
        # chân đường vuông góc từ điểm lệch xa nhất xuống đoạn nối 2 đầu
        a, b, p = sl[0], sl[-1], sl[hh["j_lech"]]
        t = np.dot(p - a, b - a) / max(np.dot(b - a, b - a), 1e-9)
        chan = a + t * (b - a)
        cv2.line(ve, diem(*p), diem(*chan), (0, 0, 200), 3)

    # ô bên trái: khung gốc thu nhỏ + hộp cắt + chấm trọng tâm (để nhìn x_cam, y_cam)
    H, W = anh.shape[:2]
    nho = cv2.resize(anh, (200, int(200 * H / W)))
    tl_n = 200 / W
    cv2.rectangle(nho, (int(hop[0] * tl_n), int(hop[1] * tl_n)), (int(hop[2] * tl_n), int(hop[3] * tl_n)),
                  (0, 255, 255), 1)
    cv2.drawMarker(nho, (100, nho.shape[0] // 2), (255, 255, 255), cv2.MARKER_CROSS, 10, 1)   # tâm khung
    cv2.circle(nho, (int((hop[0] + tam[0]) * tl_n), int((hop[1] + tam[1]) * tl_n)), 3, (255, 0, 0), -1)

    # ghép: [khung nhỏ 200 | ảnh cắt 300] cao 200, dưới là 3 dòng chữ
    tl = min(300 / ve.shape[1], 200 / ve.shape[0])
    ve = cv2.resize(ve, (int(ve.shape[1] * tl), int(ve.shape[0] * tl)))
    o = np.zeros((272, 500, 3), np.uint8)
    o[:nho.shape[0], :200] = nho
    o[:ve.shape[0], 200:200 + ve.shape[1]] = ve
    d = dong
    dong_chu = [f"{d['kg']}kg  DT {d['dien_tich']:.0f}  CV {d['chu_vi']:.0f}  D {d['dai']:.0f}  R {d['rong']:.0f}"
                f"  D/R {d['dai_rong']:.2f}",
                f"x_cam {d['x_cam']:+.2f}  y_cam {d['y_cam']:+.2f}  x_tam {d['x_tam']:.3f}",
                f"do_dac {d['do_dac']:.3f}  do_cong {d['do_cong']:.3f} (lech {hh['lech']:.0f}px)"]
    for i, s in enumerate(dong_chu):
        cv2.putText(o, s, (5, 222 + i * 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.rectangle(o, (0, 0), (499, 271), (90, 90, 90), 1)
    return o


def chon_mau_da_dang(ds, so_mau):
    """Chọn ảnh mẫu trải đều các con (mỗi con tối đa 1 ảnh) để thấy đủ kiểu: nhỏ/to, thẳng/cong, giữa/mép khung."""
    theo_con = {}
    for i, f in enumerate(ds):
        theo_con.setdefault(f.parent.name, []).append(i)
    con = random.sample(sorted(theo_con), min(so_mau, len(theo_con)))
    return {random.choice(theo_con[c]) for c in con}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Lấy 10 đặc trưng từ ảnh RGB_9579 -> dac_trung.csv")
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
    chon_mau = chon_mau_da_dang(ds, a.mau) if a.mau else set()
    print(f"Model : {MODEL_SEG}\nẢnh   : {len(ds)} ảnh trong {THU_MUC_ANH}")

    dong_csv, mau, bat_dau = [], [], time.time()
    for i, f in enumerate(ds):
        dong, ve = xu_ly_1_anh(model, f, can_ve=i in chon_mau)
        dong_csv.append(dong)
        if ve is not None:
            mau.append((dong["kg"], ve_mau(*ve, dong)))
        if (i + 1) % 200 == 0 or i + 1 == len(ds):
            da = time.time() - bat_dau
            print(f"  {i + 1}/{len(ds)} ảnh | {da / 60:.1f} phút | còn khoảng {da / (i + 1) * (len(ds) - i - 1) / 60:.1f} phút")

    bang = pd.DataFrame(dong_csv)
    # sắp cột: thông tin ảnh -> cờ chất lượng -> 10 đặc trưng -> thống kê phụ -> hộp cắt + kích thước khung
    cot = (["anh", "fold", "con", "kg", "khong_thay_lon", "conf", "conf_luot1", "so_lon_luot1", "cham_bien",
            "do_net", "kernel_px"] + TEN_DAC_TRUNG + ["dien_tich_tho", "x1", "y1", "x2", "y2", "khung_rong", "khung_cao"])
    bang = bang.reindex(columns=cot)
    ra = CSV_RA.with_name("dac_trung_thu.csv") if a.gioi_han else CSV_RA
    bang.to_csv(ra, index=False, encoding="utf-8-sig")  # utf-8-sig: mở bằng Excel không lỗi font

    if mau:
        mau = [o for _, o in sorted(mau, key=lambda t: t[0])]   # xếp theo kg tăng dần cho dễ so sánh
        while len(mau) % 3:
            mau.append(np.zeros_like(mau[0]))
        THU_MUC_MAU.mkdir(exist_ok=True)
        cv2.imwrite(str(THU_MUC_MAU / "dac_trung_mau.jpg"),
                    np.vstack([np.hstack(mau[j:j + 3]) for j in range(0, len(mau), 3)]))
        print(f"Ảnh mẫu: {THU_MUC_MAU / 'dac_trung_mau.jpg'}")

    # ------------------------------------------------------------------ tóm tắt để kiểm tra nhanh
    tot = bang[~bang.khong_thay_lon.astype(bool)]
    print(f"\nĐã lưu: {ra}  ({len(bang)} dòng)")
    print(f"Không thấy lợn       : {int(bang.khong_thay_lon.sum())}")
    print(f"Thiếu đặc trưng (NaN): {int(tot[TEN_DAC_TRUNG].isna().any(axis=1).sum())}")
    print(f"conf < 0.5           : {int((tot.conf < 0.5).sum())}")
    print(f"Mask chạm mép ảnh cắt: {int(tot.cham_bien.sum())}")
    print(f"Diện tích bị bỏ (tai/chân/đuôi) trung bình: {(1 - tot.dien_tich / tot.dien_tich_tho).mean() * 100:.1f} %")
    print("\nPhân bố 10 đặc trưng (trung vị [5 % .. 95 %]) và tương quan với kg (trung vị mỗi con):")
    con = tot.groupby("con")[TEN_DAC_TRUNG + ["kg"]].median()
    for c in TEN_DAC_TRUNG:
        q = tot[c].quantile([0.05, 0.5, 0.95]).values
        r = con[[c, "kg"]].corr().iloc[0, 1] if len(con) > 2 else float("nan")
        print(f"  {c:10s} {q[1]:10.3f}  [{q[0]:10.3f} .. {q[2]:10.3f}]   r(kg) = {r:+.3f}")


if __name__ == "__main__":
    main()
