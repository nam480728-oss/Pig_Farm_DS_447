"""Hàm xử lý ảnh DÙNG CHUNG cho model tư thế đứng / không đứng (bản CẮT VUÔNG từ ảnh gốc).

Khác bản data_posture_final (cắt sát + thêm viền xám): ở đây hộp con lợn được nới 10 % rồi mở rộng
thành HÌNH VUÔNG ngay trên khung hình gốc -> phần dư là nền chuồng thật, KHÔNG có viền nhân tạo,
nên model không thể học vẹt theo độ dày viền (= tỉ lệ dài/rộng của hộp).

Phải dùng đúng hàm cat_vuong() ở mọi chỗ để ảnh lúc train và lúc chạy thật giống nhau:
  - data/data_posture_final_ok/chia_du_lieu.py (tạo train/val/test)
  - code test / pipeline camera: khung hình camera + hộp pig_det -> cat_vuong(khung, hop)
    sys.path.insert(0, "<gốc dự án>/data/data_posture_final_ok"); from tu_the_chung import cat_vuong
"""
import cv2

KICH_THUOC = 224    # ảnh vuông đưa vào YOLO-cls
NOI = 0.10          # nới hộp 10 % mỗi cạnh trước khi mở rộng thành vuông


def cat_vuong(khung, hop, kich_thuoc=KICH_THUOC, noi=NOI, canh_trung_gian=None):
    """Cắt 1 con lợn thành ảnh vuông kich_thuoc x kich_thuoc, thân nằm ngang.

    khung: ảnh gốc BGR (cả khung hình camera).  hop: (x1, y1, x2, y2) pixel của con lợn.
    1. Nới hộp `noi` mỗi cạnh, lấy cạnh vuông = cạnh dài của hộp đã nới, giữ tâm hộp.
    2. Cắt hình vuông đó từ khung gốc; phần vượt mép khung được đệm PHẢN CHIẾU nền (không dùng màu cố định).
    3. Hộp dọc (cao > rộng) -> xoay 90° để thân lợn nằm ngang (hướng ảnh không mang thông tin nhãn).
    4. (tuỳ chọn) canh_trung_gian: thu nhỏ về cạnh này trước (vd. ảnh RGB_9579 quá nét so với PigDetect).
    5. Resize về kich_thuoc.
    """
    x1, y1, x2, y2 = map(float, hop)
    w, h = x2 - x1, y2 - y1
    canh = max(w, h) * (1 + 2 * noi)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    a, b = int(round(cx - canh / 2)), int(round(cy - canh / 2))
    c, d = a + int(round(canh)), b + int(round(canh))

    H, W = khung.shape[:2]
    dem = (max(0, -b), max(0, d - H), max(0, -a), max(0, c - W))      # trên, dưới, trái, phải
    vung = khung[max(0, b):min(H, d), max(0, a):min(W, c)]
    if any(dem):
        vung = cv2.copyMakeBorder(vung, *dem, cv2.BORDER_REFLECT_101)

    if h > w:
        vung = cv2.rotate(vung, cv2.ROTATE_90_CLOCKWISE)
    if canh_trung_gian and vung.shape[0] > canh_trung_gian:
        vung = cv2.resize(vung, (canh_trung_gian, canh_trung_gian), interpolation=cv2.INTER_AREA)
    nd = cv2.INTER_AREA if vung.shape[0] > kich_thuoc else cv2.INTER_LINEAR
    return cv2.resize(vung, (kich_thuoc, kich_thuoc), interpolation=nd)
