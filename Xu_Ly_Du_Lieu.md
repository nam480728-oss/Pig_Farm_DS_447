# Xử lý dữ liệu

Tài liệu này gom toàn bộ việc cần làm để biến hai bộ dữ liệu đã tải về thành dữ liệu train được cho 3 mô hình: `pig_det.pt`, `pig_seg.pt`, `weight.json`. Cuối tài liệu có checklist để đánh dấu tiến độ.

## 1. Hiện trạng

| Bộ dữ liệu | Vị trí | Trạng thái |
|---|---|---|
| PigDetect | gốc: `data_goc/detection/PigDetect/` → đã xử lý: `data/data_phat_hien_pig/pig_det/` | Xong |
| PIGRGB-Weight | gốc: `data_goc/PIGRGB-Weight_full/PigRGB-Weight/` → đã xử lý: `data/data_cat_lung_tinh_kg/{pig_seg,RGB_9579}/` | Xong |

Việc 1 và việc 2 đã xong. `RGB_9579` đã chép sang `data/data_cat_lung_tinh_kg/RGB_9579/`, giữ nguyên 5 fold. Kiểm tra lại bằng `python data/kiem_tra_du_lieu.py`.

## 2. Đặc điểm hai bộ dữ liệu (đã kiểm tra)

**PigDetect** (`PigDetect.zip`, 1,23 GB)
- 2.931 ảnh từ 31 chuồng, 42.177 hộp lợn, một lớp `pig`.
- Trong zip có `dev/` (2.681 ảnh) và `test/` (250 ảnh). Mỗi thư mục có `images/` và `labels/`.
- Nhãn trong zip **đã là định dạng YOLO** (`0 x_tâm y_tâm rộng cao`, chuẩn hóa 0–1). Không cần đổi từ COCO.
- Zip không chia train và val. Việc chia nằm ở `coco_annotations/`: `train.json` (2.431 ảnh), `val.json` (250 ảnh), cả hai đều là ảnh của `dev/`. `test.json` (250 ảnh) khớp với `test/`.
- Ảnh có nhiều kích thước: 640×360, 733×498, 1280×800, 1920×1080, 2688×1520. Có ảnh ban đêm hồng ngoại và ảnh bị nhiễu đốm sáng.

**PIGRGB-Weight** (`PIGRGB-Weight_full.zip`, 9,75 GB)
- `RGB_9579/`: 9.578 ảnh (README ghi 9.579, lệch 1 ảnh, nhiều khả năng sai lệch trong bộ gốc). Chụp ở độ cao **1,88 m**, ảnh 960×540. Chia sẵn 5 fold theo con lợn:

  | Fold | Số con | Số ảnh |
  |---|---|---|
  | fold1 | 15 | 1.939 |
  | fold2 | 15 | 2.030 |
  | fold3 | 15 | 1.962 |
  | fold4 | 14 | 1.807 |
  | fold5 | 14 | 1.840 |

  Mỗi thư mục con (ví dụ `73.36_124`) là một con lợn: 73,36 kg và 124 ảnh. Có ảnh bị nhòe do lợn di chuyển.
- `RGB_MASK_3394/`: 3.394 ảnh `RGB_3394/` và 3.394 mặt nạ `MASK_3394/` cùng tên, cùng thư mục con. Chụp ở độ cao **1,78 m**, ảnh 960×540.
- Mặt nạ lưu ở dạng 3 kênh, giá trị chỉ 0 hoặc 255 (một ảnh mẫu có lợn chiếm khoảng 14% diện tích).
- Mỗi ảnh có **một con lợn chính** ở giữa khung, được vẽ mặt nạ. Ở rìa ảnh có thể thấy thêm lợn khác bị cắt một phần, các con đó **không có mặt nạ**.

## 3. Cấu trúc thư mục đích

Dữ liệu đã xử lý nằm trong `data/` (không đưa lên git):

```text
data/
  data_phat_hien_pig/
    pig_det/                  # khớp config/pig_det.yaml
      images/{train,val,test}/
      labels/{train,val,test}/
    chuan_bi_pigdetect.py
  data_cat_lung_tinh_kg/
    pig_seg/                  # khớp config/pig_seg.yaml
      images/{train,val}/
      labels/{train,val}/
    RGB_9579/fold1..fold5/    # ảnh cân nặng
    dac_trung.csv             # (sẽ tạo ở việc 3) 5 đặc trưng + kg + fold cho từng ảnh
    mat_na_sang_yolo.py
```

Dữ liệu gốc đã giải nén nằm trong `data_goc/` (cũng không đưa lên git).

## 4. Việc 1: Chuẩn bị PigDetect (train `pig_det.pt`)

Mục tiêu: dữ liệu YOLO có đủ `train`, `val`, `test`.

1. Giải nén `PigDetect.zip` vào `data_goc/detection/PigDetect/` (giữ cấu trúc `dev/` và `test/`).
2. Đọc `coco_annotations/train.json` và `val.json`, lấy tên file ảnh của từng tập (so theo tên file, bỏ phần thư mục).
3. Chép ảnh và nhãn của `dev/` về `data/data_phat_hien_pig/pig_det/{images,labels}/train` hoặc `val` theo tên trong json.
   (Thực tế `chuan_bi_pigdetect.py` chia ngẫu nhiên 250 ảnh val, seed 42, không theo json; tập test vẫn giống bản chính thức.)
4. Chép `test/images` và `test/labels` về `.../test` nguyên trạng.
5. Kiểm tra sau khi chia:
   - số ảnh: train 2.431, val 250, test 250;
   - mỗi ảnh có đúng một file nhãn cùng tên (ảnh không có lợn thì file nhãn trống hoặc không có, cần xử lý nhất quán);
   - tổng số hộp khớp 33.197 (train), 3.544 (val), 5.436 (test);
   - không có ảnh nào nằm đồng thời ở hai tập.
6. Có thể dùng `config/pig_det.yaml` ngay, vì nó đã trỏ đúng các thư mục trên.

Không cần resize: Ultralytics tự co ảnh về 960 px khi train.

Đã chốt: chép file thật để `data/data_phat_hien_pig/pig_det/` độc lập với `data_goc/`.

## 5. Việc 2: Mặt nạ lưng lợn (train `pig_seg.pt`)

Mục tiêu: dữ liệu YOLO-seg từ `RGB_MASK_3394`.

Vấn đề cần xử lý: mỗi ảnh chỉ có mặt nạ cho một con ở giữa, còn các con khác ở rìa không có mặt nạ. Nếu train trên toàn ảnh, mô hình sẽ học nhầm rằng các con ở rìa là nền. Ở bước suy luận, `pig_seg.pt` lại chạy trên **ảnh cắt từng con theo hộp phát hiện**. Vì vậy phải cắt ảnh khi chuẩn bị dữ liệu train.

Các bước:
1. Giải nén `RGB_MASK_3394` vào `data_goc/PIGRGB-Weight_full/PigRGB-Weight/`.
2. Với mỗi cặp ảnh và mặt nạ cùng tên:
   - đọc mặt nạ, đổi sang một kênh, ngưỡng `> 127` thành 0/1;
   - lấy hộp bao quanh của mặt nạ, nới thêm khoảng 10–15% mỗi cạnh (giống độ lệch của hộp phát hiện thật), rồi cắt cả ảnh và mặt nạ theo hộp đó;
   - lấy đường viền bằng `cv2.findContours` (chỉ lấy đường viền ngoài lớn nhất), đơn giản hóa nhẹ bằng `cv2.approxPolyDP`;
   - ghi thành đa giác YOLO-seg: `0 x1 y1 x2 y2 …` với tọa độ chuẩn hóa trong ảnh đã cắt.
3. Chia train và val **theo con lợn** (theo thư mục con `<kg>_<số ảnh>`), không chia theo ảnh. Gợi ý 80/20, cố định hạt giống ngẫu nhiên để lặp lại được. Ghi danh sách con của mỗi tập ra file để có thể kiểm tra lại.
4. Kiểm tra bằng mắt: vẽ chồng đa giác lên 20–30 ảnh ngẫu nhiên để chắc không bị lệch hay lật trục.

Ghi chú:
- Mặt nạ vẽ tay gồm cả tai, đuôi và chân. Việc bỏ các phần này làm ở bước tính đặc trưng bằng phép mở hình thái (mục 6), không làm ở bước này.
- `pig_seg.yaml` hiện không có `test`. Nếu cần báo cáo kết quả trên tập độc lập, tách thêm một nhóm con làm test.

## 6. Việc 3: Ước tính cân nặng (train `weight.json`)

Mục tiêu: bảng đặc trưng cho 9.578 ảnh `RGB_9579`, rồi train XGBoost theo 5 fold.

Việc này **phụ thuộc vào `pig_seg.pt`** vì `RGB_9579` không có mặt nạ vẽ tay. Phải train xong việc 2 trước.

1. Dùng `data/data_cat_lung_tinh_kg/RGB_9579/` (đã chép sẵn từ `data_goc/`).
2. Đọc kg từ tên thư mục con (`73.36_124` thành 73,36 kg) và số fold từ thư mục cha.
3. Với mỗi ảnh:
   - chạy `pig_seg.pt` để lấy mặt nạ con lợn chính (chọn mặt nạ có diện tích lớn nhất hoặc gần tâm ảnh nhất);
   - bỏ tai, đuôi, chân bằng phép mở hình thái (bài báo PIGRGB-Weight cho thấy làm vậy chính xác hơn);
   - tính 5 đặc trưng theo cm: diện tích, chu vi, dài thân, rộng thân, độ lệch tâm.
4. Đổi pixel sang cm bằng `cm_moi_pixel`. Hệ số này tính từ độ cao camera 1,88 m và thông số ống kính. **Cần đọc `README.pdf` trong repo GitHub của tác giả** để lấy thông số camera. Tôi chưa tải file này.
5. Lưu bảng `data/data_cat_lung_tinh_kg/dac_trung.csv` gồm: đường dẫn ảnh, con, fold, kg, 5 đặc trưng.
6. Train XGBoost theo kiểm tra chéo 5 fold (train 4 fold, kiểm tra 1 fold, lặp lại cho đủ 5). Báo cáo sai số kg trung bình và sai số theo từng fold. Mục tiêu theo đề cương: sai số ≤ 5 kg.
7. Sau khi chọn tham số, train lại trên cả 5 fold và lưu `models/weight.json`.

Lưu ý:
- Tập huấn luyện mặt nạ (1,78 m) và tập cân nặng (1,88 m) chụp ở hai độ cao khác nhau. Cần kiểm tra chất lượng mặt nạ trên một số ảnh `RGB_9579` trước khi chạy cả bộ.
- Một số ảnh bị nhòe khi lợn di chuyển. Có thể lọc ảnh theo độ nét (ví dụ phương sai Laplacian) hoặc loại các ảnh mà mặt nạ dự đoán có diện tích bất thường, rồi ghi rõ đã loại bao nhiêu ảnh.
- Mô hình chỉ đáng tin về xếp hạng nặng nhẹ và phần trăm tăng. Số kg tuyệt đối trên camera trại là ước tính, vì camera khác độ cao (xem mục "Không cân thật thì số nào đáng tin" trong `idea.md`).

## 7. Thứ tự thực hiện

```text
Việc 1 (PigDetect) ──► train pig_det.pt
Việc 2 (mặt nạ)    ──► train pig_seg.pt ──► Việc 3 (đặc trưng cân nặng) ──► train weight.json
```

Việc 1 và việc 2 độc lập, làm song song được. Việc 3 phải chờ `pig_seg.pt`.

## 8. Script

| Script | Việc | Trạng thái |
|---|---|---|
| `data/data_phat_hien_pig/chuan_bi_pigdetect.py` | Việc 1 | Xong (đọc `data_goc/`, ghi `pig_det/`) |
| `data/data_cat_lung_tinh_kg/mat_na_sang_yolo.py` | Việc 2 | Xong (đọc `data_goc/`, ghi `pig_seg/`) |
| `data/kiem_tra_du_lieu.py` | Kiểm tra số lượng, trùng tập, vẽ thử nhãn | Xong |
| `scripts/train_cat_lung.py`, `test_cat_lung.py` | Train và test `pig_seg.pt` | Xong code, chưa train (xem `scripts/train_cat_lung.md`) |
| `scripts/train_can_nang.py` | Việc 3 | Khung `TODO` |

Chạy lại 2 script chuyển đổi sẽ **ghi đè** dữ liệu train hiện tại.

## 9. Rủi ro và điểm cần lưu ý

| Vấn đề | Cách xử lý |
|---|---|
| Ảnh cùng một con rất giống nhau, chia theo ảnh sẽ làm kết quả đẹp giả | Luôn chia theo con (mặt nạ) hoặc theo fold có sẵn (cân nặng) |
| Mặt nạ chỉ có cho một con mỗi ảnh | Cắt ảnh theo hộp mặt nạ trước khi train `pig_seg` |
| Hai độ cao camera khác nhau (1,78 m và 1,88 m) | Kiểm tra mặt nạ trên `RGB_9579`; dùng hệ số `cm_moi_pixel` riêng cho từng nhóm |
| README ghi 9.579 ảnh, thực tế 9.578 | Ghi chú trong báo cáo, không ảnh hưởng kết quả |
| Dữ liệu khoảng 11 GB sau khi giải nén sẽ lớn hơn | Dữ liệu gốc để trong `data_goc/`, có thể xóa khi không cần chạy lại bước chuyển đổi; ổ D: còn trống nhiều |
| PIGRGB-Weight không ghi giấy phép cụ thể | Chỉ dùng cho nghiên cứu, ghi trích dẫn bài báo ([Agriculture 2025](https://www.mdpi.com/2077-0472/15/8/814)) |
| PigDetect là CC BY 4.0 | Ghi trích dẫn Henrich và cộng sự trong báo cáo |

## 10. Checklist

**Đã xong**
- [x] Tải PigDetect và `coco_annotations/`
- [x] Tải PIGRGB-Weight đầy đủ và kiểm tra zip
- [x] Kiểm tra cấu trúc, kích thước ảnh, định dạng nhãn và mặt nạ

**Việc 1: PigDetect**
- [x] Giải nén `PigDetect.zip`
- [x] Chia `dev/` thành train và val (chia ngẫu nhiên 2431/250, không theo json)
- [x] Chép `test/` sang `pig_det/`
- [x] Kiểm tra số ảnh, số hộp, không trùng tập
- [x] Vẽ thử hộp lên vài ảnh để kiểm tra bằng mắt

**Việc 2: Mặt nạ**
- [x] Giải nén `RGB_MASK_3394`
- [x] Cắt ảnh theo hộp mặt nạ và đổi sang đa giác YOLO-seg
- [x] Chia train và val theo con
- [x] Vẽ thử đa giác lên vài ảnh để kiểm tra bằng mắt

**Việc 3: Cân nặng**
- [ ] Đọc thông số camera để tính `cm_moi_pixel` (chưa tìm được tiêu cự; tạm train bằng đơn vị pixel)
- [ ] Train xong `pig_seg.pt` (`python scripts/train_cat_lung.py`)
- [ ] Tính 5 đặc trưng cho 9.578 ảnh `RGB_9579`
- [ ] Lưu `dac_trung.csv`
- [ ] Train XGBoost 5 fold và đo sai số kg
- [ ] Lưu `models/weight.json`
