import os
import cv2
import numpy as np
import random
from pathlib import Path

# Đường dẫn tính từ vị trí file này nên chạy ở đâu cũng được
THU_MUC_NAY = Path(__file__).resolve().parent                     # data/data_cat_lung_tinh_kg
GOC = THU_MUC_NAY.parents[1]                                      # D:\Pig_Farm
DU_LIEU_GOC = GOC / "data_goc" / "PIGRGB-Weight_full" / "PigRGB-Weight" / "RGB_MASK_3394"


def process_pig_seg(seed=42, train_ratio=0.8):
    random.seed(seed)

    rgb_root = DU_LIEU_GOC / "RGB_3394"
    mask_root = DU_LIEU_GOC / "MASK_3394"
    out_dir = THU_MUC_NAY / "pig_seg"         # ghi đè vào dữ liệu train hiện tại (khớp config/pig_seg.yaml)
    
    for split in ["train", "val"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    if not rgb_root.exists():
        print(f"Lỗi: Không tìm thấy thư mục {rgb_root}")
        return

    # Lấy danh sách các thư mục con (mỗi thư mục tương ứng 1 con lợn)
    pig_folders = [f for f in rgb_root.iterdir() if f.is_dir() and not f.name.startswith('.')]
    print(f"Tìm thấy {len(pig_folders)} thư mục con lợn trong RGB_3394.")
    
    if len(pig_folders) == 0:
        print("Lỗi: Không tìm thấy thư mục con nào.")
        return

    random.shuffle(pig_folders)

    split_idx = int(len(pig_folders) * train_ratio)
    train_folders = pig_folders[:split_idx]
    val_folders = pig_folders[split_idx:]

    splits = [("train", train_folders), ("val", val_folders)]

    for split_name, folders in splits:
        print(f"--- Đang xử lý pig_seg tập {split_name} ({len(folders)} con lợn) ---")
        processed_count = 0
        for folder in folders:
            folder_name = folder.name
            rgb_dir = rgb_root / folder_name
            mask_dir = mask_root / folder_name

            if not rgb_dir.exists() or not mask_dir.exists():
                continue

            # Hỗ trợ quét linh hoạt các định dạng ảnh khác nhau
            img_paths = []
            for ext in ["*.jpg", "*.png", "*.JPG", "*.PNG"]:
                img_paths.extend(list(rgb_dir.glob(ext)))

            for img_path in img_paths:
                # Tìm file mask tương ứng (khớp tên gốc bất kể đuôi mở rộng)
                matched_masks = list(mask_dir.glob(f"{img_path.stem}.*"))
                if not matched_masks:
                    continue
                mask_path = matched_masks[0]

                # Đọc ảnh và mặt nạ
                img = cv2.imread(str(img_path))
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

                if img is None or mask is None:
                    continue

                # Nhị phân hóa mặt nạ
                _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                
                # Tìm bounding box của mặt nạ
                coords = cv2.findNonZero(binary_mask)
                if coords is None:
                    continue

                x, y, w, h = cv2.boundingRect(coords)
                img_h, img_w = img.shape[:2]

                # Nới rộng bbox 12% mỗi cạnh
                pad_w = int(w * 0.12)
                pad_h = int(h * 0.12)
                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(img_w, x + w + pad_w)
                y2 = min(img_h, y + h + pad_h)

                # Cắt ảnh và mặt nạ
                cropped_img = img[y1:y2, x1:x2]
                cropped_mask = binary_mask[y1:y2, x1:x2]
                crop_h, crop_w = cropped_img.shape[:2]

                if crop_w == 0 or crop_h == 0:
                    continue

                # Trích xuất contour ngoài cùng lớn nhất
                contours, _ = cv2.findContours(cropped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue

                c = max(contours, key=cv2.contourArea)
                approx = cv2.approxPolyDP(c, epsilon=0.005 * cv2.arcLength(c, True), closed=True)

                if len(approx) < 3:
                    continue

                # Chuẩn hóa tọa độ polygon theo ảnh đã crop (0 - 1)
                normalized_points = []
                for point in approx:
                    px, py = point[0]
                    normalized_points.extend([round(px / crop_w, 6), round(py / crop_h, 6)])

                # Ghi dữ liệu ảnh và nhãn yolo-seg (chuyển đuôi ảnh đích thành .jpg để đồng bộ)
                save_name = f"{folder_name}_{img_path.stem}.jpg"
                cv2.imwrite(str(out_dir / "images" / split_name / save_name), cropped_img)

                label_path = out_dir / "labels" / split_name / f"{Path(save_name).stem}.txt"
                with open(label_path, "w") as f_lbl:
                    poly_str = " ".join(map(str, normalized_points))
                    f_lbl.write(f"0 {poly_str}\n")
                processed_count += 1
                
        print(f"Đã xử lý thành công {processed_count} mẫu cho tập {split_name}.")

if __name__ == "__main__":
    process_pig_seg()