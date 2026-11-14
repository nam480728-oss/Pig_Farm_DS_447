import os
import random
import shutil
from pathlib import Path

def setup_pigdetect_auto_split(seed=42):
    random.seed(seed)
    # Đường dẫn tính từ vị trí file này nên chạy ở đâu cũng được
    thu_muc_nay = Path(__file__).resolve().parent                    # data/data_phat_hien_pig
    base_dir = thu_muc_nay.parents[1] / "data_goc" / "detection" / "PigDetect"   # PigDetect.zip đã giải nén
    dev_dir = base_dir / "dev"
    test_dir = base_dir / "test"

    out_dir = thu_muc_nay / "pig_det"         # ghi đè vào dữ liệu train hiện tại (khớp config/pig_det.yaml)
    
    # Tạo các thư mục đích
    for split in ["train", "val", "test"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    print("--- Đang xử lý tự động chia Train/Val từ thư mục dev ---")
    dev_img_dir = dev_dir / "images"
    dev_lbl_dir = dev_dir / "labels"
    
    if not dev_img_dir.exists():
        print(f"Lỗi: Không tìm thấy thư mục {dev_img_dir}")
        return

    # Lấy danh sách tất cả các file ảnh trong dev/images
    all_dev_images = list(dev_img_dir.glob("*.*"))
    total_dev = len(all_dev_images)
    print(f"Tổng số ảnh tìm thấy trong dev: {total_dev}")

    # Xáo trộn và chia theo đúng số lượng tiêu chuẩn: 250 ảnh val, phần còn lại (2431 ảnh) cho train
    random.shuffle(all_dev_images)
    val_count = 250
    val_images = all_dev_images[:val_count]
    train_images = all_dev_images[val_count:]

    splits = [("train", train_images), ("val", val_images)]

    for split_name, img_list in splits:
        for img_path in img_list:
            stem = img_path.stem
            # Sao chép ảnh
            shutil.copy2(img_path, out_dir / "images" / split_name / img_path.name)
            
            # Sao chép nhãn tương ứng (nếu không có file nhãn thì tạo file txt rỗng)
            lbl_src = dev_lbl_dir / f"{stem}.txt"
            lbl_dst = out_dir / "labels" / split_name / f"{stem}.txt"
            if lbl_src.exists():
                shutil.copy2(lbl_src, lbl_dst)
            else:
            # Tạo file nhãn rỗng cho ảnh không có lợn
                lbl_dst.touch()
        print(f"Đã sao chép thành công {len(img_list)} ảnh vào tập {split_name}.")

    print("\n--- Đang xử lý tập Test ---")
    test_img_dir = test_dir / "images"
    test_lbl_dir = test_dir / "labels"
    
    if test_img_dir.exists():
        copied_test = 0
        for img_path in test_img_dir.glob("*.*"):
            stem = img_path.stem
            shutil.copy2(img_path, out_dir / "images" / "test" / img_path.name)
            
            lbl_src = test_lbl_dir / f"{stem}.txt"
            lbl_dst = out_dir / "labels" / "test" / f"{stem}.txt"
            if lbl_src.exists():
                shutil.copy2(lbl_src, lbl_dst)
            else:
                lbl_dst.touch()
            copied_test += 1
        print(f"Tập test: Đã sao chép {copied_test} ảnh.")
    else:
        print("Lỗi: Không tìm thấy thư mục test/images")

if __name__ == "__main__":
    setup_pigdetect_auto_split()