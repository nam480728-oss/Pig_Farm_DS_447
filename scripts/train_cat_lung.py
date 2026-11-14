"""Train mô hình CẮT LƯNG LỢN (YOLO11s-seg) -> models/Pig_Seg/pig_seg.pt

Dữ liệu : config/pig_seg.yaml  (data/data_cat_lung_tinh_kg/pig_seg, train 2769 ảnh / val 625 ảnh)
Đầu ra  : models/Pig_Seg/
            pig_seg.pt          mô hình tốt nhất (copy từ train/weights/best.pt)
            train/              toàn bộ log của Ultralytics: results.csv, results.png, confusion_matrix*.png,
                                các đường cong P/R/F1/PR (Box + Mask), ảnh batch train/val, args.yaml, weights/
            danh_gia/           đánh giá lại best.pt trên tập val: biểu đồ của Ultralytics + IoU/Dice từng ảnh
            tom_tat.json        tất cả chỉ số quan trọng gom vào 1 file (để đưa vào báo cáo)

Cách chạy (xem thêm scripts/train_cat_lung.md):
    python scripts/train_cat_lung.py                    # train mặc định 80 epoch
    python scripts/train_cat_lung.py --batch 8          # nếu báo hết bộ nhớ GPU (CUDA out of memory)
    python scripts/train_cat_lung.py --tiep-tuc         # train bị ngắt giữa chừng -> chạy tiếp từ last.pt
    python scripts/train_cat_lung.py --chi-danh-gia     # không train, chỉ đánh giá lại pig_seg.pt đã có
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

# Thư mục gốc dự án = cha của thư mục scripts/. Mọi đường dẫn đều tính từ đây nên chạy ở đâu cũng được.
GOC = Path(__file__).resolve().parents[1]
DATA_YAML = GOC / "config" / "pig_seg.yaml"
THU_MUC_RA = GOC / "models" / "Pig_Seg"
MODEL_RA = THU_MUC_RA / "pig_seg.pt"

DUOI_ANH = {".jpg", ".jpeg", ".png"}


def doc_tham_so():
    """Các tham số dòng lệnh. Giá trị mặc định đã chọn cho GPU RTX 4060 Laptop 8 GB."""
    ap = argparse.ArgumentParser(description="Train YOLO11s-seg cắt lưng lợn")
    ap.add_argument("--model", default="yolo11s-seg.pt", help="trọng số khởi đầu (Ultralytics tự tải nếu chưa có)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640, help="ảnh được co về cạnh dài 640 px")
    ap.add_argument("--batch", type=int, default=16, help="hết bộ nhớ GPU thì giảm xuống 8")
    ap.add_argument("--patience", type=int, default=20, help="dừng sớm nếu 20 epoch liền không tốt lên")
    ap.add_argument("--workers", type=int, default=4, help="số tiến trình đọc ảnh song song")
    ap.add_argument("--seed", type=int, default=42, help="cố định ngẫu nhiên để train lại ra kết quả giống")
    ap.add_argument("--device", default="0", help="'0' = GPU thứ nhất, 'cpu' = chạy CPU (rất chậm)")
    ap.add_argument("--tiep-tuc", action="store_true", help="train tiếp từ train*/weights/last.pt mới nhất")
    ap.add_argument("--chi-danh-gia", action="store_true", help="bỏ qua train, chỉ đánh giá models/Pig_Seg/pig_seg.pt")
    return ap.parse_args()


def kiem_tra_moi_truong(device):
    """In thông tin GPU / thư viện, trả về dict để ghi vào tom_tat.json (chứng minh môi trường train)."""
    import torch
    import ultralytics

    tt = {
        "python": sys.version.split()[0],
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    print("Môi trường:", tt)
    if device != "cpu" and not tt["cuda"]:
        sys.exit("Không thấy GPU CUDA. Dùng --device cpu nếu vẫn muốn chạy bằng CPU.")
    return tt


def tim_last_pt():
    """Tìm last.pt của lần train gần nhất (train, train2, train3...) để train tiếp."""
    ung_vien = sorted(THU_MUC_RA.glob("train*/weights/last.pt"), key=lambda p: p.stat().st_mtime)
    if not ung_vien:
        sys.exit(f"Không có last.pt nào trong {THU_MUC_RA}/train*/weights/ để train tiếp.")
    return ung_vien[-1]


def bo_qua_loi_kiem_tra_amp():
    """AMP = train bằng số thực 16 bit -> nhanh hơn, tốn ít bộ nhớ GPU hơn.

    Trước khi train, Ultralytics tải thêm yolo26n.pt chỉ để thử xem GPU chạy AMP có đúng không. Nếu mạng/GitHub
    lỗi thì file tải về hỏng và cả quá trình train bị dừng. Hàm này bọc lại: lỗi tải file -> coi như AMP ổn
    (RTX 4060 hỗ trợ AMP), còn kết quả kiểm tra thật (đúng/sai) vẫn giữ nguyên như Ultralytics.
    """
    import ultralytics.engine.trainer as trainer

    kiem_tra_goc = trainer.check_amp

    def kiem_tra_an_toan(model):
        try:
            return kiem_tra_goc(model)
        except Exception as loi:  # noqa: BLE001
            print(f"[Cảnh báo] Không kiểm tra được AMP ({type(loi).__name__}), vẫn bật AMP.")
            return True

    trainer.check_amp = kiem_tra_an_toan


def train(a):
    """Train bằng Ultralytics. Trả về thư mục kết quả (models/Pig_Seg/train hoặc train2, ...)."""
    from ultralytics import YOLO

    bo_qua_loi_kiem_tra_amp()

    if a.tiep_tuc:
        last = tim_last_pt()
        print(f"Train tiếp từ: {last}")
        # resume=True: đọc lại toàn bộ tham số cũ (epochs, batch, thư mục lưu...) từ trong last.pt
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        model = YOLO(a.model)
        model.train(
            data=str(DATA_YAML),
            epochs=a.epochs,
            imgsz=a.imgsz,
            batch=a.batch,
            patience=a.patience,
            workers=a.workers,
            seed=a.seed,
            deterministic=True,
            device=a.device,
            # project phải là đường dẫn TUYỆT ĐỐI, nếu để tương đối Ultralytics sẽ tự nhét vào runs/segment/...
            project=str(THU_MUC_RA),
            name="train",
            exist_ok=False,       # đã có train/ thì tạo train2/, không ghi đè kết quả cũ
            plots=True,           # vẽ results.png, confusion matrix, đường cong P/R/F1/PR, ảnh batch
            val=True,             # đánh giá trên tập val sau mỗi epoch -> cột metrics/* trong results.csv
        )
    return Path(model.trainer.save_dir)


# --------------------------------------------------------------------------- đánh giá IoU / Dice từng ảnh
def mask_that(nhan_txt, w, h):
    """Đọc polygon YOLO-seg (tọa độ 0..1) rồi tô thành mask nhị phân kích thước w x h."""
    so = [float(t) for t in nhan_txt.read_text().split()[1:]]
    diem = np.array(so, dtype=np.float32).reshape(-1, 2) * [w, h]   # đổi tỉ lệ về pixel
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [diem.round().astype(np.int32)], 1)   # tô vùng bên trong polygon = 1 (lợn), ngoài = 0 (nền)
    return m


def iou_dice(a, b):
    """IoU = giao/hợp ; Dice = 2*giao/(|a|+|b|). Cả hai = 1 là trùng khít hoàn toàn."""
    giao = int(np.logical_and(a, b).sum())
    hop = int(np.logical_or(a, b).sum())
    tong = int(a.sum()) + int(b.sum())
    return (giao / hop if hop else 1.0), (2 * giao / tong if tong else 1.0)


def danh_gia_iou(model, thu_muc_danh_gia):
    """Với mỗi ảnh val: lấy mask có độ tin cậy cao nhất, so với mask thật -> IoU, Dice, sai số diện tích.

    Đây là con số dễ hiểu nhất để chứng minh mô hình cắt đúng lưng lợn, và sai số diện tích ảnh hưởng
    trực tiếp tới bước tính cân nặng phía sau.
    """
    cfg = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    goc_data = (GOC / cfg["path"]) if not Path(cfg["path"]).is_absolute() else Path(cfg["path"])
    thu_muc_anh = goc_data / cfg["val"]
    thu_muc_nhan = goc_data / cfg["val"].replace("images", "labels")
    ds = sorted(f for f in thu_muc_anh.iterdir() if f.suffix.lower() in DUOI_ANH)

    dong, xau = [], []
    for i, f in enumerate(ds, 1):
        anh = cv2.imread(str(f))
        h, w = anh.shape[:2]
        that = mask_that(thu_muc_nhan / f"{f.stem}.txt", w, h)
        # retina_masks=True: trả mask đúng kích thước ảnh gốc (không phải 640x640) để so pixel với pixel
        kq = model.predict(anh, imgsz=640, retina_masks=True, verbose=False)[0]
        if kq.masks is None:                     # không tìm thấy con lợn nào -> tính là sai hoàn toàn
            du_doan, conf = np.zeros_like(that), 0.0
        else:
            k = int(kq.boxes.conf.argmax())      # chọn con có độ tin cậy cao nhất (ảnh chỉ có 1 con)
            du_doan = (kq.masks.data[k].cpu().numpy() > 0.5).astype(np.uint8)
            conf = float(kq.boxes.conf[k])
        iou, dice = iou_dice(that, du_doan)
        # int(...) bắt buộc: sum() của mảng uint8 là số không dấu, trừ ra âm sẽ bị tràn thành số khổng lồ
        dt_du_doan, dt_that = int(du_doan.sum()), int(that.sum())
        sai_dt = (dt_du_doan - dt_that) / max(dt_that, 1) * 100   # % diện tích dư (+) / thiếu (-)
        dong.append({"anh": f.name, "conf": conf, "iou": iou, "dice": dice, "sai_so_dien_tich_%": sai_dt})
        xau.append((iou, f, that, du_doan))
        if i % 100 == 0:
            print(f"  IoU: {i}/{len(ds)} ảnh")

    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")                         # vẽ ra file, không cần cửa sổ
    import matplotlib.pyplot as plt

    bang = pd.DataFrame(dong)
    bang.to_csv(thu_muc_danh_gia / "iou_tung_anh.csv", index=False, encoding="utf-8-sig")

    # Biểu đồ phân bố IoU và sai số diện tích trên toàn bộ tập val
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].hist(bang["iou"], bins=40, color="#3b7dd8")
    ax[0].axvline(bang["iou"].mean(), color="red", ls="--", label=f"trung bình {bang['iou'].mean():.3f}")
    ax[0].set(title="Phân bố mask IoU trên tập val", xlabel="IoU", ylabel="số ảnh")
    ax[0].legend()
    ax[1].hist(bang["sai_so_dien_tich_%"], bins=40, color="#e08a2c")
    ax[1].axvline(0, color="black", lw=1)
    ax[1].set(title="Sai số diện tích mask (%)", xlabel="(dự đoán - thật) / thật × 100", ylabel="số ảnh")
    fig.tight_layout()
    fig.savefig(thu_muc_danh_gia / "phan_bo_iou.png", dpi=120)
    plt.close(fig)

    # 8 ảnh tệ nhất: xanh lá = mask thật, đỏ = mask dự đoán -> để xem mô hình hay sai ở đâu
    xau.sort(key=lambda t: t[0])
    o = []
    for iou, f, that, du_doan in xau[:8]:
        anh = cv2.imread(str(f))
        for m, mau in ((that, (0, 255, 0)), (du_doan, (0, 0, 255))):
            vien, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(anh, vien, -1, mau, 3)
        anh = cv2.resize(anh, (400, 300))
        cv2.putText(anh, f"IoU {iou:.3f}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        o.append(anh)
    while len(o) % 4:
        o.append(np.zeros_like(o[0]))
    luoi = np.vstack([np.hstack(o[i:i + 4]) for i in range(0, len(o), 4)])
    cv2.imwrite(str(thu_muc_danh_gia / "anh_te_nhat.jpg"), luoi)

    return {
        "so_anh": len(bang),
        "iou_trung_binh": float(bang["iou"].mean()),
        "iou_trung_vi": float(bang["iou"].median()),
        "dice_trung_binh": float(bang["dice"].mean()),
        "ti_le_anh_iou_tren_0.9": float((bang["iou"] > 0.9).mean()),
        "so_anh_khong_tim_thay_lon": int((bang["conf"] == 0).sum()),
        "sai_so_dien_tich_tuyet_doi_tb_%": float(bang["sai_so_dien_tich_%"].abs().mean()),
    }


def duong_dan_gon(p):
    """Đường dẫn tính từ thư mục dự án (models/Pig_Seg/...) cho gọn; nằm ngoài dự án thì giữ nguyên."""
    try:
        return Path(p).resolve().relative_to(GOC).as_posix()
    except ValueError:
        return str(p)


def danh_gia(duong_dan_model,tham_so_train=None, moi_truong=None, thu_muc_train=None):
    """Đánh giá best.pt trên tập val, lưu biểu đồ + tom_tat.json."""
    from ultralytics import YOLO

    model = YOLO(str(duong_dan_model))
    print("\n=== ĐÁNH GIÁ TRÊN TẬP VAL ===")
    # model.val vẽ lại confusion matrix, đường cong PR/P/R/F1 cho cả Box và Mask vào models/Pig_Seg/danh_gia/
    m = model.val(data=str(DATA_YAML), split="val", imgsz=640, batch=16, plots=True,
                  project=str(THU_MUC_RA), name="danh_gia", exist_ok=True)
    thu_muc_danh_gia = Path(m.save_dir)

    print("Tính IoU / Dice từng ảnh...")
    iou = danh_gia_iou(model, thu_muc_danh_gia)

    tom_tat = {
        "thoi_gian": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": duong_dan_gon(duong_dan_model),
        "thu_muc_train": duong_dan_gon(thu_muc_train) if thu_muc_train else None,
        "tham_so_train": tham_so_train,
        "moi_truong": moi_truong,
        # mAP50 / mAP50-95 / Precision / Recall của khung (B) và của mask (M) do Ultralytics tính
        "ultralytics_val": {k: float(v) for k, v in m.results_dict.items()},
        "toc_do_ms_moi_anh": {k: float(v) for k, v in m.speed.items()},
        "iou_dice_tung_anh": iou,
    }
    (THU_MUC_RA / "tom_tat.json").write_text(json.dumps(tom_tat, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== KẾT QUẢ ===")
    print(f"Mask mAP50      : {m.seg.map50:.4f}")
    print(f"Mask mAP50-95   : {m.seg.map:.4f}")
    print(f"Mask Precision  : {m.seg.mp:.4f} | Recall: {m.seg.mr:.4f}")
    print(f"IoU trung bình  : {iou['iou_trung_binh']:.4f} | Dice: {iou['dice_trung_binh']:.4f}")
    print(f"Sai số diện tích: {iou['sai_so_dien_tich_tuyet_doi_tb_%']:.2f} %")
    print(f"Đã lưu tóm tắt  : {THU_MUC_RA / 'tom_tat.json'}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")       # console Windows mặc định không in được tiếng Việt
    a = doc_tham_so()
    os.chdir(GOC)                                   # để 'path' tương đối trong pig_seg.yaml trỏ đúng thư mục data
    THU_MUC_RA.mkdir(parents=True, exist_ok=True)
    moi_truong = kiem_tra_moi_truong(a.device)

    if a.chi_danh_gia:
        if not MODEL_RA.exists():
            sys.exit(f"Chưa có {MODEL_RA}. Hãy train trước.")
        # Lấy lại tham số + thời gian train từ lần train gần nhất để tom_tat.json vẫn đầy đủ
        thu_muc_train, tham_so = None, None
        ds_train = sorted(THU_MUC_RA.glob("train*/args.yaml"), key=lambda p: p.stat().st_mtime)
        if ds_train:
            thu_muc_train = ds_train[-1].parent
            tham_so = yaml.safe_load(ds_train[-1].read_text(encoding="utf-8"))
            csv = thu_muc_train / "results.csv"
            if csv.exists():                        # cột 'time' của dòng cuối = tổng số giây đã train
                import pandas as pd
                tham_so["thoi_gian_train_phut"] = round(float(pd.read_csv(csv)["time"].iloc[-1]) / 60, 1)
        danh_gia(MODEL_RA, tham_so, moi_truong, thu_muc_train)
        return

    bat_dau = time.time()
    thu_muc_train = train(a)
    phut = (time.time() - bat_dau) / 60

    best = thu_muc_train / "weights" / "best.pt"
    if not best.exists():
        sys.exit(f"Không thấy {best}, train có thể đã lỗi.")
    shutil.copy2(best, MODEL_RA)                    # best.pt = epoch có điểm val (mAP) cao nhất
    print(f"\nĐã chép {best} -> {MODEL_RA}  (train mất {phut:.1f} phút)")

    tham_so = yaml.safe_load((thu_muc_train / "args.yaml").read_text(encoding="utf-8"))
    tham_so["thoi_gian_train_phut"] = round(phut, 1)
    danh_gia(MODEL_RA, tham_so, moi_truong, thu_muc_train)


# Bắt buộc trên Windows: các tiến trình đọc ảnh (workers) import lại file này, không có dòng này sẽ train lặp vô hạn
if __name__ == "__main__":
    main()
