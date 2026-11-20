"""Train mô hình TƯ THẾ lợn: ĐỨNG / KHÔNG ĐỨNG (YOLO11n-cls) -> models/Pig_Posture/pig_posture.pt

Dùng để LỌC khung hình trước khi cân: chỉ con "dung" mới được đưa vào pig_seg -> 10 đặc trưng -> kg.

Dữ liệu : data/data_posture_final_ok/{train,val,test}/{dung,khong_dung}/  (ảnh vuông 224, đã xử lý sẵn)
          Phần xử lý dữ liệu nằm hết trong thư mục dữ liệu (chia_du_lieu.py, tu_the_chung.py); file này CHỈ train.
          Lớp (theo thứ tự tên thư mục): 0 = dung, 1 = khong_dung.
Đầu ra  : models/Pig_Posture/
            pig_posture.pt      mô hình tốt nhất (copy từ train/weights/best.pt)
            train/              log của Ultralytics: results.csv, results.png, confusion_matrix*.png, args.yaml, weights/
            danh_gia/           đánh giá best.pt trên VAL và TEST:
                                  du_doan_val.csv, du_doan_test.csv   dự đoán từng ảnh (+ trại, số lợn bên cạnh)
                                  confusion_matrix_val.png / _test.png
                                  anh_sai_test.jpg                    các ảnh test đoán sai
            tom_tat.json        mọi chỉ số quan trọng (tổng, theo trại, theo mức độ đông lợn) để đưa vào báo cáo

Cách chạy:
    python scripts/train_tu_the.py                                   # train mặc định (data_posture_final_ok)
    python scripts/train_tu_the.py --data data/data_posture_final    # train bộ cũ (cắt sát + viền) để so sánh
    python scripts/train_tu_the.py --batch 16                        # nếu hết bộ nhớ GPU
    python scripts/train_tu_the.py --tiep-tuc                        # train bị ngắt -> chạy tiếp từ last.pt
    python scripts/train_tu_the.py --chi-danh-gia                    # không train, chỉ đánh giá pig_posture.pt
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Thư mục gốc dự án = cha của scripts/. Mọi đường dẫn tính từ đây nên chạy ở đâu cũng được.
GOC = Path(__file__).resolve().parents[1]
DATA_MAC_DINH = GOC / "data" / "data_posture_final_ok"
THU_MUC_RA = GOC / "models" / "Pig_Posture"
MODEL_RA = THU_MUC_RA / "pig_posture.pt"
LOP = ["dung", "khong_dung"]          # Ultralytics xếp lớp theo tên thư mục -> 0 dung, 1 khong_dung


def doc_tham_so():
    """Tham số dòng lệnh. Mặc định đã chọn cho GPU RTX 4060 Laptop 8 GB và bộ dữ liệu ~1.300 ảnh train."""
    ap = argparse.ArgumentParser(description="Train YOLO11n-cls tư thế đứng / không đứng")
    ap.add_argument("--data", default=str(DATA_MAC_DINH), help="thư mục có train/ val/ test/")
    ap.add_argument("--model", default="yolo11n-cls.pt", help="trọng số khởi đầu (pretrained ImageNet, tự tải)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=224, help="ảnh dữ liệu đã là 224 x 224")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--patience", type=int, default=15, help="dừng sớm nếu 15 epoch liền val không tốt lên")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="0", help="'0' = GPU thứ nhất, 'cpu' = CPU")
    ap.add_argument("--thu-muc-ra", default=str(THU_MUC_RA), help=argparse.SUPPRESS)  # chỉ dùng khi chạy thử
    ap.add_argument("--tiep-tuc", action="store_true", help="train tiếp từ train*/weights/last.pt mới nhất")
    ap.add_argument("--chi-danh-gia", action="store_true", help="bỏ qua train, chỉ đánh giá pig_posture.pt")
    return ap.parse_args()


def gon(p):
    """Đường dẫn tương đối so với gốc dự án nếu nằm trong dự án, ngược lại giữ nguyên."""
    p = Path(p).resolve()
    return str(p.relative_to(GOC)) if p.is_relative_to(GOC) else str(p)


def kiem_tra_moi_truong(device):
    """In và trả về thông tin GPU / thư viện (ghi vào tom_tat.json)."""
    import torch
    import ultralytics

    tt = {"python": sys.version.split()[0], "ultralytics": ultralytics.__version__, "torch": torch.__version__,
          "cuda": torch.cuda.is_available(),
          "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    print("Môi trường:", tt)
    if device != "cpu" and not tt["cuda"]:
        sys.exit("Không thấy GPU CUDA. Dùng --device cpu nếu vẫn muốn chạy bằng CPU.")
    return tt


def kiem_tra_du_lieu(data):
    """Đếm ảnh mỗi tập / lớp; dừng nếu thiếu thư mục."""
    dem = {}
    for tap in ["train", "val", "test"]:
        for lop in LOP:
            d = data / tap / lop
            if not d.is_dir():
                sys.exit(f"Thiếu thư mục {d}. Chạy chia_du_lieu.py trong thư mục dữ liệu trước.")
            dem[f"{tap}/{lop}"] = len(list(d.glob("*.jpg")))
    print("Dữ liệu:", dem)
    return dem


def bo_qua_loi_kiem_tra_amp():
    """Ultralytics tải yolo26n.pt để thử AMP; mạng lỗi thì cả lần train bị dừng.
    Bọc lại: lỗi tải -> coi như AMP ổn (RTX 4060 hỗ trợ), kết quả kiểm tra thật vẫn giữ nguyên."""
    import ultralytics.engine.trainer as trainer

    goc = trainer.check_amp

    def an_toan(model):
        try:
            return goc(model)
        except Exception as loi:  # noqa: BLE001
            print(f"[Cảnh báo] Không kiểm tra được AMP ({type(loi).__name__}), vẫn bật AMP.")
            return True

    trainer.check_amp = an_toan


def train(a, thu_muc_ra):
    """Train bằng Ultralytics, trả về thư mục kết quả (train, train2, ...)."""
    from ultralytics import YOLO

    bo_qua_loi_kiem_tra_amp()
    if a.tiep_tuc:
        ung_vien = sorted(thu_muc_ra.glob("train*/weights/last.pt"), key=lambda p: p.stat().st_mtime)
        if not ung_vien:
            sys.exit(f"Không có last.pt trong {thu_muc_ra}/train*/weights/")
        print("Train tiếp từ:", ung_vien[-1])
        model = YOLO(str(ung_vien[-1]))
        model.train(resume=True)
    else:
        model = YOLO(a.model)
        model.train(
            data=str(a.data),            # thư mục train/ val/ test/ -> Ultralytics tự hiểu là bài phân loại
            epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, patience=a.patience,
            workers=a.workers, seed=a.seed, deterministic=True, device=a.device,
            project=str(thu_muc_ra), name="train", exist_ok=False, plots=True, val=True,
            # ---- tăng cường dữ liệu (riêng cho bài này) ----
            fliplr=0.5,       # lật ngang: đầu lợn quay trái / phải như nhau
            flipud=0.5,       # lật dọc: camera nhìn từ trên xuống, không có "trên / dưới" cố định
            scale=0.1,        # cắt ngẫu nhiên giữ 90-100 % ảnh -> không cắt mất chân (mặc định 50-100 %)
            erasing=0.1,      # xóa ngẫu nhiên 1 vùng nhỏ, ít hơn mặc định 0.4 để không xóa mất chân
            hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,   # đổi màu / độ sáng: ảnh ngày, đêm, nhiều loại camera
        )
    return Path(model.trainer.save_dir)


# --------------------------------------------------------------------------- đánh giá
def so_lon_ben_canh(data):
    """Số lợn khác lọt vào ảnh vuông của mỗi ảnh PigDetect (để đánh giá theo mức độ đông).

    Gọi lại hàm đọc hộp nhãn của phần xử lý dữ liệu (cat_anh_tu_the.hop_pigdetect) -> không viết lại.
    RGB_9579 luôn 1 con -> 0. Trả về dict {file: số con}.
    """
    sys.path.insert(0, str(data))
    try:
        from cat_anh_tu_the import PD, hop_pigdetect
    except ImportError:
        return {}
    split = pd.read_csv(data / "split.csv")
    dsl = pd.concat([pd.read_csv(GOC / "data" / "data_posture" / f)
                     for f in ["danh_sach.csv", "danh_sach_ung_vien_dung.csv"]]).drop_duplicates("file").set_index("file")
    kq = {}
    for r in split.itertuples():
        if r.trai == "rgb9579" or r.file not in dsl.index:
            kq[r.file] = 0
            continue
        s = dsl.loc[r.file]
        f = PD / s.tap_goc / "images" / s.anh_goc
        anh = cv2.imread(str(f))
        if anh is None:
            continue
        H, W = anh.shape[:2]
        hop = hop_pigdetect(f, W, H)
        x1, y1, x2, y2 = hop[int(s.hop)]
        c = max(x2 - x1, y2 - y1) * 1.2                     # cạnh ảnh vuông (nới 10 % mỗi bên)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        a, b, cc, dd = cx - c / 2, cy - c / 2, cx + c / 2, cy + c / 2
        kq[r.file] = sum(1 for j, (p, q, u, v) in enumerate(hop) if j != int(s.hop)
                         and max(0, min(cc, u) - max(a, p)) * max(0, min(dd, v) - max(b, q)) > 0.05 * (u - p) * (v - q))
    return kq


def chi_so(y, p):
    """Accuracy + precision / recall / F1 cho từng lớp (y, p: mảng nhãn 'dung' / 'khong_dung')."""
    y, p = np.asarray(y), np.asarray(p)
    kq = {"so_anh": int(len(y)), "accuracy": round(float((y == p).mean()), 4) if len(y) else None}
    for lop in LOP:
        tp = int(((p == lop) & (y == lop)).sum())
        pr = tp / max(1, int((p == lop).sum()))
        rc = tp / max(1, int((y == lop).sum()))
        kq[lop] = {"precision": round(pr, 4), "recall": round(rc, 4),
                   "f1": round(2 * pr * rc / max(1e-9, pr + rc), 4), "so_anh": int((y == lop).sum())}
    return kq


def ve_confusion(y, p, tieu_de, duong_dan):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = np.array([[int(((y == t) & (p == d)).sum()) for d in LOP] for t in LOP])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(m, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{m[i, j]}\n({m[i, j] / max(1, m[i].sum()) * 100:.1f}%)", ha="center", va="center",
                    color="white" if m[i, j] > m.max() / 2 else "black")
    ax.set_xticks([0, 1], ["dung", "khong_dung"])
    ax.set_yticks([0, 1], ["dung", "khong_dung"])
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thật")
    ax.set_title(tieu_de)
    fig.tight_layout()
    fig.savefig(duong_dan, dpi=130)
    plt.close(fig)


def luoi_anh_sai(bang, data, duong_dan, toi_da=40):
    sai = bang[bang.that != bang.du_doan].head(toi_da)
    if sai.empty:
        return 0
    o = []
    for r in sai.itertuples():
        anh = cv2.imdecode(np.fromfile(str(data / r.tap / r.that / r.file), np.uint8), cv2.IMREAD_COLOR)
        anh = cv2.resize(anh, (224, 224))
        for mau, day in (((0, 0, 0), 4), ((0, 255, 255), 1)):
            cv2.putText(anh, f"that {r.that[:5]} -> {r.du_doan[:5]} {r.conf:.2f}", (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, mau, day)
            cv2.putText(anh, r.trai, (4, 214), cv2.FONT_HERSHEY_SIMPLEX, 0.45, mau, day)
        o.append(anh)
    while len(o) % 8:
        o.append(np.zeros_like(o[0]))
    cv2.imwrite(str(duong_dan), np.vstack([np.hstack(o[i:i + 8]) for i in range(0, len(o), 8)]))
    return len(sai)


def danh_gia(duong_dan_model, data, thu_muc_ra, them=None):
    """Đánh giá trên VAL và TEST: tổng, theo trại, theo mức độ đông lợn; lưu csv, biểu đồ, tom_tat.json."""
    from ultralytics import YOLO

    model = YOLO(str(duong_dan_model))
    ten = model.names                                      # {0: 'dung', 1: 'khong_dung'}
    thu_muc_dg = thu_muc_ra / "danh_gia"
    thu_muc_dg.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(data / "split.csv").set_index("file") if (data / "split.csv").exists() else None
    so_ben = so_lon_ben_canh(data)

    tom_tat = dict(them or {})
    tom_tat.update({"thoi_gian": time.strftime("%Y-%m-%d %H:%M:%S"), "model": gon(duong_dan_model),
                    "du_lieu": gon(data), "lop": ten})
    for tap in ["val", "test"]:
        dong = []
        for lop in LOP:
            files = sorted((data / tap / lop).glob("*.jpg"))
            for i in range(0, len(files), 64):
                for f, r in zip(files[i:i + 64], model.predict([str(x) for x in files[i:i + 64]],
                                                              imgsz=224, verbose=False)):
                    k = int(r.probs.top1)
                    dong.append({"file": f.name, "tap": tap, "that": lop, "du_doan": ten[k],
                                 "conf": round(float(r.probs.top1conf), 4),
                                 "xac_suat_dung": round(float(r.probs.data[0]), 4)})
        b = pd.DataFrame(dong)
        b["trai"] = [split.loc[f, "trai"] if split is not None and f in split.index else "?" for f in b.file]
        b["so_lon_ben"] = [so_ben.get(f, -1) for f in b.file]
        b.to_csv(thu_muc_dg / f"du_doan_{tap}.csv", index=False)
        ve_confusion(b.that.values, b.du_doan.values, f"{tap.upper()} ({len(b)} ảnh)",
                     thu_muc_dg / f"confusion_matrix_{tap}.png")

        kq = {"tong": chi_so(b.that, b.du_doan),
              "theo_trai": {t: chi_so(g.that, g.du_doan) for t, g in b.groupby("trai")},
              "chi_pigdetect": chi_so(b[b.trai != "rgb9579"].that, b[b.trai != "rgb9579"].du_doan)}
        pdt = b[(b.trai != "rgb9579") & (b.so_lon_ben >= 0)]
        if len(pdt):
            nhom = pd.cut(pdt.so_lon_ben, [-1, 1, 2, 100], labels=["0-1 con", "2 con", ">=3 con"])
            kq["pigdetect_theo_so_lon_ben_canh"] = {str(n): chi_so(g.that, g.du_doan)
                                                     for n, g in pdt.groupby(nhom, observed=True)}
        if tap == "test":
            kq["so_anh_sai"] = luoi_anh_sai(b.assign(tap=tap), data, thu_muc_dg / "anh_sai_test.jpg")
        tom_tat[tap] = kq
        t = kq["tong"]
        print(f"[{tap.upper()}] acc {t['accuracy']:.4f} | dung P {t['dung']['precision']:.3f} R {t['dung']['recall']:.3f}"
              f" | khong_dung P {t['khong_dung']['precision']:.3f} R {t['khong_dung']['recall']:.3f}")

    (thu_muc_ra / "tom_tat.json").write_text(json.dumps(tom_tat, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Đã lưu:", thu_muc_ra / "tom_tat.json")


def main():
    a = doc_tham_so()
    a.data = Path(a.data).resolve()
    thu_muc_ra = Path(a.thu_muc_ra).resolve()
    thu_muc_ra.mkdir(parents=True, exist_ok=True)
    model_ra = thu_muc_ra / MODEL_RA.name

    if a.chi_danh_gia:
        if not model_ra.exists():
            sys.exit(f"Chưa có {model_ra}. Hãy train trước.")
        danh_gia(model_ra, a.data, thu_muc_ra)
        return

    moi_truong = kiem_tra_moi_truong(a.device)
    dem = kiem_tra_du_lieu(a.data)
    bat_dau = time.time()
    thu_muc_train = train(a, thu_muc_ra)
    phut = round((time.time() - bat_dau) / 60, 1)
    best = thu_muc_train / "weights" / "best.pt"
    if not best.exists():
        sys.exit(f"Không thấy {best}")
    shutil.copy2(best, model_ra)
    print(f"Train xong sau {phut} phút -> {model_ra}")

    them = {"thoi_gian_train_phut": phut, "moi_truong": moi_truong, "so_anh": dem,
            "thu_muc_train": gon(thu_muc_train),
            "tham_so": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}}
    danh_gia(model_ra, a.data, thu_muc_ra, them)


if __name__ == "__main__":
    main()
