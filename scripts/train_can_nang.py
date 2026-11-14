"""Train mô hình ƯỚC TÍNH CÂN NẶNG (XGBoost) từ 5 đặc trưng lưng lợn. Chạy 2 LẦN:

LẦN 1 - TÌM THAM SỐ (khoảng 3-5 phút, CPU):
    python scripts/train_can_nang.py --tim-tham-so
    Thử 36 bộ tham số (max_depth × learning_rate × min_child_weight). Mỗi bộ chạy 5-fold CV theo con lợn:
        lượt i: TEST = fold i (các con chưa từng thấy) | trong 4 fold còn lại: 1 fold làm VAL (dừng sớm chọn số cây),
                3 fold làm TRAIN -> sau đó học lại trên đủ 4 fold với số cây đã chọn -> dự đoán fold test.
    Xếp hạng 36 bộ theo MAE trung bình theo con -> models/Pig_Kg/tim_tham_so.csv
    Với bộ hạng 1: lưu đầy đủ kết quả CV (chỉ số, dự đoán từng ảnh/con, train-val-test từng lượt, biểu đồ)
    -> BẰNG CHỨNG đánh giá (mỗi con lợn được chấm đúng 1 lần khi chưa thấy).
    In thêm gợi ý theo QUY TẮC 1 SAI SỐ CHUẨN: bộ đơn giản nhất trong các bộ tốt ngang nhau.

LẦN 2 - TRAIN MÔ HÌNH CUỐI (vài giây):
    python scripts/train_can_nang.py              # dùng bộ hạng 1 trong tim_tham_so.csv
    python scripts/train_can_nang.py --hang 4     # dùng bộ hạng 4 (vd bộ do quy tắc 1 sai số chuẩn gợi ý)
    Đọc tham số từ tim_tham_so.csv -> học trên CẢ 5 fold (không chia val/test, vì đánh giá đã làm bằng CV)
    -> models/Pig_Kg/weight.json + thong_tin_mo_hinh.json + biểu đồ độ quan trọng đặc trưng.
    Nếu bộ được chọn KHÁC bộ đang có đánh giá chi tiết -> tự chạy lại 5-fold CV cho bộ đó (khoảng 5 giây)
    để tom_tat.json, du_doan_*.csv, ket_qua_tung_fold.csv và biểu đồ luôn khớp với weight.json.

Đầu vào: data/data_cat_lung_tinh_kg/dac_trung.csv (tạo bởi scripts/lay_dac_trung.py). Hướng dẫn: scripts/train_can_nang.md
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parents[1]
CSV_VAO = GOC / "data" / "data_cat_lung_tinh_kg" / "dac_trung.csv"
THU_MUC_RA = GOC / "models" / "Pig_Kg"
MODEL_RA = THU_MUC_RA / "weight.json"
CSV_THAM_SO = THU_MUC_RA / "tim_tham_so.csv"

DAC_TRUNG = ["dien_tich", "chu_vi", "dai", "rong", "lech_tam"]   # THỨ TỰ cột đưa vào mô hình - không được đổi
NHOM_KG = [70, 90, 110, 130, 150, 170, 195]                        # biên các nhóm kg cho confusion matrix

# Lưới tìm tham số: 4 × 3 × 3 = 36 bộ.
#   learning_rate nhỏ (≤ 0,1) + nhiều cây: Friedman (2001); cây nông: Hastie et al., ESL chương 10;
#   chỉ có 5 đặc trưng, quan hệ với kg gần tuyến tính -> không cần cây sâu.
LUOI = {
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.03, 0.05, 0.1],
    "min_child_weight": [1, 5, 10],
}
# Cố định, không tìm
SUBSAMPLE = 0.8      # mỗi cây chỉ học 80 % ảnh ngẫu nhiên -> chống học vẹt
SO_CAY_TOI_DA = 3000 # số cây tối đa; số thật do dừng sớm quyết định
DUNG_SOM = 100       # dừng khi 100 cây liền không giảm MAE trên VAL
SEED = 42


def doc_tham_so():
    ap = argparse.ArgumentParser(description="Train XGBoost ước tính cân nặng lợn (2 lần: tìm tham số, train cuối)")
    ap.add_argument("--tim-tham-so", action="store_true", help="LẦN 1: thử 36 bộ tham số bằng 5-fold CV")
    ap.add_argument("--hang", type=int, default=1, help="LẦN 2: dùng bộ tham số hạng mấy trong tim_tham_so.csv")
    # Bộ lọc dữ liệu: chỉ có tác dụng ở LẦN 1. LẦN 2 tự đọc lại bộ lọc từ tim_tham_so.csv để dùng ĐÚNG cùng dữ liệu.
    ap.add_argument("--giu-cham-bien", action="store_true", help="giữ ảnh có mask chạm mép (lợn có thể bị cắt cụt)")
    ap.add_argument("--loc-nhoe", type=float, default=0, help="loại ảnh có do_net < giá trị này; 0 = không lọc")
    ap.add_argument("--dac-trung-tho", action="store_true", help="dùng đặc trưng mask thô (chưa bỏ tai/chân/đuôi)")
    ap.add_argument("--nguong-kg", type=float, default=5, help="ngưỡng sai số chấp nhận (kg) để tính tỉ lệ đạt")
    return ap.parse_args()


# =============================================================================== dữ liệu
def doc_du_lieu(giu_cham_bien, loc_nhoe, dac_trung_tho):
    """Đọc CSV, loại ảnh xấu, ghi lại đã loại bao nhiêu ảnh và vì sao (để báo cáo minh bạch)."""
    if not CSV_VAO.exists():
        sys.exit(f"Không thấy {CSV_VAO}. Chạy scripts/lay_dac_trung.py trước.")
    d = pd.read_csv(CSV_VAO)
    cot = [f"{c}_tho" for c in DAC_TRUNG] if dac_trung_tho else DAC_TRUNG
    loai = {"tong_anh_ban_dau": len(d)}

    bo = d["khong_thay_lon"].astype(bool) | d[cot].isna().any(axis=1)
    loai["khong_thay_lon_hoac_thieu_dac_trung"] = int(bo.sum())
    d = d[~bo]
    if not giu_cham_bien:
        bo = d["cham_bien"].astype(bool)
        loai["mask_cham_mep_anh"] = int(bo.sum())
        d = d[~bo]
    if loc_nhoe > 0:
        bo = d["do_net"] < loc_nhoe
        loai[f"anh_nhoe_do_net_duoi_{loc_nhoe:g}"] = int(bo.sum())
        d = d[~bo]
    loai["con_lai"] = len(d)
    loai["so_con"] = int(d["con"].nunique())

    if dac_trung_tho:   # đổi tên cột *_tho -> tên chuẩn để phần sau dùng chung DAC_TRUNG
        d = d.drop(columns=DAC_TRUNG).rename(columns={f"{c}_tho": c for c in DAC_TRUNG})
    return d.reset_index(drop=True), loai


# =============================================================================== chỉ số
def chi_so(that, du_doan, nguong):
    """Các chỉ số hồi quy chuẩn cho 1 tập dự đoán."""
    from sklearn.metrics import r2_score
    sai = np.asarray(du_doan) - np.asarray(that)
    return {
        "MAE_kg": float(np.mean(np.abs(sai))),                       # sai trung bình bao nhiêu kg (chỉ số chính)
        "RMSE_kg": float(np.sqrt(np.mean(sai ** 2))),                # = căn của loss bình phương; phạt nặng lần sai lớn
        "MAPE_%": float(np.mean(np.abs(sai) / np.asarray(that)) * 100),
        "R2": float(r2_score(that, du_doan)),                        # 1 = hoàn hảo, 0 = đoán bừa bằng trung bình
        "lech_TB_kg": float(np.mean(sai)),                           # + : hay đoán dư, - : hay đoán thiếu
        "sai_lon_nhat_kg": float(np.max(np.abs(sai))),
        f"ti_le_sai_duoi_{nguong:g}kg_%": float(np.mean(np.abs(sai) <= nguong) * 100),   # "accuracy" của hồi quy
        "so_mau": int(len(sai)),
    }


def gop_theo_con(bang, cot_du_doan):
    """Mỗi con có nhiều ảnh -> lấy TRUNG VỊ dự đoán làm cân nặng của con đó (giống thực tế: đo nhiều khung hình)."""
    return bang.groupby("con").agg(fold=("fold", "first"), kg=("kg", "first"),
                                   du_doan=(cot_du_doan, "median"), so_anh=("kg", "size")).reset_index()


# =============================================================================== mô hình
def tao_model(tham_so, n_cay, dung_som=None):
    """Tạo XGBRegressor với 3 tham số được tìm + các tham số cố định."""
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=n_cay,
        max_depth=int(tham_so["max_depth"]),
        learning_rate=float(tham_so["learning_rate"]),
        min_child_weight=float(tham_so["min_child_weight"]),
        subsample=SUBSAMPLE,
        objective="reg:squarederror",        # loss = bình phương sai số
        eval_metric="mae",                   # theo dõi MAE (kg) trên tập val để dừng sớm
        early_stopping_rounds=dung_som,
        random_state=SEED,
        n_jobs=-1,
    )


def cross_validation(d, tham_so, nguong, ghi_log=False):
    """5-fold CV theo 5 fold có sẵn của bộ dữ liệu (mỗi fold khoảng 15 con, không con nào nằm ở 2 fold).

    Mỗi lượt:  TEST = 1 fold | VAL = 1 fold (xoay vòng) | TRAIN = 3 fold còn lại
        B1. học trên TRAIN, theo dõi MAE trên VAL -> dừng sớm -> số cây tốt nhất
        B2. học lại trên TRAIN + VAL (đủ 4 fold) với đúng số cây đó -> dự đoán TEST
    Trả về (bảng có cột du_doan = dự đoán out-of-fold, danh sách thông tin từng lượt).
    """
    from sklearn.linear_model import LinearRegression

    folds = sorted(d["fold"].unique())
    d = d.copy()
    d["du_doan"] = np.nan          # XGBoost
    d["du_doan_co_so"] = np.nan    # mô hình cơ sở: kg = a × diện tích + b (để chứng minh XGBoost có ích)
    luot = []
    for i, f_test in enumerate(folds):
        f_con_lai = [f for f in folds if f != f_test]
        f_val = f_con_lai[i % len(f_con_lai)]
        test = d["fold"] == f_test
        val = d["fold"] == f_val
        train = ~test & ~val
        X, y = d[DAC_TRUNG], d["kg"]

        # B1: dừng sớm trên VAL
        m = tao_model(tham_so, SO_CAY_TOI_DA, DUNG_SOM)
        m.fit(X[train], y[train], eval_set=[(X[train], y[train]), (X[val], y[val])], verbose=False)
        so_cay = m.best_iteration + 1
        duong_cong = m.evals_result()                       # MAE theo số cây trên TRAIN và VAL -> đường cong học
        mae_train = duong_cong["validation_0"]["mae"][so_cay - 1]
        mae_val = duong_cong["validation_1"]["mae"][so_cay - 1]

        # B2: học lại trên đủ 4 fold, dự đoán TEST
        m = tao_model(tham_so, so_cay)
        m.fit(X[~test], y[~test], verbose=False)
        d.loc[test, "du_doan"] = m.predict(X[test])
        mae_train_4fold = float(np.mean(np.abs(m.predict(X[~test]) - y[~test])))   # sai số trên dữ liệu ĐÃ HỌC

        lr = LinearRegression().fit(d.loc[~test, ["dien_tich"]], y[~test])
        d.loc[test, "du_doan_co_so"] = lr.predict(d.loc[test, ["dien_tich"]])

        ct = chi_so(y[test], d.loc[test, "du_doan"], nguong)
        con = gop_theo_con(d[test], "du_doan")
        ctc = chi_so(con["kg"], con["du_doan"], nguong)
        luot.append({"fold_test": f_test, "fold_val": f_val, "so_cay": so_cay,
                     "MAE_train_3fold": float(mae_train), "MAE_val": float(mae_val),
                     "MAE_train_4fold": mae_train_4fold, "MAE_test": ct["MAE_kg"],
                     "theo_anh": ct, "theo_con": ctc, "duong_cong": duong_cong})
        if ghi_log:
            print(f"  test {f_test} | val {f_val} | {so_cay:4d} cây | MAE train {mae_train_4fold:.2f} "
                  f"| val {mae_val:.2f} | test {ct['MAE_kg']:.2f} kg (ảnh) / {ctc['MAE_kg']:.2f} kg (con)")
    return d, luot


# =============================================================================== biểu đồ
def plt_agg():
    import matplotlib
    matplotlib.use("Agg")                       # vẽ ra file, không mở cửa sổ
    import matplotlib.pyplot as plt
    return plt


def ve_bieu_do_cv(d, con, luot, nguong, thu_muc):
    """Biểu đồ đánh giá từ 5-fold CV của bộ tham số tốt nhất."""
    plt = plt_agg()
    from sklearn.metrics import confusion_matrix

    thu_muc.mkdir(parents=True, exist_ok=True)
    sai = d["du_doan"] - d["kg"]
    lo, hi = d["kg"].min() - 5, d["kg"].max() + 5

    # 1. Dự đoán vs thật: điểm nằm trên đường chéo = đoán đúng
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.scatter(d["kg"], d["du_doan"], s=4, alpha=0.15, color="#7aa6d9", label="từng ảnh")
    ax.scatter(con["kg"], con["du_doan"], s=28, color="#c0392b", label="từng con (trung vị)")
    ax.plot([lo, hi], [lo, hi], "k-", lw=1, label="đoán đúng")
    ax.fill_between([lo, hi], [lo - nguong, hi - nguong], [lo + nguong, hi + nguong],
                    color="green", alpha=0.08, label=f"±{nguong:g} kg")
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Cân nặng thật (kg)", ylabel="Cân nặng dự đoán (kg)",
           title="Dự đoán vs thật (5-fold CV: mỗi điểm dự đoán khi CHƯA thấy con đó)")
    ax.legend()
    fig.tight_layout(); fig.savefig(thu_muc / "du_doan_vs_thuc_te.png", dpi=130); plt.close(fig)

    # 2. Bland-Altman: đường giữa = lệch hệ thống, 2 đường ngoài = 95 % ảnh nằm trong
    tb, lech, sd = (d["kg"] + d["du_doan"]) / 2, sai.mean(), sai.std()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(tb, sai, s=4, alpha=0.2)
    for y, nhan in ((lech, f"lệch TB {lech:+.2f}"), (lech + 1.96 * sd, f"+1.96SD {lech + 1.96 * sd:+.2f}"),
                    (lech - 1.96 * sd, f"-1.96SD {lech - 1.96 * sd:+.2f}")):
        ax.axhline(y, color="red", ls="--" if "SD" in nhan else "-", lw=1)
        ax.text(tb.max(), y, nhan, va="bottom", ha="right", color="red")
    ax.set(xlabel="(thật + dự đoán) / 2 (kg)", ylabel="dự đoán − thật (kg)", title="Bland-Altman (theo ảnh)")
    fig.tight_layout(); fig.savefig(thu_muc / "bland_altman.png", dpi=130); plt.close(fig)

    # 3. Sai số theo cân nặng (từng con) + phân bố sai số (từng ảnh)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].scatter(con["kg"], con["du_doan"] - con["kg"], c="#c0392b", s=25)
    ax[0].axhline(0, color="k", lw=1)
    for s in (nguong, -nguong):
        ax[0].axhline(s, color="green", ls="--", lw=1)
    ax[0].set(xlabel="Cân nặng thật (kg)", ylabel="dự đoán − thật (kg)", title="Sai số từng con theo cân nặng")
    ax[1].hist(sai, bins=60, color="#3b7dd8")
    ax[1].axvline(0, color="k", lw=1)
    ax[1].set(xlabel="dự đoán − thật (kg)", ylabel="số ảnh", title="Phân bố sai số (từng ảnh)")
    fig.tight_layout(); fig.savefig(thu_muc / "sai_so.png", dpi=130); plt.close(fig)

    # 4. TRAIN - VAL - TEST từng lượt (+ cơ sở): kiểm tra overfit và độ ổn định giữa các fold
    f = [v["fold_test"] for v in luot]
    x = np.arange(len(f))
    co_so = [np.mean(np.abs(d.loc[d.fold == k, "du_doan_co_so"] - d.loc[d.fold == k, "kg"])) for k in f]
    cot = [("MAE train (4 fold, đã học)", [v["MAE_train_4fold"] for v in luot], "#9ecae1"),
           ("MAE val (dừng sớm)", [v["MAE_val"] for v in luot], "#4292c6"),
           ("MAE test - ảnh", [v["MAE_test"] for v in luot], "#08519c"),
           ("MAE test - con", [v["theo_con"]["MAE_kg"] for v in luot], "#c0392b"),
           ("Cơ sở (kg ~ diện tích) test - ảnh", co_so, "#bbbbbb")]
    w = 0.16
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for j, (nhan, gt, mau) in enumerate(cot):
        ax.bar(x + (j - 2) * w, gt, w, label=nhan, color=mau)
    cao = max(max(gt) for _, gt, _ in cot)
    ax.set(xticks=x, xticklabels=[f"test {k}" for k in f], ylabel="MAE (kg)", ylim=(0, cao * 1.35),
           title="Sai số train / val / test từng lượt CV (test ≫ train là dấu hiệu overfit)")
    ax.legend(fontsize=8, ncol=3, loc="upper center"); fig.tight_layout(); fig.savefig(thu_muc / "train_val_test_tung_fold.png", dpi=130)
    plt.close(fig)

    # 5. Đường cong học: MAE (loss) trên TRAIN và VAL theo số cây
    fig, axs = plt.subplots(1, len(luot), figsize=(4 * len(luot), 3.6), sharey=True)
    for ax, v in zip(np.atleast_1d(axs), luot):
        ax.plot(v["duong_cong"]["validation_0"]["mae"], label="train (3 fold)")
        ax.plot(v["duong_cong"]["validation_1"]["mae"], label="val (1 fold)")
        ax.axvline(v["so_cay"] - 1, color="k", ls=":", lw=1)
        ax.set(title=f"test {v['fold_test']} | val {v['fold_val']} | {v['so_cay']} cây", xlabel="số cây", ylim=(0, 20))
    np.atleast_1d(axs)[0].set_ylabel("MAE (kg)")
    np.atleast_1d(axs)[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(thu_muc / "duong_cong_hoc.png", dpi=130); plt.close(fig)

    # 6. Confusion matrix theo NHÓM cân nặng (hồi quy không có lớp -> chia nhóm 20 kg)
    nhan = [f"{NHOM_KG[i]}-{NHOM_KG[i + 1]}" for i in range(len(NHOM_KG) - 1)]
    that = pd.cut(d["kg"], NHOM_KG, labels=nhan, right=False)
    doan = pd.cut(d["du_doan"].clip(NHOM_KG[0], NHOM_KG[-1] - 0.01), NHOM_KG, labels=nhan, right=False)
    cm = confusion_matrix(that, doan, labels=nhan)
    cm_tl = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(cm_tl, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(nhan)):
        for j in range(len(nhan)):
            ax.text(j, i, f"{cm[i, j]}\n{cm_tl[i, j] * 100:.0f}%", ha="center", va="center", fontsize=8,
                    color="white" if cm_tl[i, j] > 0.5 else "black")
    ax.set(xticks=range(len(nhan)), yticks=range(len(nhan)), xticklabels=nhan, yticklabels=nhan,
           xlabel="Nhóm dự đoán (kg)", ylabel="Nhóm thật (kg)",
           title=f"Confusion matrix theo nhóm cân nặng (từng ảnh)\nđúng nhóm: {np.trace(cm) / cm.sum() * 100:.1f} %")
    fig.tight_layout(); fig.savefig(thu_muc / "confusion_matrix_nhom_kg.png", dpi=130); plt.close(fig)

    # 7. Tương quan giữa các đặc trưng và kg
    tq = d[DAC_TRUNG + ["kg"]].corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(tq, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(len(tq)):
        for j in range(len(tq)):
            ax.text(j, i, f"{tq.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set(xticks=range(len(tq)), yticks=range(len(tq)), xticklabels=tq.columns, yticklabels=tq.columns,
           title="Hệ số tương quan Pearson")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(thu_muc / "tuong_quan_dac_trung.png", dpi=130); plt.close(fig)


def ve_bieu_do_tim_tham_so(bang, hang_chon, nguong_1se, thu_muc):
    """MAE theo con của các bộ, xếp hạng; tô đỏ bộ đang được dùng; vạch ngưỡng quy tắc 1 sai số chuẩn."""
    plt = plt_agg()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    nhan = [f"#{r.hang} d{r.max_depth} lr{r.learning_rate:g} m{r.min_child_weight:g}" for r in bang.itertuples()]
    mau = ["#c0392b" if h == hang_chon else "#3b7dd8" for h in bang["hang"]]
    ax.bar(range(len(bang)), bang["MAE_con_TB"], yerr=bang["MAE_con_SD"], color=mau, capsize=2)
    ax.axhline(nguong_1se, color="k", ls="--", lw=1, label=f"ngưỡng 1 SE = {nguong_1se:.3f} kg")
    ax.set(xticks=range(len(bang)), ylabel="MAE theo con (kg), TB ± SD 5 fold",
           title=f"{len(bang)} bộ tham số xếp hạng theo MAE 5-fold CV (đỏ = bộ đang dùng, hạng {hang_chon})")
    ax.set_xticklabels(nhan, rotation=90, fontsize=7)
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(thu_muc / "tim_tham_so.png", dpi=130); plt.close(fig)


# =============================================================================== chọn bộ + lưu chi tiết CV
def quy_tac_1se(bang):
    """Quy tắc 1 sai số chuẩn (Hastie, Tibshirani, Friedman - The Elements of Statistical Learning):
    các bộ có MAE ≤ MAE tốt nhất + 1 sai số chuẩn được coi là tốt NGANG NHAU (không phân biệt được bằng thống kê)
    -> chọn bộ ĐƠN GIẢN NHẤT (cây nông nhất), nếu nhiều bộ cùng độ sâu thì lấy bộ MAE thấp nhất.
    Sai số chuẩn = SD của MAE 5 fold / √5. Trả về (hạng gợi ý, ngưỡng)."""
    tot = bang.loc[bang["hang"] == 1].iloc[0]
    nguong = float(tot["MAE_con_TB"] + tot["MAE_con_SD"] / np.sqrt(5))
    trong = bang[bang["MAE_con_TB"] <= nguong].sort_values(["max_depth", "MAE_con_TB"])
    return int(trong.iloc[0]["hang"]), nguong


def luu_chi_tiet_cv(d_oof, luot, bang, hang, loai, bo_loc, nguong_kg, them=None):
    """Lưu TOÀN BỘ bằng chứng đánh giá 5-fold CV của 1 bộ tham số (bộ đang dùng làm weight.json):
    du_doan_tung_anh.csv, du_doan_tung_con.csv, ket_qua_tung_fold.csv, bieu_do/*.png, tom_tat.json.
    Trả về (chỉ số XGBoost, chỉ số mô hình cơ sở)."""
    r = bang.loc[bang["hang"] == hang].iloc[0]
    hang_1se, nguong_1se = quy_tac_1se(bang)
    con = gop_theo_con(d_oof, "du_doan")
    con_co_so = gop_theo_con(d_oof, "du_doan_co_so")

    d_oof[["anh", "fold", "con", "kg", "du_doan", "du_doan_co_so", "conf", "do_net"] + DAC_TRUNG].assign(
        sai_kg=d_oof["du_doan"] - d_oof["kg"]).to_csv(THU_MUC_RA / "du_doan_tung_anh.csv", index=False,
                                                      encoding="utf-8-sig")
    con.assign(sai_kg=con["du_doan"] - con["kg"]).to_csv(THU_MUC_RA / "du_doan_tung_con.csv", index=False,
                                                         encoding="utf-8-sig")
    pd.DataFrame([{k: v[k] for k in ("fold_test", "fold_val", "so_cay", "MAE_train_3fold", "MAE_val",
                                     "MAE_train_4fold", "MAE_test")}
                  | {"chenh_test_tru_train": v["MAE_test"] - v["MAE_train_4fold"]}
                  | {f"anh_{k}": x for k, x in v["theo_anh"].items()}
                  | {f"con_{k}": x for k, x in v["theo_con"].items()} for v in luot]).to_csv(
        THU_MUC_RA / "ket_qua_tung_fold.csv", index=False, encoding="utf-8-sig")

    print("Vẽ biểu đồ...")
    ve_bieu_do_cv(d_oof, con, luot, nguong_kg, THU_MUC_RA / "bieu_do")
    ve_bieu_do_tim_tham_so(bang, hang, nguong_1se, THU_MUC_RA / "bieu_do")

    cv = {"theo_anh": chi_so(d_oof["kg"], d_oof["du_doan"], nguong_kg),
          "theo_con": chi_so(con["kg"], con["du_doan"], nguong_kg)}
    co_so = {"theo_anh": chi_so(d_oof["kg"], d_oof["du_doan_co_so"], nguong_kg),
             "theo_con": chi_so(con_co_so["kg"], con_co_so["du_doan"], nguong_kg)}
    mae1 = float(bang.loc[bang["hang"] == 1, "MAE_con_TB"].iloc[0])
    tom_tat = {
        "thoi_gian": time.strftime("%Y-%m-%d %H:%M:%S"),
        **(them or {}),
        "du_lieu": loai,
        "bo_loc": bo_loc,
        "luoi_tham_so": LUOI, "co_dinh": {"subsample": SUBSAMPLE, "so_cay_toi_da": SO_CAY_TOI_DA,
                                          "dung_som": DUNG_SOM, "seed": SEED},
        # tom_tat.json + các file/biểu đồ CV luôn mô tả bộ ĐANG DÙNG (bo_duoc_danh_gia), không nhất thiết là hạng 1
        "bo_duoc_danh_gia": {"hang": int(hang), **{k: float(r[k]) for k in LUOI}, "so_cay_TB": int(r["so_cay_TB"]),
                             "MAE_con_TB": float(r["MAE_con_TB"]), "MAE_con_SD": float(r["MAE_con_SD"])},
        "quy_tac_1_sai_so_chuan": {"nguong_kg": nguong_1se, "hang_goi_y": hang_1se,
                                   "so_bo_trong_nguong": int((bang["MAE_con_TB"] <= nguong_1se).sum())},
        "chenh_bo_dang_dung_so_voi_hang1_MAE_con_kg": float(r["MAE_con_TB"] - mae1),
        "chenh_hang1_hang_cuoi_MAE_con_kg": float(bang["MAE_con_TB"].max() - mae1),
        "cv": cv,
        "cv_co_so_kg_theo_dien_tich": co_so,
        "tung_luot": [{k: v[k] for k in ("fold_test", "fold_val", "so_cay", "MAE_train_3fold", "MAE_val",
                                         "MAE_train_4fold", "MAE_test", "theo_anh", "theo_con")} for v in luot],
    }
    (THU_MUC_RA / "tom_tat.json").write_text(json.dumps(tom_tat, ensure_ascii=False, indent=2), encoding="utf-8")
    return cv, co_so, luot


def in_ket_qua(ts, so_cay, hang, cv, co_so, luot, nguong_kg):
    n = f"ti_le_sai_duoi_{nguong_kg:g}kg_%"
    print(f"\n=== HẠNG {hang}: depth {int(ts['max_depth'])}, lr {ts['learning_rate']:g}, "
          f"mcw {ts['min_child_weight']:g}, {so_cay} cây ===")
    print(f"{'5-fold CV':22s}{'MAE kg':>8s}{'RMSE kg':>9s}{'MAPE %':>8s}{'R²':>7s}{'≤' + format(nguong_kg, 'g') + 'kg %':>9s}")
    for ten, c in (("XGBoost - từng ảnh", cv["theo_anh"]), ("XGBoost - từng con", cv["theo_con"]),
                   ("Cơ sở   - từng ảnh", co_so["theo_anh"]), ("Cơ sở   - từng con", co_so["theo_con"])):
        print(f"{ten:22s}{c['MAE_kg']:8.2f}{c['RMSE_kg']:9.2f}{c['MAPE_%']:8.2f}{c['R2']:7.3f}{c[n]:9.1f}")
    print("Train / val / test từng lượt (MAE kg, theo ảnh):")
    for v in luot:
        print(f"  test {v['fold_test']}: train {v['MAE_train_4fold']:.2f} | val {v['MAE_val']:.2f} | test {v['MAE_test']:.2f}")


# =============================================================================== LẦN 1
def lan_1_tim_tham_so(a):
    d, loai = doc_du_lieu(a.giu_cham_bien, a.loc_nhoe, a.dac_trung_tho)
    print(f"Dữ liệu: {loai}")
    cac_bo = [dict(zip(LUOI, v)) for v in itertools.product(*LUOI.values())]
    print(f"\n=== TÌM THAM SỐ: {len(cac_bo)} bộ × 5 fold ===")

    dong, ket_qua, bat_dau = [], {}, time.time()
    for k, ts in enumerate(cac_bo, 1):
        d_oof, luot = cross_validation(d, ts, a.nguong_kg)
        con = gop_theo_con(d_oof, "du_doan")
        mae_con_fold = [v["theo_con"]["MAE_kg"] for v in luot]
        tong_anh = chi_so(d_oof["kg"], d_oof["du_doan"], a.nguong_kg)
        tong_con = chi_so(con["kg"], con["du_doan"], a.nguong_kg)
        dong.append({**ts,
                     "MAE_con_TB": float(np.mean(mae_con_fold)), "MAE_con_SD": float(np.std(mae_con_fold)),
                     "MAE_anh_TB": float(np.mean([v["theo_anh"]["MAE_kg"] for v in luot])),
                     "MAE_train_TB": float(np.mean([v["MAE_train_4fold"] for v in luot])),
                     "RMSE_anh": tong_anh["RMSE_kg"], "R2_anh": tong_anh["R2"], "R2_con": tong_con["R2"],
                     "so_cay_TB": int(round(np.mean([v["so_cay"] for v in luot]))),
                     **{f"MAE_con_{v['fold_test']}": v["theo_con"]["MAE_kg"] for v in luot}})
        ket_qua[k - 1] = (d_oof, luot)              # giữ lại để lưu chi tiết cho bộ tốt nhất
        da = time.time() - bat_dau
        print(f"  [{k:2d}/{len(cac_bo)}] depth {ts['max_depth']} lr {ts['learning_rate']:<5g} mcw {ts['min_child_weight']:<3g}"
              f"| MAE con {dong[-1]['MAE_con_TB']:.3f} ± {dong[-1]['MAE_con_SD']:.3f} | ảnh {dong[-1]['MAE_anh_TB']:.3f}"
              f" | {dong[-1]['so_cay_TB']:4d} cây | còn ~{da / k * (len(cac_bo) - k) / 60:.0f} phút")

    # ---- xếp hạng: MAE theo con (trung bình 5 fold) nhỏ nhất; hòa thì xét MAE theo ảnh
    bang = pd.DataFrame(dong)
    bang["_vt"] = range(len(bang))
    bang = bang.sort_values(["MAE_con_TB", "MAE_anh_TB"]).reset_index(drop=True)
    bang.insert(0, "hang", range(1, len(bang) + 1))
    # ghi bộ lọc dữ liệu vào từng dòng -> LẦN 2 đọc lại để train trên ĐÚNG cùng dữ liệu
    bang["giu_cham_bien"], bang["loc_nhoe"], bang["dac_trung_tho"] = a.giu_cham_bien, a.loc_nhoe, a.dac_trung_tho
    d_oof, luot = ket_qua[int(bang.loc[0, "_vt"])]
    bang = bang.drop(columns="_vt")
    bang.to_csv(CSV_THAM_SO, index=False, encoding="utf-8-sig")
    tot = bang.iloc[0]

    # ---- lưu đầy đủ kết quả CV của bộ hạng 1 (nếu LẦN 2 chọn hạng khác, LẦN 2 sẽ tự lưu lại cho hạng đó)
    print()
    bo_loc = {"giu_cham_bien": a.giu_cham_bien, "loc_nhoe": a.loc_nhoe, "dac_trung_tho": a.dac_trung_tho}
    cv, co_so, luot = luu_chi_tiet_cv(d_oof, luot, bang, 1, loai, bo_loc, a.nguong_kg,
                                      {"thoi_gian_tim_phut": round((time.time() - bat_dau) / 60, 1)})
    in_ket_qua({k: tot[k] for k in LUOI}, int(tot["so_cay_TB"]), 1, cv, co_so, luot, a.nguong_kg)

    hang_1se, nguong_1se = quy_tac_1se(bang)
    r1se = bang.loc[bang["hang"] == hang_1se].iloc[0]
    print(f"\nQuy tắc 1 sai số chuẩn: ngưỡng {nguong_1se:.3f} kg, {int((bang['MAE_con_TB'] <= nguong_1se).sum())} bộ trong ngưỡng"
          f" -> bộ đơn giản nhất: HẠNG {hang_1se} (depth {int(r1se['max_depth'])}, lr {r1se['learning_rate']:g}, "
          f"mcw {r1se['min_child_weight']:g}, MAE con {r1se['MAE_con_TB']:.3f} kg)")
    print(f"\nĐã lưu: {CSV_THAM_SO}\nTiếp theo (LẦN 2): python scripts/train_can_nang.py --hang {hang_1se}"
          f"   (hoặc --hang 1 nếu muốn bộ có MAE thấp nhất)")


# =============================================================================== LẦN 2
def lan_2_train_cuoi(a):
    if not CSV_THAM_SO.exists():
        sys.exit(f"Chưa có {CSV_THAM_SO}.\nChạy LẦN 1 trước: python scripts/train_can_nang.py --tim-tham-so")
    bang = pd.read_csv(CSV_THAM_SO)
    if not 1 <= a.hang <= len(bang):
        sys.exit(f"--hang phải từ 1 đến {len(bang)}")
    r = bang.loc[bang["hang"] == a.hang].iloc[0]
    ts = {k: r[k] for k in LUOI}
    so_cay = int(r["so_cay_TB"])

    # Bộ lọc lấy từ tim_tham_so.csv (không lấy từ dòng lệnh) -> chắc chắn cùng dữ liệu với lúc tìm tham số
    giu, nhoe, tho = bool(r["giu_cham_bien"]), float(r["loc_nhoe"]), bool(r["dac_trung_tho"])
    d, loai = doc_du_lieu(giu, nhoe, tho)
    print(f"Dữ liệu: {loai}")
    print(f"Tham số hạng {a.hang}: depth {int(ts['max_depth'])}, lr {ts['learning_rate']:g}, "
          f"mcw {ts['min_child_weight']:g}, {so_cay} cây (TB 5 lượt) | MAE CV theo con {r['MAE_con_TB']:.3f} kg")

    # Các file đánh giá chi tiết (tom_tat.json, du_doan_*.csv, ket_qua_tung_fold.csv, biểu đồ) phải mô tả ĐÚNG bộ
    # đang dùng. LẦN 1 chỉ lưu chi tiết cho hạng 1 -> nếu chọn hạng khác thì chạy lại 5-fold CV cho bộ này
    # (cùng quy trình, cùng seed -> số liệu khớp dòng của nó trong tim_tham_so.csv, mất khoảng 5 giây).
    tt_cu = THU_MUC_RA / "tom_tat.json"
    cu = json.loads(tt_cu.read_text(encoding="utf-8")) if tt_cu.exists() else {}
    if cu.get("bo_duoc_danh_gia", {}).get("hang") != a.hang:
        print(f"\nĐánh giá chi tiết hiện có không phải của hạng {a.hang} -> chạy lại 5-fold CV cho bộ này...")
        d_oof, luot = cross_validation(d, ts, a.nguong_kg)
        bo_loc = {"giu_cham_bien": giu, "loc_nhoe": nhoe, "dac_trung_tho": tho}
        them = {k: cu[k] for k in ("thoi_gian_tim_phut",) if k in cu}
        cv, co_so, luot = luu_chi_tiet_cv(d_oof, luot, bang, a.hang, loai, bo_loc, a.nguong_kg, them)
        in_ket_qua(ts, so_cay, a.hang, cv, co_so, luot, a.nguong_kg)
        print(f"Kiểm tra khớp tim_tham_so.csv: R² ảnh {cv['theo_anh']['R2']:.5f} (CSV {r['R2_anh']:.5f})")

    # Học trên CẢ 5 fold: không chia val/test vì tham số + số cây đã chọn ở LẦN 1, đánh giá cũng đã làm ở LẦN 1
    m = tao_model(ts, so_cay)
    m.fit(d[DAC_TRUNG], d["kg"], verbose=False)
    m.save_model(MODEL_RA)                       # định dạng gốc XGBoost (.json): đọc được, không phụ thuộc pickle
    mae_da_hoc = float(np.mean(np.abs(m.predict(d[DAC_TRUNG]) - d["kg"])))

    # Độ quan trọng đặc trưng (chỉ có ở mô hình cuối): gain = mức giảm sai số khi đặc trưng được dùng để chia
    plt = plt_agg()
    gain = m.get_booster().get_score(importance_type="gain")
    gt = np.array([gain.get(c, 0) for c in DAC_TRUNG])
    gt = gt / max(gt.sum(), 1e-9) * 100
    (THU_MUC_RA / "bieu_do").mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(DAC_TRUNG, gt, color="#3b7dd8")
    for i, v in enumerate(gt):
        ax.text(v, i, f" {v:.1f} %", va="center")
    ax.set(xlabel="% tổng gain", title="Độ quan trọng đặc trưng (mô hình cuối)")
    fig.tight_layout(); fig.savefig(THU_MUC_RA / "bieu_do" / "do_quan_trong_dac_trung.png", dpi=130); plt.close(fig)

    # Thông tin để test_kg.py / pipeline dùng ĐÚNG như lúc train
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lay_dac_trung import NOI_HOP, TI_LE_KERNEL
    import xgboost
    thong_tin = {
        "thoi_gian": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dac_trung_theo_thu_tu": DAC_TRUNG,
        "dac_trung_tho": tho,
        "ti_le_kernel_phep_mo": TI_LE_KERNEL,
        "noi_hop": NOI_HOP,
        "don_vi": "pixel",
        "dieu_kien_anh": "RGB_9579: ảnh 960x540, camera thẳng đứng cao 1,88 m. Camera khác -> phải hiệu chỉnh tỉ lệ pixel.",
        "khoang_kg_da_hoc": [float(d["kg"].min()), float(d["kg"].max())],
        "du_lieu": loai,
        "hang_tham_so": a.hang,
        "tham_so": {"max_depth": int(ts["max_depth"]), "learning_rate": float(ts["learning_rate"]),
                    "min_child_weight": float(ts["min_child_weight"]), "subsample": SUBSAMPLE, "seed": SEED},
        "so_cay": so_cay,
        "danh_gia_5fold_CV": {"MAE_con_TB_kg": float(r["MAE_con_TB"]), "MAE_anh_TB_kg": float(r["MAE_anh_TB"]),
                              "R2_anh": float(r["R2_anh"]), "R2_con": float(r["R2_con"]),
                              "nguon": "tim_tham_so.csv + tom_tat.json (5-fold CV của đúng bộ này)"},
        "MAE_tren_du_lieu_da_hoc_kg": mae_da_hoc,
        "ghi_chu_MAE_da_hoc": "Sai số khi đoán lại chính dữ liệu đã học - CHỈ để so với MAE CV, KHÔNG phải điểm đánh giá.",
        "xgboost": xgboost.__version__,
    }
    (THU_MUC_RA / "thong_tin_mo_hinh.json").write_text(json.dumps(thong_tin, ensure_ascii=False, indent=2),
                                                       encoding="utf-8")
    print(f"\nĐã train trên {len(d)} ảnh / {loai['so_con']} con (cả 5 fold).")
    print(f"MAE trên dữ liệu đã học: {mae_da_hoc:.2f} kg  (chỉ để so sánh; điểm đánh giá = MAE CV {r['MAE_anh_TB']:.2f} kg/ảnh)")
    print(f"Mô hình: {MODEL_RA}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    a = doc_tham_so()
    THU_MUC_RA.mkdir(parents=True, exist_ok=True)
    if a.tim_tham_so:
        lan_1_tim_tham_so(a)
    else:
        lan_2_train_cuoi(a)


if __name__ == "__main__":
    main()
