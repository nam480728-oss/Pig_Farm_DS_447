"""Test phát hiện người trên 1 video: chỉ XEM, không lưu gì.

Kế thừa nguyên bộ phát hiện trong nghi_trom.py (YOLO11n + BoT-SORT, chế độ
ngày/đêm, lọc hộp nhỏ, xác nhận 10 khung, cooldown, sự kiện nguoi_la/nghi_trom),
chỉ khác 3 điểm:
  1. Video phát đúng tốc độ thật (nghi_trom.py chạy nhanh nhất có thể).
  2. Sự kiện chỉ in ra terminal + hiện chữ trên video, KHÔNG lưu ảnh / events.jsonl.
  3. Mặc định canh gác cả ngày để luôn thấy được cả 'nguoi_la' và 'nghi_trom'.

Cách chạy (đứng ở thư mục models/Person_det):
    python test_nguoi.py video_test/demo2.mp4
    python test_nguoi.py video_test/co_nguoi_toi.mp4 --conf 0.4
    python test_nguoi.py <video> --guard-start 22:00 --guard-end 05:00   # canh gác theo giờ
Bấm q hoặc Esc (hoặc tắt cửa sổ) để dừng.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Cho phép import nghi_trom.py dù đang đứng ở thư mục nào
THU_MUC = Path(__file__).resolve().parent
sys.path.insert(0, str(THU_MUC))
from nghi_trom import CauHinhPhatHienNguoi, PhatHienNguoi  # noqa: E402

# Giá trị mặc định lấy thẳng từ nghi_trom.py -> sửa bên đó thì test tự theo
MAC_DINH = CauHinhPhatHienNguoi()
TEN_CUA_SO = "Test phat hien nguoi (q / Esc de thoat)"
GIAY_HIEN_CANH_BAO = 3.0  # chữ cảnh báo sự kiện hiện trên video bao nhiêu giây


class PhatHienNguoiTest(PhatHienNguoi):
    """Y hệt PhatHienNguoi, chỉ tắt phần LƯU (ảnh + log) và cho phép canh gác cả ngày."""

    def __init__(self, cau_hinh: CauHinhPhatHienNguoi, canh_gac_ca_ngay: bool):
        # __init__ gốc luôn tạo thư mục sự kiện -> trỏ vào thư mục tạm của hệ thống
        # (đã có sẵn) để không sinh thư mục rác trong dự án
        cau_hinh.thu_muc_su_kien = tempfile.gettempdir()
        super().__init__(cau_hinh)
        # _tao_su_kien gốc ghi 1 dòng vào file_log -> đổi sang thiết bị rỗng (ghi rồi mất)
        self.file_log = Path(os.devnull)
        self.canh_gac_ca_ngay = canh_gac_ca_ngay

    def _luu_anh(self, frame: Any, loai: str, track_id: int) -> str:
        # Ghi đè: không lưu ảnh sự kiện
        return "(test - khong luu anh)"

    def trong_gio_can_gac(self, thoi_diem: datetime) -> bool:
        # Không nhập giờ canh gác -> coi như canh gác cả ngày; có nhập thì dùng logic gốc
        if self.canh_gac_ca_ngay:
            return True
        return super().trong_gio_can_gac(thoi_diem)


def ghi_chu(frame: Any, chu: str, vi_tri: tuple[int, int], mau: tuple[int, int, int],
            co: float = 0.6, day: int = 2) -> None:
    """Viết chữ có viền đen để dễ đọc trên cả nền sáng lẫn nền tối."""
    cv2.putText(frame, chu, vi_tri, cv2.FONT_HERSHEY_SIMPLEX, co, (0, 0, 0), day + 2)
    cv2.putText(frame, chu, vi_tri, cv2.FONT_HERSHEY_SIMPLEX, co, mau, day)


def chay_video(duong_dan: Path, bo: PhatHienNguoiTest) -> None:
    """Phát video đúng tốc độ thật, chạy YOLO theo nhịp fps_xu_ly_muc_tieu như nghi_trom.py."""
    cap = cv2.VideoCapture(str(duong_dan))
    if not cap.isOpened():
        print(f"Không mở được video: {duong_dan}")
        sys.exit(1)

    # FPS của video (sai / không có thì coi là 30, giống chay_nguon gốc)
    fps_nguon = cap.get(cv2.CAP_PROP_FPS)
    if not fps_nguon or fps_nguon <= 1 or fps_nguon > 240:
        fps_nguon = 30.0
    tong_so_khung = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    thoi_luong = tong_so_khung / fps_nguon
    rong = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cao = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    chu_ky_xu_ly = 1.0 / max(bo.cfg.fps_xu_ly_muc_tieu, 0.1)  # 5 khung/giây -> 0,2 s
    print(f"Video: {duong_dan.name} | {rong}x{cao} | {fps_nguon:.1f} fps | {thoi_luong:.1f} s")
    print(f"YOLO chạy {bo.cfg.fps_xu_ly_muc_tieu:g} khung/giây video, xác nhận sau "
          f"{bo.cfg.so_frame_xac_nhan} khung | canh gác: "
          f"{'cả ngày' if bo.canh_gac_ca_ngay else f'{bo.cfg.gio_bat_dau_can_gac} -> {bo.cfg.gio_ket_thuc_can_gac}'}")

    # Chạy thử model 1 lần trên ảnh đen (predict, không đụng tracker) để nạp GPU trước,
    # tránh khung đầu tiên bị giật
    bo.model.predict(np.zeros((cao or 480, rong or 640, 3), np.uint8), classes=[0],
                     imgsz=bo.cfg.image_size, verbose=False)

    cv2.namedWindow(TEN_CUA_SO, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # kéo giãn được
    if rong and cao:
        cv2.resizeWindow(TEN_CUA_SO, rong, cao)

    so_doc = 0                    # số khung đã đọc (kể cả khung bỏ qua)
    so_bo_qua = 0                 # khung bị bỏ vì máy chậm hơn video
    tong_frame = 0                # số khung đã chạy YOLO
    tong_thoi_gian = 0.0
    moc_xu_ly_cuoi = -1e9
    dem_su_kien: dict[str, int] = {}
    canh_bao: list[tuple[str, tuple[int, int, int], float]] = []  # (chữ, màu, hết hạn - giây video)
    che_do = ""
    t_bat_dau = time.perf_counter()

    try:
        while True:
            thanh_cong, frame = cap.read()
            if not thanh_cong:
                break
            thoi_frame = so_doc / fps_nguon  # vị trí khung trong video (giây)
            so_doc += 1

            if thoi_frame - moc_xu_ly_cuoi >= chu_ky_xu_ly:
                # Đến lượt: chạy đúng hàm xử lý gốc của nghi_trom.py
                moc_xu_ly_cuoi = thoi_frame
                bat_dau = time.perf_counter()
                frame_da_ve, su_kien_moi = bo.xu_ly_khung_hinh(frame)
                tong_thoi_gian += time.perf_counter() - bat_dau
                tong_frame += 1

                # Tính lại độ sáng chỉ để HIỂN THỊ chế độ ngày/đêm (cùng ngưỡng với gốc)
                do_sang = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
                che_do = (f"{'DEM' if do_sang < bo.cfg.nguong_toi else 'NGAY'} "
                          f"(do sang {do_sang:.0f}, nguong {bo.cfg.nguong_toi:g})")

                for su_kien in su_kien_moi:
                    print(f"\n[{thoi_frame:6.1f} s trong video] Sự kiện mới:")
                    print(json.dumps(su_kien, ensure_ascii=False, indent=2))
                    loai = su_kien["loai"]
                    dem_su_kien[loai] = dem_su_kien.get(loai, 0) + 1
                    mau = (0, 0, 255) if loai == "nghi_trom" else (0, 165, 255)  # đỏ / cam
                    canh_bao.append((f"{loai.upper()} - ID {su_kien['du_lieu']['track_id']} "
                                     f"(conf TB {su_kien['du_lieu']['do_tin_cay']:.2f})",
                                     mau, thoi_frame + GIAY_HIEN_CANH_BAO))
            else:
                # Chưa đến lượt: vẽ lại kết quả lần xử lý gần nhất (hàm gốc)
                frame_da_ve = bo.ve_lai_ket_qua_cuoi(frame)

            # Dòng thông tin: FPS mục tiêu (giống gốc) + thời gian video + chế độ ngày/đêm
            ghi_chu(frame_da_ve, f"FPS muc tieu: {bo.cfg.fps_xu_ly_muc_tieu:.1f}",
                    (15, 60), (255, 255, 0))
            ghi_chu(frame_da_ve, f"{thoi_frame:5.1f} / {thoi_luong:.1f} s | Che do: {che_do}",
                    (15, 90), (255, 255, 255), 0.5, 2)
            # Chữ cảnh báo sự kiện ở đáy khung, giữ trong GIAY_HIEN_CANH_BAO giây
            canh_bao = [c for c in canh_bao if c[2] > thoi_frame]
            for i, (chu, mau, _) in enumerate(canh_bao[-3:]):
                ghi_chu(frame_da_ve, chu, (15, frame_da_ve.shape[0] - 20 - 32 * i), mau, 0.8, 2)

            cv2.imshow(TEN_CUA_SO, frame_da_ve)
            if so_doc == 1:
                # Khung đầu tốn thời gian mở cửa sổ + khởi tạo tracker -> bắt đầu tính giờ
                # từ đây, nếu không sẽ phải bỏ mất vài khung đầu để đuổi kịp
                t_bat_dau = time.perf_counter() - 1 / fps_nguon

            # Giữ tốc độ thật: đợi tới đúng thời điểm của khung kế tiếp
            han = t_bat_dau + so_doc / fps_nguon
            phim = cv2.waitKey(max(1, int((han - time.perf_counter()) * 1000))) & 0xFF
            if phim in (ord("q"), 27):  # q hoặc Esc
                break
            if cv2.getWindowProperty(TEN_CUA_SO, cv2.WND_PROP_VISIBLE) < 1:  # bấm X tắt cửa sổ
                break
            # Máy chậm hơn video quá 1 khung -> bỏ bớt khung để video không bị chậm lại
            while time.perf_counter() - t_bat_dau > (so_doc + 1) / fps_nguon:
                if not cap.grab():
                    break
                so_doc += 1
                so_bo_qua += 1
    except KeyboardInterrupt:
        print("\nĐã dừng (Ctrl+C).")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Thống kê (giống chay_nguon gốc + số sự kiện, số khung bỏ qua)
    print("\n===== THONG KE TEST VIDEO =====")
    print(f"Đã xem: {so_doc / fps_nguon:.1f} / {thoi_luong:.1f} s ({so_doc} khung đọc, "
          f"{so_bo_qua} khung bỏ qua vì máy chậm)")
    print("Tổng khung đã chạy YOLO:", tong_frame)
    if tong_frame:
        print(f"Thời gian xử lý trung bình/khung: {tong_thoi_gian / tong_frame:.4f} giây")
        if tong_thoi_gian > 0:
            print(f"FPS xử lý tối đa (nếu không bỏ khung): {tong_frame / tong_thoi_gian:.2f}")
    print("Sự kiện:", ", ".join(f"{k} x{v}" for k, v in dem_su_kien.items()) or "không có")


def tim_video(chuoi: str) -> Path:
    """Tìm video theo đường dẫn nhập vào; không thấy thì thử tính từ thư mục chứa file này."""
    for p in (Path(chuoi), THU_MUC / chuoi):
        if p.is_file():
            return p.resolve()
    print(f"Không tìm thấy video: {chuoi}")
    sys.exit(1)


def main() -> None:
    # Các tham số giống hệt main() của nghi_trom.py (trừ --source/--no-display),
    # mặc định lấy từ CauHinhPhatHienNguoi
    ap = argparse.ArgumentParser(description="Test phát hiện người trên video (chỉ xem, không lưu)")
    ap.add_argument("video", help="Đường dẫn file video")
    ap.add_argument("--model", default=str(THU_MUC / MAC_DINH.model_path),
                    help="File trọng số YOLO (mặc định yolo11n.pt cạnh file này)")
    ap.add_argument("--conf", type=float, default=MAC_DINH.confidence)
    ap.add_argument("--conf-confirm", type=float, default=MAC_DINH.conf_xac_nhan,
                    help="Conf trung bình để xác nhận")
    ap.add_argument("--confirm-frames", type=int, default=MAC_DINH.so_frame_xac_nhan,
                    help="Số khung (đã xử lý) để xác nhận một ID")
    ap.add_argument("--fps", type=float, default=MAC_DINH.fps_xu_ly_muc_tieu,
                    help="Số khung chạy YOLO mỗi giây video")
    ap.add_argument("--min-height", type=float, default=MAC_DINH.chieu_cao_box_toi_thieu,
                    help="Chiều cao hộp tối thiểu / chiều cao khung")
    ap.add_argument("--cooldown", type=float, default=MAC_DINH.cooldown_giay,
                    help="Giây giữa 2 sự kiện cùng loại")
    ap.add_argument("--guard-start", default=None, help="Giờ bắt đầu canh gác HH:MM (bỏ trống = cả ngày)")
    ap.add_argument("--guard-end", default=None, help="Giờ kết thúc canh gác HH:MM")
    ap.add_argument("--pen", default=MAC_DINH.ma_o, help="Mã ô chuồng")
    a = ap.parse_args()

    video = tim_video(a.video)
    cfg = CauHinhPhatHienNguoi(
        model_path=a.model,
        confidence=a.conf,
        conf_xac_nhan=a.conf_confirm,
        so_frame_xac_nhan=a.confirm_frames,
        fps_xu_ly_muc_tieu=a.fps,
        chieu_cao_box_toi_thieu=a.min_height,
        cooldown_giay=a.cooldown,
        gio_bat_dau_can_gac=a.guard_start,
        gio_ket_thuc_can_gac=a.guard_end,
        ma_o=a.pen,
    )
    canh_gac_ca_ngay = a.guard_start is None and a.guard_end is None
    chay_video(video, PhatHienNguoiTest(cfg, canh_gac_ca_ngay))


if __name__ == "__main__":
    main()
