
from __future__ import annotations
import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, time as datetime_time
from pathlib import Path
from typing import Any, Optional, Union
import cv2
from ultralytics import YOLO

@dataclass
class CauHinhPhatHienNguoi:
    model_path: str = "yolo11n.pt"
    # Ngưỡng conf giữ hộp người: ngày / đêm
    confidence: float = 0.5 
    confidence_dem: float = 0.35             
    # Conf trung bình tối thiểu để xác nhận 1 ID: ngày / đêm
    conf_xac_nhan: float = 0.55   
    conf_xac_nhan_dem: float = 0.40
    # Độ sáng TB dưới mức này -> chế độ đêm
    nguong_toi: float = 60.0       
    # Kích thước ảnh đưa vào YOLO
    image_size: int =640
    # Số khung đã xử lý 1 ID phải xuất hiện mới tạo sự kiện
    so_frame_xac_nhan: int = 10
    # Vắng bấy nhiêu khung liên tiếp thì xóa ID
    so_frame_vang_de_ket_thuc_dot: int = 10
    # Số khung chạy YOLO mỗi giây
    fps_xu_ly_muc_tieu: float = 5.0
    # Hộp phải cao >= 12% khung hình
    chieu_cao_box_toi_thieu: float = 0.12  
    # Tỉ lệ cao/rộng tối thiểu của hộp, 0 = tắt
    ti_le_cao_rong_toi_thieu: float = 0.0 
    # Số giây tối thiểu giữa 2 sự kiện cùng loại
    cooldown_giay: float = 30.0
    ma_o: str = "A"
    thu_muc_su_kien: str = "recordings/su_kien"
    # Giờ canh gác HH:MM; None = không canh gác (không có 'nghi_trom')
    gio_bat_dau_can_gac: Optional[str] = None
    gio_ket_thuc_can_gac: Optional[str] = None
# Phát hiện, theo dõi người và tạo sự kiện
class PhatHienNguoi:
    # Nạp model, khởi tạo trạng thái theo dõi và giờ canh gác
    def __init__(self, cau_hinh: CauHinhPhatHienNguoi):
        self.cfg = cau_hinh
        self.model = YOLO(self.cfg.model_path)
        self.trang_thai_id: dict[int, dict[str, Any]] = {}
        self.so_frame_vang_id: dict[int, int] = {}
        self.lan_cuoi_su_kien: dict[str, datetime] = {}
        self.hop_cuoi: list[tuple[int, int, int, int, int, float]] = []
        self.thong_tin_cuoi: str = ""
        self.thu_muc = Path(self.cfg.thu_muc_su_kien)
        self.thu_muc.mkdir(parents=True, exist_ok=True)
        self.file_log = self.thu_muc / "events.jsonl"
        self.gio_bat_dau = self._parse_time(self.cfg.gio_bat_dau_can_gac)
        self.gio_ket_thuc = self._parse_time(self.cfg.gio_ket_thuc_can_gac)
        if (self.gio_bat_dau is None) != (self.gio_ket_thuc is None):
            raise ValueError("Phai nhap ca gio bat dau va gio ket thuc can gac.")
    # Đổi chuỗi 'HH:MM' thành giờ
    @staticmethod
    def _parse_time(value: Optional[str]) -> Optional[datetime_time]:
        if value is None:
            return None
        try:
            return datetime.strptime(value, "%H:%M").time()
        except ValueError as exc:
            raise ValueError(
                f"Gio '{value}' khong hop le. Dung dinh dang HH:MM, vi du 22:00."
            ) from exc

    # Làm sáng ảnh tối bằng CLAHE, ảnh đủ sáng giữ nguyên
    def xu_ly_anh(self, frame: Any) -> Any:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.mean() >= 60:
            return frame
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # clipLimit càng lớn ảnh càng tương phản nhưng càng nhiễu
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # Kiểm tra thời điểm có nằm trong giờ canh gác không
    def trong_gio_can_gac(self, thoi_diem: datetime) -> bool:
        if self.gio_bat_dau is None or self.gio_ket_thuc is None:
            return False
        gio = thoi_diem.time()
        if self.gio_bat_dau <= self.gio_ket_thuc:
            return self.gio_bat_dau <= gio < self.gio_ket_thuc
        return gio >= self.gio_bat_dau or gio < self.gio_ket_thuc

    # Kiểm tra đã hết cooldown chưa, hết thì ghi mốc mới
    def _het_cooldown(self, loai: str, thoi_diem: datetime) -> bool:
        lan_cuoi = self.lan_cuoi_su_kien.get(loai)
        if lan_cuoi and (thoi_diem - lan_cuoi).total_seconds() < self.cfg.cooldown_giay:
            return False
        self.lan_cuoi_su_kien[loai] = thoi_diem
        return True

    # Lưu ảnh sự kiện
    def _luu_anh(self, frame: Any, loai: str, track_id: int) -> str:
        ten = f"{loai}_ID{track_id}_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
        duong_dan = self.thu_muc / ten
        if not cv2.imwrite(str(duong_dan), frame):
            raise IOError(f"Khong the ghi anh su kien: {duong_dan}")
        return str(duong_dan).replace("\\", "/")

    # Tạo sự kiện và ghi vào events.jsonl
    def _tao_su_kien(
        self, loai: str, muc: str, frame: Any, thoi_diem: datetime,
        so_nguoi: int, track_id: int, confidence: float, so_frame: int
    ) -> dict:
        anh = self._luu_anh(frame, loai, track_id)
        su_kien = {
            "loai": loai,
            "o": self.cfg.ma_o,
            "thoi_gian": thoi_diem.isoformat(timespec="seconds"),
            "muc": muc,
            "anh": anh,
            "du_lieu": {
                "so_nguoi": so_nguoi,
                "track_id": track_id,
                "so_frame_lien_tiep": so_frame,
                "do_tin_cay": round(confidence, 4),
                "gio_can_gac": self.trong_gio_can_gac(thoi_diem),
            },
        }
        with self.file_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(su_kien, ensure_ascii=False) + "\n")
        return su_kien

    # Loại hộp quá nhỏ hoặc sai tỉ lệ
    def _hop_hop_le(self, x1: int, y1: int, x2: int, y2: int, h_khung: int) -> bool:
        w_box, h_box = x2 - x1, y2 - y1
        if h_box < self.cfg.chieu_cao_box_toi_thieu * h_khung:
            return False
        if self.cfg.ti_le_cao_rong_toi_thieu > 0:
            if h_box / max(w_box, 1) < self.cfg.ti_le_cao_rong_toi_thieu:
                return False
        return True

    # Vẽ hộp người kèm ID và conf
    @staticmethod
    def _ve_hop(frame: Any, hop: tuple[int, int, int, int, int, float]) -> None:
        x1, y1, x2, y2, track_id, confidence = hop
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, f"PERSON ID:{track_id} {confidence:.2f}",
            (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2
        )

    # Vẽ lại kết quả gần nhất cho khung không chạy YOLO
    def ve_lai_ket_qua_cuoi(self, frame: Any) -> Any:
        frame_ve = frame.copy()
        for hop in self.hop_cuoi:
            self._ve_hop(frame_ve, hop)
        cv2.putText(frame_ve, self.thong_tin_cuoi, (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        return frame_ve

    # Hàm chính: xử lý 1 khung -> (khung đã vẽ, sự kiện mới)
    def xu_ly_khung_hinh(
        self, frame: Any, thoi_diem: Optional[datetime] = None
    ) -> tuple[Any, list[dict]]:
        if thoi_diem is None:
            thoi_diem = datetime.now()
        h_khung = frame.shape[0]
        do_sang = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
        la_toi = do_sang < self.cfg.nguong_toi
        conf_detect = self.cfg.confidence_dem if la_toi else self.cfg.confidence
        conf_xn = self.cfg.conf_xac_nhan_dem if la_toi else self.cfg.conf_xac_nhan
        frame_xuly = self.xu_ly_anh(frame)
        # classes=[0]: chỉ lấy người; persist=True: giữ ID giữa các khung
        ket_qua = self.model.track(
            source=frame_xuly,
            persist=True,
            tracker="botsort.yaml",
            classes=[0],
            conf=conf_detect,
            imgsz=self.cfg.image_size,
            verbose=False,
        )
        frame_da_ve = frame.copy()
        su_kien_moi: list[dict] = []
        ids_xuat_hien: set[int] = set()
        hop_hien_tai: list[tuple[int, int, int, int, int, float]] = []
        so_nguoi = 0
        if ket_qua and ket_qua[0].boxes is not None:
            boxes = ket_qua[0].boxes
            for box in boxes:
                if box.id is None:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if not self._hop_hop_le(x1, y1, x2, y2, h_khung):
                    continue
                so_nguoi += 1
                track_id = int(box.id[0].item())
                confidence = float(box.conf[0].item())
                ids_xuat_hien.add(track_id)

                if track_id not in self.trang_thai_id:
                    self.trang_thai_id[track_id] = {
                        "so_frame": 0,
                        "tong_conf": 0.0,
                        "da_tao_nguoi_la": False,
                        "da_tao_nghi_trom": False,
                    }
                state = self.trang_thai_id[track_id]
                state["so_frame"] += 1
                state["tong_conf"] += confidence
                self.so_frame_vang_id[track_id] = 0

                hop = (x1, y1, x2, y2, track_id, confidence)
                hop_hien_tai.append(hop)
                self._ve_hop(frame_da_ve, hop)

                # Xác nhận khi đủ số khung và conf trung bình đủ cao
                conf_tb = state["tong_conf"] / state["so_frame"]
                du_dieu_kien = (
                    state["so_frame"] >= self.cfg.so_frame_xac_nhan
                    and conf_tb >= conf_xn      
                )
                if du_dieu_kien and not state["da_tao_nguoi_la"]:
                    state["da_tao_nguoi_la"] = True
                    if self._het_cooldown("nguoi_la", thoi_diem):
                        su_kien_moi.append(self._tao_su_kien(
                            "nguoi_la", "canh_bao", frame_da_ve, thoi_diem,
                            so_nguoi, track_id, conf_tb, state["so_frame"]
                        ))

                if (du_dieu_kien and self.trong_gio_can_gac(thoi_diem)
                        and not state["da_tao_nghi_trom"]):
                    state["da_tao_nghi_trom"] = True
                    if self._het_cooldown("nghi_trom", thoi_diem):
                        su_kien_moi.append(self._tao_su_kien(
                            "nghi_trom", "khan_cap", frame_da_ve, thoi_diem,
                            so_nguoi, track_id, conf_tb, state["so_frame"]
                        ))
        for track_id in list(self.trang_thai_id):
            if track_id not in ids_xuat_hien:
                self.so_frame_vang_id[track_id] = self.so_frame_vang_id.get(track_id, 0) + 1
                if self.so_frame_vang_id[track_id] >= self.cfg.so_frame_vang_de_ket_thuc_dot:
                    del self.trang_thai_id[track_id]
                    del self.so_frame_vang_id[track_id]

        thong_tin = f"Persons: {so_nguoi} | Tracking IDs: {len(ids_xuat_hien)}"
        cv2.putText(frame_da_ve, thong_tin, (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        self.hop_cuoi = hop_hien_tai
        self.thong_tin_cuoi = thong_tin
        return frame_da_ve, su_kien_moi
# Đọc video/camera, chạy YOLO theo fps mục tiêu, in sự kiện và thống kê
def chay_nguon(
    nguon: Union[str, int],
    bo_phat_hien: PhatHienNguoi,
    hien_thi: bool = True,
) -> None:
    cap = cv2.VideoCapture(nguon)
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc camera/video: {nguon}")

    la_camera = isinstance(nguon, int)
    fps_nguon = cap.get(cv2.CAP_PROP_FPS)
    if not fps_nguon or fps_nguon <= 1 or fps_nguon > 240:
        fps_nguon = 30.0
    chu_ky_xu_ly = 1.0 / max(bo_phat_hien.cfg.fps_xu_ly_muc_tieu, 0.1)

    tong_doc = 0          
    tong_frame = 0        
    tong_thoi_gian = 0.0
    moc_xu_ly_cuoi = -1e9

    try:
        while True:
            thanh_cong, frame = cap.read()
            if not thanh_cong:
                break
            if la_camera:
                thoi_frame = time.perf_counter()
            else:
                thoi_frame = tong_doc / fps_nguon
            tong_doc += 1
            if thoi_frame - moc_xu_ly_cuoi >= chu_ky_xu_ly:
                moc_xu_ly_cuoi = thoi_frame
                bat_dau = time.perf_counter()
                frame_da_ve, su_kien_moi = bo_phat_hien.xu_ly_khung_hinh(frame)
                thoi_gian_xu_ly = time.perf_counter() - bat_dau

                tong_frame += 1
                tong_thoi_gian += thoi_gian_xu_ly

                for su_kien in su_kien_moi:
                    print(json.dumps(su_kien, ensure_ascii=False, indent=2))
            else:
                frame_da_ve = (
                    bo_phat_hien.ve_lai_ket_qua_cuoi(frame) if hien_thi else None
                )

            if hien_thi and frame_da_ve is not None:
                cv2.putText(
                    frame_da_ve,
                    f"FPS muc tieu: {bo_phat_hien.cfg.fps_xu_ly_muc_tieu:.1f}",
                    (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
                )
                cv2.imshow("Thanh vien 5 - Phat hien nguoi", frame_da_ve)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if hien_thi:
            cv2.destroyAllWindows()
    if tong_frame:
        print("\n===== THONG KE CHAY VIDEO =====")
        print("Tong frame doc:", tong_doc)
        print("Tong frame da xu ly:", tong_frame)
        print(f"Thoi gian xu ly trung binh/frame: "
              f"{tong_thoi_gian / tong_frame:.4f} giay")
        if tong_thoi_gian > 0:
            print(f"FPS xu ly toi da (neu khong bo frame): "
                  f"{tong_frame / tong_thoi_gian:.2f}")
# Đọc tham số dòng lệnh rồi chạy
def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO11n phat hien nguoi")
    parser.add_argument("--source", default="0",
                        help="Duong dan video hoac so camera, mac dinh 0")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--conf-confirm", type=float, default=0.55,
                        help="Conf trung binh de xac nhan")
    parser.add_argument("--confirm-frames", type=int, default=10,
                        help="So frame (da xu ly) de xac nhan mot ID")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="FPS xu ly muc tieu. Cang thap cang bot nhay")
    parser.add_argument("--min-height", type=float, default=0.12,
                        help="Chieu cao box toi thieu / chieu cao khung")
    parser.add_argument("--cooldown", type=float, default=30.0,
                        help="Giay giua 2 su kien cung loai")
    parser.add_argument("--guard-start", default=None,
                        help="Gio bat dau can gac, HH:MM")
    parser.add_argument("--guard-end", default=None,
                        help="Gio ket thuc can gac, HH:MM")
    parser.add_argument("--pen", default="A", help="Ma o chuong")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    nguon: Union[str, int] = (
        int(args.source) if args.source.isdigit() else args.source
    )

    cfg = CauHinhPhatHienNguoi(
        model_path=args.model,
        confidence=args.conf,
        conf_xac_nhan=args.conf_confirm,
        so_frame_xac_nhan=args.confirm_frames,
        fps_xu_ly_muc_tieu=args.fps,
        chieu_cao_box_toi_thieu=args.min_height,
        cooldown_giay=args.cooldown,
        gio_bat_dau_can_gac=args.guard_start,
        gio_ket_thuc_can_gac=args.guard_end,
        ma_o=args.pen,
    )

    bo_phat_hien = PhatHienNguoi(cfg)
    chay_nguon(nguon, bo_phat_hien, hien_thi=not args.no_display)
if __name__ == "__main__":
    main()