# CLAUDE.md — UTR-Track

> Hướng dẫn cốt lõi. Các quy tắc chi tiết nằm trong [.claude/rules/](.claude/rules/) và chỉ được đọc khi cần (xem bảng định tuyến ở §7).

## 1. Ngôn ngữ & phong cách

- Trả lời bằng **tiếng Việt**. Giữ nguyên thuật ngữ kỹ thuật tiếng Anh (IDF1, MOTA, sigma point, covariance...), không dịch ép.
- Kỹ thuật chặt chẽ nhưng thực dụng: ưu tiên lệnh/code cụ thể hơn lời khuyên chung chung.
- **Không bịa** số liệu thực nghiệm, citation, dataset hay benchmark. Phân biệt rõ: đã verify / giả định / giả thuyết.

## 2. Dự án

**UTR-Track** — framework tracking tổn thương trong video nội soi tiêu hóa trên, kiến trúc StrongSORT-style.

| Thành phần | Vai trò |
|---|---|
| YOLOv8 (Ultralytics) | Phát hiện tổn thương, chọn model theo cơ quan |
| OSNet + DCN (`osnet_dcn_x0_5_endocv.pt`) | Trích xuất đặc trưng ReID |
| **TL-UKF** | Unscented KF + transfer-learning covariance |
| **VTA** | Virtual Trajectory Augmentation — sinh box ảo khi mất detection |
| Hybrid Matching | Kết hợp appearance + motion + IoU |

**Mục tiêu nghiên cứu:** giữ identity và ổn định quỹ đạo trong video nội soi (biến dạng phi cứng, camera motion, blur, specular highlight, occlusion, tổn thương giống nhau).

## 3. Bản đồ mã nguồn

```
utrtrack_pipeline.py                 # Entry point: detect → track → xuất video/CSV/MOT (--tracker_type utrtrack_v0|utrtrack_v1|tlukf|xysr)
utrtrack_versions.py                 # ★ Cấu hình chính thức v0/v1 (nguồn chân lý duy nhất), build_tracker, NMS không phân biệt lớp
run_utrtrack_v0.py / run_utrtrack_v1.py   # Chạy đúng cấu hình v0 / v1
examples/track_video_api.py          # Dùng v0/v1 qua API, ghi đầu ra THÔ của tracker
boxmot/                              # Fork BoxMOT (AGPL-3.0) — thư viện tracking
  boxmot/trackers/utrtrack/          # ★ UTR-Track-v0 = UTRTrack (alias UTRTrackV0): utrtrack.py + sort/{track,tracker,linear_assignment,...}.py
  boxmot/trackers/utrtrack_v1/       # ★ UTR-Track-v1 = UTRTrackV1: utrtrack_v1.py + sort/{track,tracker}.py (dùng chung detection/iou_matching/linear_assignment của utrtrack/sort)
  boxmot/trackers/strongsort/        # Baseline StrongSORT (xysr)
  boxmot/trackers/endoocsort/        # Biến thể OC-SORT cho nội soi
  boxmot/motion/kalman_filters/aabb/ # ★ tlukf.py (v0), xysr_tlukf_v1.py (v1), xysr_tlukf.py (endoocsort), xysr_kf.py, xyah_kf.py
  boxmot/appearance/                 # ReID backbones/backends
  tests/{unit,performance}/
paper/                               # MBEC_Ver3.tex, refs.bib, images/, review/ (git-ignored)
luanvan/                             # Luận văn (git-ignored)
plan.md                              # Roadmap A.x (pipeline) + B.x (bài báo) — đọc trước khi đề xuất việc mới
```

Điểm vào tracker: `boxmot/boxmot/__init__.py` export `UTRTrack` (= `UTRTrackV0`) và `UTRTrackV1`.
v1 (luận văn) là phiên bản chính thức hiện tại; v0 là bản của bài báo MBEC. Khác biệt v0/v1: README, mục "UTR-Track-v0 and UTR-Track-v1".

## 4. Lệnh thường dùng

```bash
# UTR-Track-v1 (cấu hình luận văn) / UTR-Track-v0 (cấu hình bài báo)
python run_utrtrack_v1.py --video data/video_test --output_dir content/exp_XXX_v1
python run_utrtrack_v0.py --video data/video_test --output_dir content/exp_XXX_v0

# Pipeline đầy đủ; --tracker_type mặc định tlukf (= v0, giữ nghĩa các lệnh cũ)
python utrtrack_pipeline.py --video_dir data/video_test --model_dir model_yolo \
    --output_dir content/exp_XXX --tracker_type utrtrack_v1

# Baseline StrongSORT
python utrtrack_pipeline.py --video_dir data/video_test --model_dir model_yolo \
    --output_dir content/exp_XXX_ss --tracker_type xysr

# Ép một model YOLO duy nhất (bỏ qua chọn theo cơ quan) — dùng cho ablation detector
python utrtrack_pipeline.py --model_weights model_yolo/daday.pt ...

# Test (chạy trong thư mục boxmot/)
cd boxmot && python -m pytest tests/unit -q
cd boxmot && python -m pytest tests/unit/test_trackers.py -q -k utrtrack

# Smoke test import
python -c "from boxmot import UTRTrack, UTRTrackV1; print('ok')"

# Lint (cấu hình trong boxmot/pyproject.toml: flake8, max-line-length=120)
cd boxmot && python -m flake8 boxmot/trackers/utrtrack
```

Tham số CLI đầy đủ: xem `main()` trong [utrtrack_pipeline.py](utrtrack_pipeline.py) — nguồn chân lý, đừng chép lại từ README.

## 5. Môi trường

- Python 3.12.7 · PyTorch 2.8.0+cu128 · CUDA khả dụng · Ultralytics 8.3.147 · pytest 7.4.4
- GPU: **RTX 4070, 12 GB VRAM** — cân nhắc batch size / AMP trước khi đề xuất thí nghiệm nặng.
- Không có venv riêng trong repo; dùng Python hệ thống trên Windows. Shell mặc định là PowerShell.
- `boxmot/` dùng `uv` upstream, nhưng repo này cài qua `pip install -r requirements.txt`.

## 6. Quy tắc bắt buộc (không được vi phạm)

1. **Không bao giờ ghi đè kết quả thí nghiệm.** Mỗi lần chạy dùng `--output_dir` mới, đặt tên theo `exp_<id>_<biến><giá trị>`, ví dụ `content/exp_014_tau0.7_lambda0.5`.
2. **Không tự ý đổi điều kiện thí nghiệm** giữa baseline và proposed. Giữ cố định detector, resolution, split, evaluation protocol — trừ khi biến đó chính là đối tượng nghiên cứu.
3. **Không khẳng định metric cải thiện** khi chưa đối chiếu số thật. Relative improvement = `(New − Old) / Old × 100%`; với metric càng thấp càng tốt (IDs, FP, FN, RMSE) phải nói rõ chiều.
4. **Không claim hiệu quả lâm sàng** từ metric kỹ thuật. Đây là hệ thống nghiên cứu, chưa validate lâm sàng.
5. **`paper/` và `luanvan/` đang bị git-ignore** — sửa file ở đó không được commit; đừng giả định chúng nằm trong lịch sử git.
6. **Không commit weights** (`*.pt`, `*.onnx`, `*.engine`) hay thư mục `data/`, `content*/`.
7. `boxmot/` là fork **AGPL-3.0** — cân nhắc license trước khi public kèm bài báo.
8. Trước khi đề xuất hướng đi mới, **đọc [plan.md](plan.md)** — roadmap đã có thứ tự ưu tiên A.1→B.8.

## 7. Định tuyến quy tắc chi tiết (path-scoped)

Đọc file rule tương ứng **trước khi** làm việc trong phạm vi đó. Không đọc trước — tiết kiệm context.

| Khi làm việc với... | Đọc |
|---|---|
| `boxmot/**/kalman_filters/**` · toán TL-UKF, sigma point, covariance | [.claude/rules/kalman-tlukf.md](.claude/rules/kalman-tlukf.md) |
| `boxmot/**/trackers/**` · association, track lifecycle, VTA | [.claude/rules/tracker-code.md](.claude/rules/tracker-code.md) |
| `utrtrack_pipeline.py` · I/O, định dạng xuất, vẽ box | [.claude/rules/pipeline.md](.claude/rules/pipeline.md) |
| Báo cáo/so sánh IDF1, MOTA, IDs, FP/FN | [.claude/rules/tracking-eval.md](.claude/rules/tracking-eval.md) |
| YOLOv8, dataset, mAP, per-class | [.claude/rules/detection-yolo.md](.claude/rules/detection-yolo.md) |
| Thiết kế thí nghiệm, ablation, reproducibility | [.claude/rules/experiment-protocol.md](.claude/rules/experiment-protocol.md) |
| Viết/sửa nội dung khoa học, phản biện | [.claude/rules/paper-writing.md](.claude/rules/paper-writing.md) |
| `paper/**/*.tex`, `*.bib`, lỗi biên dịch LaTeX | [.claude/rules/latex.md](.claude/rules/latex.md) |
| Hình vẽ, plot, qualitative results | [.claude/rules/visualization.md](.claude/rules/visualization.md) |
| Lỗi CUDA/driver/OOM/môi trường | [.claude/rules/gpu-env.md](.claude/rules/gpu-env.md) |
| Dữ liệu bệnh nhân, phát ngôn y khoa | [.claude/rules/medical-ai.md](.claude/rules/medical-ai.md) |
| Chọn skill nào trong 95 skill đã cài | [.claude/rules/skills-routing.md](.claude/rules/skills-routing.md) |

## 8. Quy ước code

- Modular · config-driven · đường dẫn tường minh · an toàn khi chọn GPU/CPU (`torch.device`, fallback CPU).
- Thêm **docstring + type hint** cho mọi hàm/lớp mới (theo `boxmot/AGENTS.md`).
- Commit prefix: `feat:` `fix:` `refactor:` `docs:` `perf:` `ci:` `test:`.
- Ưu tiên CLI arg / config file hơn hard-code. Nếu buộc phải hard-code, để `TODO:` kèm lý do.
- Khi sửa code sinh ra số liệu đã công bố: ghi rõ trong commit message rằng số liệu cũ cần chạy lại.

## 9. Nguyên tắc cốt lõi

> Problem → Hypothesis → Method → Controlled Experiment → Evidence → Explanation → Limitation → Conclusion

Mỗi thay đổi kiến trúc phải có: động cơ rõ ràng, hiệu ứng đo được, bằng chứng thực nghiệm diễn giải được. Metric cao hơn không phải là mục tiêu — luận cứ khoa học bảo vệ được mới là.
