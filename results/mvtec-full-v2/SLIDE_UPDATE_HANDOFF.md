# Handoff cho GPT cập nhật slide

## Kết quả được phép dùng

Đã chạy xong full MVTec AD: 15 category × 3 seed, 450 dòng metric, mọi model dùng cùng feature/split.
NBD image AUROC: **78.79 ± 2.37%**.
Dùng nguyên bảng `summary.csv` và `results_table.tex`, không chọn riêng seed/category tốt.

| Method | Image AUROC (%) | AP (%) | Test FPR (%) | Test TPR (%) |
|---|---:|---:|---:|---:|
| PatchScore_same_centers | 95.86 ± 0.23 | 98.67 ± 0.05 | 11.36 | 79.28 |
| PatchScore_byte_budget | 98.56 ± 0.21 | 99.59 ± 0.06 | 13.23 | 87.04 |
| RBF_SVDD | 78.20 ± 0.26 | 90.46 ± 0.17 | 6.86 | 45.26 |
| DeepSVDD_head | 81.85 ± 0.52 | 91.58 ± 0.32 | 6.88 | 49.56 |
| DROCC_head | 54.45 ± 5.43 | 75.73 ± 3.29 | 3.51 | 8.38 |
| Bubble_B | 82.70 ± 1.81 | 93.19 ± 0.85 | 11.78 | 54.04 |
| Bubble_BA | 82.80 ± 1.62 | 93.26 ± 0.75 | 11.07 | 53.80 |
| Bubble_BAD | 79.72 ± 2.46 | 91.53 ± 1.29 | 10.31 | 47.72 |
| Bubble_BAF | 82.82 ± 1.88 | 93.27 ± 0.91 | 10.89 | 53.20 |
| NBD | 78.79 ± 2.37 | 91.06 ± 1.44 | 10.81 | 47.27 |

- NBD_minus_same_centers: -17.07 ± 2.22 AUROC percentage points.
- NBD_minus_byte_budget: -19.77 ± 2.29 AUROC percentage points.

## Sửa slide review.pdf

- Trang 3: backbone của thí nghiệm mới là Wide ResNet-50-2 IMAGENET1K_V1 frozen, layer2+layer3, 1536 chiều. ResNet-18 chỉ thuộc các notebook trước đó. PatchScore vẫn là baseline nearest-memory, không gọi là full PatchCore. Đổi image max thành top-1% mean trong bảng so sánh mới.
- Trang 4–6: ghi rõ Deep SVDD-head và DROCC-head trên shared frozen CNN features. AE 50 + SVDD 100 epoch; DROCC 100 epoch/10 warmup/50 ascent. SVDD giải dual QP RBF trên coreset 2048 patch normal; ghi rõ adaptation.
- Trang 7–8: giữ WDBC/Wine/Digits nếu cần lịch sử, gắn nhãn pilot cũ, không trộn điểm với MVTec. Thêm protocol mới: 3629 train normal, 467 test normal, 1258 test anomaly; split train normal 60/20/20; seeds 0/1/2.
- Trang 9–16: NBD đã được đánh giá thực nghiệm theo công thức B+A+D+F. Thêm ablation B, BA, BAD, BAF, full NBD; giữ mọi kết quả dù kém baseline. Không đổi công thức để chạy theo điểm test.
- Trang 16: phân biệt component normal-tail scaling và image threshold calibration; score không phải xác suất anomaly. Threshold alpha=0.05 có thể vô hạn khi thiếu normal calibration, gồm toothbrush. SD là sample SD của ba category-macro seed means.
- Trang 17: đổi trạng thái NBD từ proposal/not benchmarked sang measured on full MVTec AD under controlled shared-CNN protocol. Không tuyên bố official-paper benchmark reproduction hoặc pixel-level performance.

## Hình nên chèn

1. `plots/macro_auroc.png`: bảng xếp hạng có error bar và đủ 10 variants.
2. `plots/category_auroc_full_scale.png`: kết quả đủ 15 categories, thang màu 0–100%.
3. `plots/paired_deltas.png`: chênh lệch NBD với same-center và byte-budget controls.
4. `plots/training_losses.png`: lịch sử train thật. Đọc thêm per-run Deep SVDD variance diagnostics trước khi nói về collapse.

## Nguồn bằng chứng

Đọc `REPORT.md`, `verification.json`, `per_category_per_seed.csv`, `split_counts.csv`,
`predictions_all.csv`, `protocol_lock.json`, `backbone.json`. Code ở repo OCC,
config `configs/full.json`, lệnh chạy lại `scripts/run_full.sh`.
Không tái sử dụng các số pilot cũ làm kết quả mới; không tự điền số còn thiếu.

## Diễn giải kết quả và nội dung cần tránh hiểu sai

Thông điệp chính: **NBD B+A+D+F đạt 78.79 ± 2.37% image AUROC và chưa vượt hai control PatchScore trong protocol đã chạy.** Same-center đạt 95.86 ± 0.23%; byte-budget đạt 98.56 ± 0.21%. Đây là kết quả đầy đủ của 15 category × 3 seed, không thay NBD bằng ablation có điểm cao hơn.

- NBD thấp hơn same-center 17.07 ± 2.22 điểm phần trăm AUROC; thấp hơn byte-budget 19.77 ± 2.29 điểm. Trên 45 cặp category–seed, NBD thắng/thua lần lượt 5/40 và 4/41, không có hòa. Các cặp dùng lại test set theo seed; các con số này là mô tả, không phải kiểm định ý nghĩa thống kê.
- Ablation BAF đạt 82.82 ± 1.88%, cao hơn full NBD. Giữ tên BAF; không đưa điểm này thành “final NBD”.
- Thêm D vào BA: −3.07 ± 0.92 điểm AUROC; thêm D vào BAF: −4.03 ± 0.73 điểm. Thêm A vào B: +0.10 ± 0.20; thêm F vào BA: +0.03 ± 0.61; thêm F vào BAD: −0.93 ± 0.33. Đây là khác biệt đo được trong cấu hình này, chưa chứng minh nguyên nhân hoặc kết luận chung về diffusion/frame consistency.
- Target calibration alpha=0.05 không phải measured test FPR=5%. NBD có test FPR macro **10.81%**, TPR **47.27%**. Phải ghi số đo thật. Toothbrush có 12 ảnh threshold-calibration nên ngưỡng +inf; FPR/TPR bằng 0 ở ngưỡng này được giữ nguyên.
- Deep SVDD-head đạt **81.85 ± 0.52%**. Tỉ lệ phương sai output normal cuối/đầu có median 0.0009754, min 0.0001308, max 0.0116951 trên 45 lượt. Phương sai co mạnh là chẩn đoán cần công bố; riêng nó chưa đủ kết luận mathematical collapse hoặc lỗi triển khai.
- DROCC-head đạt **54.45 ± 5.43%**, TPR **8.38%**. Loss cuối có median 1.3862944 trên 45 lượt. Báo cáo kết quả yếu của adaptation này; không suy rộng thành điểm paper DROCC gốc.
- Neural head batch size **8192** là lựa chọn mới của shared-feature protocol. Toàn bộ fit patch được dùng trong mọi epoch: tổng **11,250 dòng epoch history**, **1,278,312,000 lượt đưa patch normal qua các epoch**, **162,750 optimizer steps**. Các con số này cộng ba stage AE/SVDD/DROCC; không phải số ảnh độc nhất hoặc số ascent step.
- Mỗi seed có tổng 2,174 ảnh fit, 721 ảnh score calibration, 734 ảnh threshold calibration; test gồm 467 normal + 1,258 anomaly. Split diễn ra theo ảnh, sau đó mới lấy patch, và mọi model trong một run dùng cùng split/cache.
- RBF SVDD thật sự giải dual QP nhưng trên coreset normal 2,048 patch đã khai báo. Đây không phải full dense-kernel optimization trên tất cả patch. Deep SVDD/DROCC đều là head trên frozen Wide ResNet-50-2. Không gọi bảng này là tái lập nguyên xi bảng benchmark gốc của các paper.
- Pilot WDBC/Wine/Digits trước đó không được rerun và không đóng góp số nào vào bảng MVTec mới. Không có kết quả pixel AUROC/AUPRO được đo trong đợt này.

Ưu tiên hình `plots/category_auroc_full_scale.png` cho slide: thang màu 0–100 thể hiện rõ cả kết quả dưới 50%. Hình `category_auroc.png` nguyên gốc vẫn được giữ; nó bão hòa màu ở các giá trị dưới 50%. Hai hình dùng cùng CSV và cùng giá trị số.

## Trạng thái bàn giao đã kiểm chứng

- Toàn bộ 45 lượt train/test đã hoàn tất; 450 dòng metric và 51,750 dự đoán ảnh đã qua audit CSV độc lập.
- Đã trích xuất lại đủ 15 CNN feature caches và replay 135 nhóm calibration/threshold/test từ checkpoint. Sai số tuyệt đối lớn nhất: **0.0**.
- Source, config, dataset, backbone và từng artifact của 45 lượt đều được đối chiếu SHA-256; bản sao dataset/checkpoint/source đã lưu local trước khi tắt GPU.
- Vast.ai xác nhận `cur_state=stopped`, `actual_status=exited`, `intended_status=stopped` lúc 2026-09-14T06:23:35.982472+00:00.
- Xem `reverification.json`, `independent_csv_audit.json`, `backup_verified.json`, `dataset_backup_verified.json`, `execution_closure.json`, `published_artifacts_sha256.json`.
- Toàn bộ bảng/CSV/biểu đồ/protocol và diagnostics có trong Git. Dataset, checkpoint lớn và tài liệu gốc nằm trong bản sao local ngoài Git.
