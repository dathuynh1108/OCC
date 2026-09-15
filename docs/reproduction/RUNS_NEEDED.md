# Trạng thái reproduce

**Đã đủ; không còn run thiếu trong matrix đã publish.**

- Bảng reproduce MNIST/CIFAR: full matrix gồm 400 Deep SVDD fixed-final runs và 30 DROCC fixed-final runs; audit độc lập replay 4,600,000 prediction rows.
- Bảng so sánh MVTec: 30 RBF SVDD, 30 kết quả Deep SVDD và 15 DROCC; đủ 15 categories, seed 0.
- PatchCore và NBD dùng seed 0 trên cùng 1,725 ảnh test; không trộn mean/SD của nhiều seed vào bảng này.
- DROCC test-selected chỉ giữ làm diagnostic; bảng publish dùng fixed final epoch.

Chi tiết MVTec full schedule: [MVTEC_IMAGE_FULL.md](MVTEC_IMAGE_FULL.md).
Kết quả native: [REPORT.md](../../results/native-full-4090-20260915/REPORT.md).

`runs_needed.csv` hiện chỉ còn header vì không có run thiếu. Kiểm tra lại bằng:

```sh
python -m reproduction.pending_runs --output docs/reproduction/runs_needed.csv
```

Metrics/predictions/config/log đã có ở máy local. Checkpoint đầy đủ vẫn ở máy thuê đã dừng; xem `backup_receipt.json`. Chưa tải xong archive checkpoint về local, không được gọi đó là backup đầy đủ.
