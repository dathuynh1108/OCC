# RIFT — Role-conditioned Interventional Flow Tensor

Pilot cho logical anomaly: một quan hệ nguồn–đích có được duy trì qua đúng loại
thành phần trung gian hay không? Đây là một **giả thuyết descriptor**, chưa phải
phương pháp đã chứng minh contribution hay kết quả SOTA. Không thay đổi code
bubble/PatchScore và các kết quả OCC cũ.

## Phần đang thử chế

Diffusion transfer thường hỏi “nguồn truyền tới đích thế nào?”. RIFT hỏi thêm:
“quan hệ đó thay đổi thế nào khi làm yếu từng cạnh, và cạnh đó thuộc role trung
gian nào?”. Giữ tensor **nguồn × đích × mediator × thang diffusion**, thay vì
pooling mất thông tin mediator.

Dùng resolvent diffusion, không phải DDPM và không phải canonical diffusion-map
coordinates. Với graph vô hướng có conductance không âm:

    L = diag(W 1) - W
    R_t = (I + t L)^(-1) = integral_0^infinity exp(-s) exp(-stL) ds
    Q = R_t B; K_t = B^T Q

B có các cột anchor được chuẩn hóa tổng bằng 1. Khi làm yếu riêng cạnh e=(u,v)
bằng w_e'=(1-eta)w_e, giữ B cố định:

    v_e = Q[u] - Q[v]
    r_e = R[u,u] + R[v,v] - 2 R[u,v]
    Delta K_e = eta*t*w_e/(1-eta*t*w_e*r_e) * v_e v_e^T
    M[g,:,:] = sum_e 0.5*(C[u,g]+C[v,g]) * Delta K_e

C là soft mediator memberships, tổng theo role của mỗi node bằng 1. Đây là
Sherman–Morrison cho can thiệp một cạnh: không cần giải lại hệ cho mọi cạnh.
**M là tổng các can thiệp đơn cạnh độc lập, không phải can thiệp đồng thời lên
cả vùng.** Các thành phần ngoài đường chéo có thể âm; không lấy abs trước pooling.

Đóng góp đang khảo sát là cách dùng finite-response tensor theo role để chấm
logical anomaly; công thức nghịch đảo hạng một không phải một phát minh mới.
Chưa thực hiện literature audit đủ để tuyên bố “first”.

## Ba mức thí nghiệm

1. `pilot.py`: graph tổng hợp có role đúng, kiểm tra cơ chế và ablation. Số ở đây
   KHÔNG phải image AD performance.
2. `run.py --root ...`: ảnh → frozen CNN → grid graph → role proxies học từ normal
   → RIFT. Không cần nhãn part hay mask anomaly. **Graph feature không phải graph
   dây dẫn vật lý**; role proxies không phải đầu nối đã được nhận diện đúng.
3. `run.py --graph-manifest ...`: nhận graph quan sát từ detector/annotation riêng.
   Dùng nhánh này để kiểm tra khi có part/endpoint thật. Graph phải xây từ quan sát,
   không được xây bằng cách áp sẵn quy tắc normal.

Ở nhánh ảnh, anchors và mediator dùng cùng bộ role proxies. Ở probe graph,
anchors là bốn đầu mối, mediator có cả đầu mối, vùng buffer và vùng guide.

## Cài đặt

Chạy từ **repo root**. Python 3.12/3.13; môi trường kiểm tra ở `evidence/`.
Giữ bản CUDA `torch`/`torchvision` đang chạy tốt trên máy; file requirements không
thay cặp này bằng một wheel CPU khác.

```bash
python -m pip install -r research/rift/requirements.txt
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
python -m pytest research/rift/test_rift.py -q
```

Package chạy bằng `python -m research.rift...` từ repo root; không cần sửa
`pyproject.toml` hiện tại. Pytest mặc định của OCC chỉ tìm trong `tests/`, vì vậy
phải chỉ rõ đường dẫn test mới như trên.

## Probe CPU có thể tái lập

```bash
python -m research.rift.pilot \
  --out research/rift/outputs/graph_probe \
  --seeds 0 1 2 --noises 0 0.35 --train 32 --cal 64 --test 64
```

Hai loại lỗi: đổi đầu ra được nối (`connection`), hoặc đổi role của guide trong
khi giữ kết nối đầu–cuối (`routing`). Noise là biến thiên conductance log-normal
độc lập với nhãn, không phải lỗi detector đã được hiệu chuẩn. Test còn có normal
với conductance nhân 1.7: đây là **stress test**, không được trộn vào AUROC chính
rồi giấu đi các báo lỗi nhầm. Seed/split độc lập; normal và anomaly test được ghép
cặp theo cùng nuisance để tránh shortcut.

## Chạy GPU trên MVTec LOCO

Tải và giải nén dataset từ nguồn chính thức, tuân thủ giấy phép. Không dùng MVTec
AD thông thường thay LOCO. Code không tự tải hoặc phân phối lại ảnh.

```text
/data/mvtec_loco_anomaly_detection/
  splicing_connectors/
    train/good/*.png
    validation/good/*.png
    test/good/*.png
    test/logical_anomalies/*.png
    test/structural_anomalies/*.png
```

Lượt đầu, một category và một seed:

```bash
python -m research.rift.run \
  --root /data/mvtec_loco_anomaly_detection \
  --category splicing_connectors --seeds 0 \
  --config research/rift/configs/loco_resnet18.json \
  --device cuda --cache research/rift/cache \
  --out research/rift/outputs/loco_r18_v1
```

Lượt rộng hơn, WRN-50-2 và ba seed:

```bash
python -m research.rift.run \
  --root /data/mvtec_loco_anomaly_detection \
  --category splicing_connectors pushpins screw_bag breakfast_box juice_bottle \
  --seeds 0 1 2 --config research/rift/configs/loco_wrn50.json \
  --device cuda --cache research/rift/cache \
  --out research/rift/outputs/loco_wrn_v1
```

CNN dùng `IMAGENET1K_V1`, layer2+layer3, fixed random projection có seed cố định.
Không có train epoch: GPU dùng cho frozen feature extraction và graph solves.
Lượt đầu torchvision có thể tải weights. Máy offline: thêm `weights_path` vào
JSON, trỏ tới state_dict tương thích với backbone. Không có fallback sang random
weights. CUDA không có thì báo lỗi; dùng `--device cpu` một cách tường minh.

Ảnh được resize toàn bộ về hình vuông, không center crop. Phép resize thay đổi
aspect ratio. Graph chỉ nối 4-neighbor, bandwidth fit trên normal; không nối tắt
qua feature-kNN giữa các vùng xa. Grid vẫn có thể bỏ mất dây mảnh hoặc biến một
khe hở thành cạnh mạnh. Kiểm tra role/graph trước khi quy kết kết quả cho diffusion.

## Baseline và ablation

| Tên | Thông tin sử dụng |
|---|---|
| `occupancy` | Khối lượng anchor/role |
| `appearance` | Mean/std frozen features; chỉ có ở nhánh ảnh |
| `direct` | One-hop role-pair adjacency + occupancy |
| `shortest` | Weighted shortest-path distances pooled theo tất cả role |
| `transfer` | Diffusion giữa anchors + occupancy |
| `allrole_transfer` | Diffusion có quyền truy cập cả các mediator role |
| `rift` | Endpoint transfer + finite-response tensor theo mediator |
| `rift_linear` | Thay finite response bằng xấp xỉ vi phân |
| `rift_no_mediator` | Cộng mất danh tính mediator |
| `rift_no_identity` | Spectra/sorted occupancy, bỏ tên role |

Tất cả dùng cùng graph/role và cùng normal-only scaling + 1-NN scorer. Baseline
all-role và shortest không bị cố tình hạn chế đầu vào. `pca_dim` trong JSON cho
phép một ablation cùng giới hạn chiều (0 = không PCA); PCA chỉ fit normal, nhưng
có thể xóa hướng anomaly ngoài span normal, cần báo cáo riêng, không chọn theo test.

## Output và protocol

`summary.csv` ở output root: AUROC mean/sample SD qua seed, theo category/method/
group. Mỗi `category/seed_N/` có:

- `metrics.csv`: overall, logical và structural tách riêng; AUROC, AP, FPR/TPR
  tại mức calibration 5%/10%. Không chọn threshold bằng anomaly test.
- `scores.csv`: raw scores, normal-calibrated p-value, nearest normal ID.
- `manifest.json`, `run.json`: split, SHA-256 ảnh, cấu hình, phiên bản, source hashes.
- `role_graph.npz`, `descriptors.npz`, `memory_*.npz`: trạng thái fit và dữ liệu
  đủ để kiểm toán lại việc scoring, không chứa ảnh gốc.
- `audit.jsonl`: ba source/target/mediator proxy triples lệch mạnh so với normal
  gần nhất. Đây là attribution của descriptor, chưa phải giải thích pixel được
  xác nhận. Khi dùng PCA, không gán ý nghĩa role cho các chiều PCA.

Chỉ `train/good` fit roles, bandwidth, scaler, memory, PCA. `validation/good` dùng
calibration; nếu không có validation, tách 20% normal train bằng seed. Không dùng
mask ground truth. Phát hiện duplicate ảnh bằng SHA-256 và dừng, không tự bỏ mẫu.
P-value tối thiểu là 1/(n_cal+1); calibration quá nhỏ sẽ cảnh báo. Coverage không
được bảo đảm dưới distribution shift. Cache bám theo nội dung ảnh, weights,
preprocessing, phiên bản và source extractor. Cache dùng lại được, output run
không ghi đè: khi chạy lại hãy dùng tên output mới.

## Graph quan sát từ pipeline riêng

```json
{"id":"fit_001","graph":"graphs/fit_001.npz","split":"fit","kind":"good","label":0}
{"id":"cal_001","graph":"graphs/cal_001.npz","split":"cal","kind":"good","label":0}
{"id":"test_001","graph":"graphs/test_001.npz","split":"test","kind":"routing","label":1}
```

JSONL cần >=2 mẫu ở mỗi split. Mỗi NPZ có `w [N,N]`, `anchors [N,R]`, `roles [N,G]`.
W đối xứng, không âm, diagonal=0; roles có tổng hàng=1. Giữ ý nghĩa các cột R/G
nhất quán giữa mẫu; N có thể khác. Zero-mass role được giữ riêng bằng occupancy.
Train/calibration không được chứa anomaly. Các graph giống hệt có thể hợp lệ
nếu được trích từ những ảnh khác nhau; cần kiểm toán provenance phía upstream.

```bash
python -m research.rift.run --graph-manifest /data/graphs/manifest.jsonl \
  --out research/rift/outputs/observed_graphs --device cuda
```

## Những gì đã và chưa chứng minh

Xem `evidence/VERIFICATION.md` và `evidence/synthetic_summary.csv`.
Probe đã chạy trên CPU; các test toán, parser, cache, RGB image plumbing và
ResNet forward không tải weights đã chạy. **Chưa chạy pretrained CNN trên LOCO,
chưa chạy CUDA, chưa có kết quả image benchmark.**

Routing probe có AUROC 1.0 cho RIFT, nhưng all-role diffusion và shortest path
cũng 1.0. Vì vậy chưa có bằng chứng tensor mới hơn baseline về hiệu quả. Với noise
0.35, stress đổi conductance hợp lệ gây FPR khoảng 97.4% cho RIFT. Không giấu lỗi
này bằng chọn lại time/threshold trên test. Normalization scale là hướng sửa cần
đối chứng tương ứng cho mọi baseline, chưa được tuyên bố là giải pháp ở pilot này.

Tiêu chí tiếp tục: RIFT phải thêm giá trị so với all-role diffusion/shortest trên
ảnh thật hoặc graph quan sát nhiễu, và không tăng false positive ở normal hợp lệ.
Nếu không, dừng descriptor này thay vì cộng thêm nhiều score cho đẹp leaderboard.

## Nguồn

- MVTec LOCO và giấy phép: https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad
- Bergmann et al., *Beyond Dents and Scratches*, IJCV 2022,
  DOI: 10.1007/s11263-022-01578-9.
- ResNet-18 weights/API: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
- Cholesky solve: https://docs.pytorch.org/docs/stable/generated/torch.linalg.cholesky.html
