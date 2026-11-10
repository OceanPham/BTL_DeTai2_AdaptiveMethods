# Code demo – Đề tài 2: AdaGrad, RMSProp, Adadelta, Adam

Môn: Tối ưu hóa ứng dụng (Cao học CNTT – Đại học Thủy lợi) – Nhóm 9

## Cài đặt

Yêu cầu Python 3.10 trở lên.

```bash
pip install -r requirements.txt
```

Script 01–03 không cần PyTorch (`pip install numpy matplotlib scikit-learn`).

## Các file

| File | Nội dung | Hình sinh ra (thư mục `../hinh`) |
|---|---|---|
| `toi_uu.py` | Tự cài đặt các lớp `SGD`, `AdaGrad`, `RMSProp`, `Adadelta`, `Adam` bằng NumPy (dùng chung cho các script) | – |
| `01_vi_du_tinh_tay.py` | In chi tiết 3 bước lặp đầu của từng thuật toán trên f = x₁² + 10x₂² | – |
| `02_ham_hai_bien.py` | So sánh trên hàm toàn phương điều kiện xấu và hàm Rosenbrock; tốc độ học hiệu dụng | `hinh1`, `hinh2`, `hinh3` |
| `03_hoc_may_numpy.py` | Hồi quy softmax trên bộ digits bằng bản tự cài đặt; độ nhạy với tốc độ học | `hinh4`, `hinh5` |
| `04_pytorch_thu_vien.py` | Đối chiếu với `torch.optim`; huấn luyện MLP bằng PyTorch; `MLPClassifier` của scikit-learn | `hinh6` |

## Cách chạy

```bash
python 01_vi_du_tinh_tay.py
python 02_ham_hai_bien.py
python 03_hoc_may_numpy.py
python 04_pytorch_thu_vien.py
```

Mỗi script chạy độc lập (cần `toi_uu.py` nằm cùng thư mục), in kết quả ra màn hình và lưu hình vào `../hinh`.
Dữ liệu dùng bộ digits có sẵn trong scikit-learn nên không cần tải từ Internet. Các seed ngẫu nhiên đã cố định.
