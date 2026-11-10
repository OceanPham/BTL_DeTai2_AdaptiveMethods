"""
04 - SỬ DỤNG THƯ VIỆN: PyTorch (torch.optim) và scikit-learn

(1) Đối chiếu bản tự cài đặt (toi_uu.py) với torch.optim.Adagrad / RMSprop /
    Adadelta / Adam: chạy 50 bước trên f = x1^2 + 10*x2^2 và so sánh nghiệm.
(2) Huấn luyện mạng nơ-ron MLP 64-64-10 trên bộ digits bằng 5 bộ tối ưu của
    PyTorch với tham số mặc định (mini-batch 32, 30 epoch)        -> hinh6
(3) scikit-learn: MLPClassifier(solver="adam") so với solver="sgd".

Chạy:  python 04_pytorch_thu_vien.py
"""
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import toi_uu

HINH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hinh")
os.makedirs(HINH, exist_ok=True)

# ------------------------------------------------------------------ (1)
print("(1) ĐỐI CHIẾU BẢN TỰ CÀI ĐẶT VỚI torch.optim (50 bước, x0 = (1, 1))")
cap = [
    ("AdaGrad", toi_uu.AdaGrad(lr=0.5), torch.optim.Adagrad, dict(lr=0.5)),
    ("RMSProp", toi_uu.RMSProp(lr=0.1, rho=0.9),
     torch.optim.RMSprop, dict(lr=0.1, alpha=0.9)),
    ("Adadelta", toi_uu.Adadelta(lr=1.0, rho=0.9),
     torch.optim.Adadelta, dict(lr=1.0, rho=0.9)),
    ("Adam", toi_uu.Adam(lr=0.1), torch.optim.Adam, dict(lr=0.1)),
]
for ten, cua_nhom, lop_torch, kw in cap:
    x = np.array([1.0, 1.0])
    for _ in range(50):
        x = cua_nhom.buoc(x, np.array([2 * x[0], 20 * x[1]]))
    xt = torch.tensor([1.0, 1.0], dtype=torch.float64, requires_grad=True)
    opt = lop_torch([xt], **kw)
    for _ in range(50):
        opt.zero_grad()
        (xt[0] ** 2 + 10 * xt[1] ** 2).backward()
        opt.step()
    sai_khac = np.abs(x - xt.detach().numpy()).max()
    print(f"    {ten:<9} tự cài đặt: ({x[0]: .6f}, {x[1]: .6f})   "
          f"torch: ({xt[0].item(): .6f}, {xt[1].item(): .6f})   "
          f"sai khác lớn nhất = {sai_khac:.1e}")

# ------------------------------------------------------------------ (2)
X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0,
                                      stratify=y)
sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
Xtr_t = torch.tensor(Xtr_s, dtype=torch.float32)
Xte_t = torch.tensor(Xte_s, dtype=torch.float32)
ytr_t = torch.tensor(ytr, dtype=torch.long)
yte_t = torch.tensor(yte, dtype=torch.long)
tap_train = TensorDataset(Xtr_t, ytr_t)
ham_mat_mat = nn.CrossEntropyLoss()
EPOCHS = 30


def huan_luyen(tao_optimizer, seed=0):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
    optimizer = tao_optimizer(model.parameters())
    loader = DataLoader(tap_train, batch_size=32, shuffle=True)
    ls_loss, ls_acc = [], []
    t0 = time.perf_counter()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = ham_mat_mat(model(xb), yb)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            ls_loss.append(ham_mat_mat(model(Xtr_t), ytr_t).item())
            ls_acc.append((model(Xte_t).argmax(1) == yte_t).float().mean().item())
    return ls_loss, ls_acc, time.perf_counter() - t0


cau_hinh = [
    ("SGD", "lr = 0.01", lambda p: torch.optim.SGD(p, lr=0.01)),
    ("AdaGrad", "lr = 0.01", lambda p: torch.optim.Adagrad(p, lr=0.01)),
    ("RMSProp", "lr = 0.001", lambda p: torch.optim.RMSprop(p, lr=0.001, alpha=0.9)),
    ("Adadelta", "lr = 1.0", lambda p: torch.optim.Adadelta(p, lr=1.0, rho=0.9)),
    ("Adam", "lr = 0.001", lambda p: torch.optim.Adam(p, lr=0.001)),
]
print(f"\n(2) MLP 64-64-10 TRÊN BỘ DIGITS ({len(Xtr)} mẫu huấn luyện, "
      f"{len(Xte)} mẫu kiểm tra, mini-batch 32, {EPOCHS} epoch)")
print(f"    {'Bộ tối ưu':<10}{'Tham số':<12}{'loss 1 epoch':>14}{'loss 5 epoch':>14}"
      f"{'loss 30 epoch':>15}{'acc test':>10}{'t (s)':>8}")
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
for ten, ts, tao in cau_hinh:
    ls_loss, ls_acc, tg = huan_luyen(tao)
    print(f"    {ten:<10}{ts:<12}{ls_loss[0]:>14.4f}{ls_loss[4]:>14.4f}"
          f"{ls_loss[-1]:>15.5f}{ls_acc[-1]:>10.4f}{tg:>8.2f}")
    ep = range(1, EPOCHS + 1)
    axes[0].semilogy(ep, ls_loss, color=toi_uu.MAU[ten], lw=1.8,
                     label=f"{ten} ({ts})")
    axes[1].plot(ep, ls_acc, color=toi_uu.MAU[ten], lw=1.8, label=ten)
axes[0].set(xlabel="Epoch", ylabel="Cross-entropy huấn luyện (log)",
            title="Hàm mất mát")
axes[1].set(xlabel="Epoch", ylabel="Độ chính xác trên tập kiểm tra",
            title="Độ chính xác", ylim=(0.85, 1.0))
for ax in axes:
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
fig.suptitle("Huấn luyện MLP 64-64-10 bằng các bộ tối ưu của torch.optim",
             fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(HINH, "hinh6_pytorch_mlp.png"), dpi=160)
plt.close(fig)

# ------------------------------------------------------------------ (3)
print("\n(3) scikit-learn: MLPClassifier, mạng 64-64-10, 30 epoch")
for solver, kw in [("adam", dict(learning_rate_init=0.001)),
                   ("sgd", dict(learning_rate_init=0.01, momentum=0.0))]:
    mlp = MLPClassifier(hidden_layer_sizes=(64,), solver=solver, batch_size=32,
                        alpha=0.0, max_iter=30, n_iter_no_change=1000,
                        random_state=0, **kw)
    mlp.fit(Xtr_s, ytr)
    print(f"    solver = {solver:<5}: loss = {mlp.loss_curve_[-1]:.5f}, "
          f"accuracy test = {mlp.score(Xte_s, yte):.4f}")
print("\nĐã lưu: hinh6_pytorch_mlp.png")
