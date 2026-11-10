"""
03 - ÁP DỤNG TRONG HỌC MÁY (tự cài đặt bằng NumPy)

Bài toán: phân loại chữ số viết tay (bộ digits của scikit-learn: 1797 ảnh 8x8,
          10 lớp) bằng hồi quy softmax, hàm mất mát cross-entropy.
Dữ liệu KHÔNG chuẩn hóa (điểm ảnh 0..16; nhiều điểm ảnh ở viền gần như luôn
bằng 0) -> các tham số có gradient với độ lớn rất khác nhau, là tình huống mà
phương pháp tốc độ học thích nghi được thiết kế để xử lý.

Thí nghiệm A: so sánh SGD, AdaGrad, RMSProp, Adadelta, Adam với tham số mặc định
              của PyTorch (mini-batch 32, 30 epoch)                -> hinh4
Thí nghiệm B: độ nhạy với tốc độ học (mất mát sau 10 epoch theo η) -> hinh5

Chạy:  python 03_hoc_may_numpy.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from toi_uu import MAU, SGD, AdaGrad, Adadelta, Adam, RMSProp

HINH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hinh")
os.makedirs(HINH, exist_ok=True)

X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0,
                                      stratify=y)
N, d = Xtr.shape
K = 10
Ytr = np.eye(K)[ytr]                                  # mã hóa one-hot
print(f"Tập huấn luyện: {N} mẫu x {d} đặc trưng, tập kiểm tra: {len(Xte)} mẫu")
print(f"Độ lệch chuẩn của các đặc trưng: nhỏ nhất = {Xtr.std(0).min():.2f}, "
      f"lớn nhất = {Xtr.std(0).max():.2f}\n")


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def mat_mat(theta, Xa, Ya):
    W = theta.reshape(d + 1, K)
    P = softmax(Xa @ W[:-1] + W[-1])
    return -np.mean(np.sum(Ya * np.log(P + 1e-12), axis=1))


def gradient(theta, Xb, Yb):
    """Gradient cross-entropy của hồi quy softmax trên một lô."""
    W = theta.reshape(d + 1, K)
    P = softmax(Xb @ W[:-1] + W[-1])
    R = (P - Yb) / len(Xb)
    return np.vstack([Xb.T @ R, R.sum(axis=0)]).ravel()


def do_chinh_xac(theta, Xa, ya):
    W = theta.reshape(d + 1, K)
    return np.mean((Xa @ W[:-1] + W[-1]).argmax(1) == ya)


def huan_luyen(opt, epochs, batch=32, seed=0):
    rng = np.random.default_rng(seed)
    theta = np.zeros((d + 1) * K)
    ls = [mat_mat(theta, Xtr, Ytr)]
    for _ in range(epochs):
        cs = rng.permutation(N)
        for s in range(0, N, batch):
            lo = cs[s:s + batch]
            theta = opt.buoc(theta, gradient(theta, Xtr[lo], Ytr[lo]))
        ls.append(mat_mat(theta, Xtr, Ytr))
    return theta, np.array(ls)


# ======================================================== Thí nghiệm A
EPOCHS = 30
cau_hinh = [(SGD(lr=0.01), "η = 0.01"), (AdaGrad(lr=0.01), "η = 0.01"),
            (RMSProp(lr=0.001, rho=0.9), "η = 0.001, ρ = 0.9"),
            (Adadelta(), "ρ = 0.9, ε = 1e-6"), (Adam(lr=0.001), "η = 0.001")]
print("THÍ NGHIỆM A: hồi quy softmax, mini-batch 32, 30 epoch")
print(f"{'Thuật toán':<10}{'Tham số':<22}{'mất mát 1 epoch':>17}"
      f"{'mất mát 30 epoch':>18}{'acc train':>11}{'acc test':>10}")
fig, ax = plt.subplots(figsize=(7.6, 4.7))
for opt, ts in cau_hinh:
    with np.errstate(all="ignore"):
        theta, ls = huan_luyen(opt, EPOCHS)
    print(f"{opt.ten:<10}{ts:<22}{ls[1]:>17.4f}{ls[-1]:>18.4f}"
          f"{do_chinh_xac(theta, Xtr, ytr):>11.4f}"
          f"{do_chinh_xac(theta, Xte, yte):>10.4f}")
    ax.semilogy(ls, color=MAU[opt.ten], lw=1.8, label=f"{opt.ten} ({ts})")
ax.set(xlabel="Epoch", ylabel="Cross-entropy trên tập huấn luyện (log)",
       title="Hồi quy softmax trên bộ digits (dữ liệu chưa chuẩn hóa)")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(HINH, "hinh4_softmax_digits.png"), dpi=160)
plt.close(fig)

# ======================================================== Thí nghiệm B
print("\nTHÍ NGHIỆM B: mất mát huấn luyện sau 10 epoch theo tốc độ học")
cac_lr = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
print(f"{'η':>8}" + "".join(f"{t:>11}" for t in ["SGD", "AdaGrad", "RMSProp", "Adam"]))
kq = {t: [] for t in ["SGD", "AdaGrad", "RMSProp", "Adam"]}
for lr in cac_lr:
    dong = f"{lr:>8g}"
    for cls in [SGD, AdaGrad, RMSProp, Adam]:
        with np.errstate(all="ignore"):
            _, ls = huan_luyen(cls(lr=lr), 10)
        v = ls[-1] if np.isfinite(ls[-1]) else np.inf
        kq[cls.ten].append(v)
        dong += f"{v:>11.4f}"
    print(dong)
with np.errstate(all="ignore"):
    _, ls_ada = huan_luyen(Adadelta(), 10)
print(f"Adadelta (không có η): {ls_ada[-1]:.4f}")

fig, ax = plt.subplots(figsize=(7.6, 4.7))
for t, v in kq.items():
    v = np.minimum(np.array(v), 50.0)
    ax.loglog(cac_lr, v, "o-", color=MAU[t], lw=1.8, ms=5, label=t)
ax.axhline(ls_ada[-1], color=MAU["Adadelta"], ls="--", lw=1.6,
           label="Adadelta (không cần η)")
ax.set(xlabel="Tốc độ học η (thang log)",
       ylabel="Cross-entropy sau 10 epoch (log, chặn trên ở 50)",
       title="Độ nhạy của các thuật toán với tốc độ học")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(HINH, "hinh5_do_nhay_lr.png"), dpi=160)
plt.close(fig)
print("\nĐã lưu: hinh4_softmax_digits.png, hinh5_do_nhay_lr.png")
