"""
02 - SO SÁNH CÁC THUẬT TOÁN TRÊN HÀM HAI BIẾN

Thí nghiệm 1: hàm toàn phương điều kiện xấu  f = x1^2 + 10*x2^2,  x0 = (-4, 2)
              -> quỹ đạo nghiệm và tốc độ hội tụ (hinh1), bước hiệu dụng (hinh2).
Thí nghiệm 2: hàm Rosenbrock (không lồi)  f = (1 - x1)^2 + 100*(x2 - x1^2)^2,
              x0 = (-1.5, 2), nghiệm x* = (1, 1)  (hinh3).

Chạy:  python 02_ham_hai_bien.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from toi_uu import MAU, SGD, AdaGrad, Adadelta, Adam, RMSProp, toi_uu_hoa

HINH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hinh")
os.makedirs(HINH, exist_ok=True)


# =========================================================== Thí nghiệm 1
def f1(x):
    return x[..., 0] ** 2 + 10 * x[..., 1] ** 2


def g1(x):
    return np.array([2 * x[0], 20 * x[1]])


x0 = (-4.0, 2.0)
SO_BUOC = 150
cau_hinh = [SGD(lr=0.09), AdaGrad(lr=1.0), RMSProp(lr=0.1),
            Adadelta(), Adam(lr=0.3)]
tham_so = ["η = 0.09", "η = 1.0", "η = 0.1, ρ = 0.9", "ρ = 0.9, ε = 1e-6",
           "η = 0.3, β₁ = 0.9, β₂ = 0.999"]

print("THÍ NGHIỆM 1: f = x1^2 + 10*x2^2, x0 = (-4, 2), f(x0) = 56")
print(f"{'Thuật toán':<10}{'Tham số':<32}{'f sau 20':>12}{'f sau 50':>12}"
      f"{'f sau 150':>12}{'bước để f<1e-6':>16}")
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
a, b = np.meshgrid(np.linspace(-4.6, 4.6, 300), np.linspace(-2.6, 2.6, 300))
axes[0].contour(a, b, a ** 2 + 10 * b ** 2,
                levels=[0.5, 2, 6, 12, 22, 36, 56, 80], colors="gray",
                linewidths=0.5)
luu = {}
for opt, ts in zip(cau_hinh, tham_so):
    qd = toi_uu_hoa(g1, x0, opt, SO_BUOC)
    fx = f1(qd)
    luu[opt.ten] = qd
    dat = np.where(fx < 1e-6)[0]
    print(f"{opt.ten:<10}{ts:<32}{fx[20]:>12.3e}{fx[50]:>12.3e}{fx[150]:>12.3e}"
          f"{(str(dat[0]) if len(dat) else '> 150'):>16}")
    axes[0].plot(qd[:60, 0], qd[:60, 1], "-", color=MAU[opt.ten], lw=1.3,
                 marker="o", ms=2.5, label=opt.ten)
    axes[1].semilogy(np.maximum(fx, 1e-16), color=MAU[opt.ten], lw=1.6,
                     label=f"{opt.ten} ({ts})")
axes[0].plot(0, 0, "k*", ms=12)
axes[0].set(xlabel="$x_1$", ylabel="$x_2$", title="Quỹ đạo nghiệm (60 bước đầu)")
axes[0].legend(loc="lower right", fontsize=9)
axes[1].set(xlabel="Bước lặp k", ylabel="$f(x_k)$ (thang log)",
            title="Giá trị hàm mục tiêu", ylim=(1e-16, 1e3))
axes[1].grid(alpha=0.3)
axes[1].legend(fontsize=8, loc="upper right")
fig.suptitle("Hàm toàn phương điều kiện xấu $f = x_1^2 + 10x_2^2$",
             fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(HINH, "hinh1_ham_toan_phuong.png"), dpi=160)
plt.close(fig)

# --- Hình 2: bước hiệu dụng theo từng tọa độ (tính lại để lấy trạng thái)
print("\nBước hiệu dụng |Δx_i| / |g_i| (tốc độ học thực tế của từng tọa độ):")
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True)
for opt_cls, kw in [(AdaGrad, dict(lr=1.0)), (RMSProp, dict(lr=0.1)),
                    (Adadelta, {}), (Adam, dict(lr=0.3))]:
    opt = opt_cls(**kw)
    x = np.array(x0)
    hd = []
    for _ in range(60):
        g = g1(x)
        x_moi = opt.buoc(x, g)
        hd.append(np.abs(x_moi - x) / np.maximum(np.abs(g), 1e-300))
        x = x_moi
    hd = np.array(hd)
    print(f"  {opt.ten:<9} bước 1: {hd[0]},  bước 10: {hd[9]},  bước 50: {hd[49]}")
    for i in range(2):
        axes[i].semilogy(range(1, 61), hd[:, i], color=MAU[opt.ten], lw=1.6,
                         label=opt.ten)
for i in range(2):
    axes[i].axhline(0.09, color=MAU["SGD"], ls="--", lw=1.2,
                    label="SGD (η = 0.09, cố định)")
    axes[i].set(xlabel="Bước lặp k",
                title=f"Tọa độ $x_{i + 1}$ (độ cong {'nhỏ: 2' if i == 0 else 'lớn: 20'})")
    axes[i].grid(alpha=0.3, which="both")
axes[0].set_ylabel("Bước hiệu dụng $|Δx_i| / |g_i|$ (log)")
axes[1].legend(fontsize=9)
fig.suptitle("Tốc độ học hiệu dụng của từng tọa độ", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(HINH, "hinh2_buoc_hieu_dung.png"), dpi=160)
plt.close(fig)


# =========================================================== Thí nghiệm 2
def f2(x):
    return (1 - x[..., 0]) ** 2 + 100 * (x[..., 1] - x[..., 0] ** 2) ** 2


def g2(x):
    return np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
                     200 * (x[1] - x[0] ** 2)])


x0 = (-1.5, 2.0)
SO_BUOC = 3000
cau_hinh = [SGD(lr=0.001), AdaGrad(lr=0.5), RMSProp(lr=0.005),
            Adadelta(), Adam(lr=0.05)]
tham_so = ["η = 0.001", "η = 0.5", "η = 0.005", "ρ = 0.9", "η = 0.05"]
print("\nTHÍ NGHIỆM 2: hàm Rosenbrock, x0 = (-1.5, 2), x* = (1, 1)")
print(f"{'Thuật toán':<10}{'Tham số':<12}{'f sau 500':>12}{'f sau 3000':>12}"
      f"{'x sau 3000 bước':>26}")
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
a, b = np.meshgrid(np.linspace(-2, 2, 400), np.linspace(-0.6, 3, 400))
axes[0].contour(a, b, (1 - a) ** 2 + 100 * (b - a ** 2) ** 2,
                levels=np.geomspace(0.3, 2500, 12), colors="gray",
                linewidths=0.5)
for opt, ts in zip(cau_hinh, tham_so):
    qd = toi_uu_hoa(g2, x0, opt, SO_BUOC)
    fx = f2(qd)
    print(f"{opt.ten:<10}{ts:<12}{fx[500]:>12.3e}{fx[-1]:>12.3e}"
          f"{'(' + f'{qd[-1, 0]:.4f}, {qd[-1, 1]:.4f}' + ')':>26}")
    axes[0].plot(qd[:, 0], qd[:, 1], "-", color=MAU[opt.ten], lw=1.2,
                 label=opt.ten)
    axes[1].semilogy(np.maximum(fx, 1e-16), color=MAU[opt.ten], lw=1.5,
                     label=f"{opt.ten} ({ts})")
axes[0].plot(*x0, "ks", ms=6)
axes[0].plot(1, 1, "k*", ms=12)
axes[0].set(xlabel="$x_1$", ylabel="$x_2$", xlim=(-2, 2), ylim=(-0.6, 3),
            title="Quỹ đạo nghiệm (■ xuất phát, ★ nghiệm)")
axes[0].legend(loc="lower left", fontsize=9)
axes[1].set(xlabel="Bước lặp k", ylabel="$f(x_k)$ (thang log)",
            title="Giá trị hàm mục tiêu")
axes[1].grid(alpha=0.3)
axes[1].legend(fontsize=8)
fig.suptitle("Hàm Rosenbrock (không lồi, thung lũng hẹp và cong)",
             fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(HINH, "hinh3_rosenbrock.png"), dpi=160)
plt.close(fig)
print("\nĐã lưu: hinh1_ham_toan_phuong.png, hinh2_buoc_hieu_dung.png, "
      "hinh3_rosenbrock.png")
