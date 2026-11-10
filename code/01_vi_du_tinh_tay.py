"""
01 - VÍ DỤ TÍNH TAY (kiểm chứng các con số trình bày trong báo cáo)

Hàm mục tiêu:  f(x1, x2) = x1^2 + 10*x2^2,   grad f = (2*x1, 20*x2)
Điểm xuất phát x0 = (1, 1). Nghiệm tối ưu x* = (0, 0).
In chi tiết 3 bước lặp đầu của GD, AdaGrad, RMSProp, Adadelta và Adam.

Chạy:  python 01_vi_du_tinh_tay.py
"""
import numpy as np

np.set_printoptions(precision=5, suppress=True, floatmode="fixed")


def f(x):
    return x[0] ** 2 + 10 * x[1] ** 2


def grad(x):
    return np.array([2 * x[0], 20 * x[1]])


x0 = np.array([1.0, 1.0])
SO_BUOC = 3


def tieu_de(t):
    print("\n" + "=" * 70 + "\n" + t + "\n" + "=" * 70)


tieu_de("GD thường, eta = 0.09 (gần ngưỡng hội tụ 2/L = 0.1)")
x = x0.copy()
for k in range(1, SO_BUOC + 1):
    g = grad(x)
    x = x - 0.09 * g
    print(f"k={k}: g={g}  ->  x={x}, f={f(x):.5f}")

tieu_de("AdaGrad, eta = 0.5, eps = 1e-10")
x, G = x0.copy(), np.zeros(2)
for k in range(1, SO_BUOC + 1):
    g = grad(x)
    G += g * g
    buoc = 0.5 * g / (np.sqrt(G) + 1e-10)
    x = x - buoc
    print(f"k={k}: g={g}, G={G}, eta/sqrt(G)={0.5 / np.sqrt(G)}\n"
          f"      buoc={buoc}  ->  x={x}, f={f(x):.5f}")

tieu_de("RMSProp, eta = 0.1, rho = 0.9, eps = 1e-8")
x, v = x0.copy(), np.zeros(2)
for k in range(1, SO_BUOC + 1):
    g = grad(x)
    v = 0.9 * v + 0.1 * g * g
    buoc = 0.1 * g / (np.sqrt(v) + 1e-8)
    x = x - buoc
    print(f"k={k}: g={g}, v={v}\n"
          f"      buoc={buoc}  ->  x={x}, f={f(x):.5f}")

tieu_de("Adadelta, rho = 0.9, eps = 1e-6 (không có tốc độ học)")
x, v, u = x0.copy(), np.zeros(2), np.zeros(2)
for k in range(1, SO_BUOC + 1):
    g = grad(x)
    v = 0.9 * v + 0.1 * g * g
    d = -np.sqrt(u + 1e-6) / np.sqrt(v + 1e-6) * g
    u = 0.9 * u + 0.1 * d * d
    x = x + d
    print(f"k={k}: g={g}, v={v}\n"
          f"      d={d}, u={u}  ->  x={x}, f={f(x):.5f}")

tieu_de("Adam, eta = 0.1, beta1 = 0.9, beta2 = 0.999, eps = 1e-8")
x, m, v = x0.copy(), np.zeros(2), np.zeros(2)
for k in range(1, SO_BUOC + 1):
    g = grad(x)
    m = 0.9 * m + 0.1 * g
    v = 0.999 * v + 0.001 * g * g
    m_hat = m / (1 - 0.9 ** k)
    v_hat = v / (1 - 0.999 ** k)
    buoc = 0.1 * m_hat / (np.sqrt(v_hat) + 1e-8)
    x = x - buoc
    print(f"k={k}: g={g}, m={m}, v={v}\n"
          f"      m_hat={m_hat}, v_hat={v_hat}\n"
          f"      buoc={buoc}  ->  x={x}, f={f(x):.5f}")
