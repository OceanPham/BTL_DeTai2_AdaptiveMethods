"""
toi_uu.py - Tự cài đặt các bộ tối ưu bậc một bằng NumPy

Mỗi lớp có phương thức  buoc(theta, g) -> theta mới,
trong đó theta là véctơ tham số hiện tại, g là gradient tại theta.
Công thức và giá trị mặc định theo đúng quy ước của PyTorch (torch.optim)
để có thể đối chiếu trực tiếp (xem 04_pytorch_thu_vien.py).
"""
import numpy as np


class SGD:
    """Gradient descent thường: theta <- theta - lr * g  (dùng làm mốc so sánh)."""
    ten = "SGD"

    def __init__(self, lr=0.01):
        self.lr = lr

    def buoc(self, theta, g):
        return theta - self.lr * g


class AdaGrad:
    """Duchi, Hazan, Singer (2011).
    G <- G + g^2 ;  theta <- theta - lr * g / (sqrt(G) + eps)"""
    ten = "AdaGrad"

    def __init__(self, lr=0.01, eps=1e-10):
        self.lr, self.eps = lr, eps
        self.G = None                       # tổng tích lũy bình phương gradient

    def buoc(self, theta, g):
        if self.G is None:
            self.G = np.zeros_like(theta)
        self.G += g * g
        return theta - self.lr * g / (np.sqrt(self.G) + self.eps)


class RMSProp:
    """Tieleman & Hinton (2012).
    v <- rho * v + (1 - rho) * g^2 ;  theta <- theta - lr * g / (sqrt(v) + eps)"""
    ten = "RMSProp"

    def __init__(self, lr=0.01, rho=0.9, eps=1e-8):
        self.lr, self.rho, self.eps = lr, rho, eps
        self.v = None                       # trung bình trượt của g^2

    def buoc(self, theta, g):
        if self.v is None:
            self.v = np.zeros_like(theta)
        self.v = self.rho * self.v + (1 - self.rho) * g * g
        return theta - self.lr * g / (np.sqrt(self.v) + self.eps)


class Adadelta:
    """Zeiler (2012) - không cần chọn tốc độ học.
    v <- rho * v + (1 - rho) * g^2
    d <- - sqrt(u + eps) / sqrt(v + eps) * g
    u <- rho * u + (1 - rho) * d^2
    theta <- theta + lr * d            (lr = 1 trong bài báo gốc)"""
    ten = "Adadelta"

    def __init__(self, lr=1.0, rho=0.9, eps=1e-6):
        self.lr, self.rho, self.eps = lr, rho, eps
        self.v = None                       # trung bình trượt của g^2
        self.u = None                       # trung bình trượt của (bước cập nhật)^2

    def buoc(self, theta, g):
        if self.v is None:
            self.v = np.zeros_like(theta)
            self.u = np.zeros_like(theta)
        self.v = self.rho * self.v + (1 - self.rho) * g * g
        d = -np.sqrt(self.u + self.eps) / np.sqrt(self.v + self.eps) * g
        self.u = self.rho * self.u + (1 - self.rho) * d * d
        return theta + self.lr * d


class Adam:
    """Kingma & Ba (2015).
    m <- b1 * m + (1 - b1) * g          (mô men bậc 1)
    v <- b2 * v + (1 - b2) * g^2        (mô men bậc 2)
    m_hat = m / (1 - b1^t) ;  v_hat = v / (1 - b2^t)      (hiệu chỉnh độ chệch)
    theta <- theta - lr * m_hat / (sqrt(v_hat) + eps)"""
    ten = "Adam"

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = None
        self.v = None
        self.t = 0

    def buoc(self, theta, g):
        if self.m is None:
            self.m = np.zeros_like(theta)
            self.v = np.zeros_like(theta)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        return theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# Màu dùng thống nhất trong mọi hình vẽ
MAU = {"SGD": "#7f7f7f", "AdaGrad": "#1f77b4", "RMSProp": "#2ca02c",
       "Adadelta": "#9467bd", "Adam": "#d62728"}


def toi_uu_hoa(ham_grad, x0, bo_toi_uu, so_buoc):
    """Chạy `so_buoc` bước lặp từ x0; trả về quỹ đạo (so_buoc + 1, d)."""
    x = np.array(x0, dtype=float)
    quy_dao = [x.copy()]
    for _ in range(so_buoc):
        x = bo_toi_uu.buoc(x, ham_grad(x))
        quy_dao.append(x.copy())
    return np.array(quy_dao)
