"""
Gap 1: 自适应核对齐理论 —— 实验 (v3)

改用真实神经网络 + 经验 NTK 来对比：
- 初始 NTK（懒训练）
- 训练后的经验 NTK（特征学习）

这样我们不需要猜测 λ̃ᵢ 的公式，而是直接测量。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.linalg import eigh
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ============================================================
# 1. 两层 ReLU 网络
# ============================================================

class TwoLayerNet(nn.Module):
    def __init__(self, d_in, width, d_out=1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, width, bias=False)
        self.fc2 = nn.Linear(width, d_out, bias=False)
        nn.init.normal_(self.fc1.weight, std=np.sqrt(2.0 / d_in))
        nn.init.normal_(self.fc2.weight, std=np.sqrt(2.0 / width))

    def forward(self, x, return_features=False):
        h = torch.relu(self.fc1(x))
        out = self.fc2(h)
        if return_features:
            return out, h
        return out


def compute_empirical_ntk(model, x, return_features=False):
    """通过 Jacobian 计算经验 NTK: K(xᵢ, xⱼ) = ∇f(xᵢ)ᵀ ∇f(xⱼ)"""
    model.eval()
    x = x.to(DEVICE)
    n = x.shape[0]
    params = [p for p in model.parameters() if p.requires_grad]
    jacobians = []
    for i in range(n):
        xi = x[i:i+1]
        xi.requires_grad_(True)
        fi = model(xi)
        grads = []
        for p in params:
            g = torch.autograd.grad(fi, p, create_graph=False, retain_graph=True, allow_unused=True)
            if g[0] is not None:
                grads.append(g[0].detach().cpu().numpy().flatten())
            else:
                grads.append(np.zeros(p.numel()))
        jac = np.concatenate(grads)
        jacobians.append(jac)
    jacobians = np.array(jacobians)
    K = jacobians @ jacobians.T
    return K


# ============================================================
# 2. 合成数据
# ============================================================

def sample_sphere(n, d):
    x = np.random.randn(n, d)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def ntk_kernel(X, Y=None):
    """两层 ReLU NTK 精确公式（无限宽极限）"""
    if Y is None:
        Y = X
    dot = X @ Y.T
    cos_theta = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    k_nngp = (np.sin(theta) + (np.pi - theta) * cos_theta) / np.pi
    k_dot = ((np.pi - theta) / np.pi) * dot
    return k_nngp + k_dot


def construct_target_from_kernel(K, spectral_decay=1.0, seed=SEED):
    """从核矩阵构造目标函数"""
    rng = np.random.RandomState(seed)
    eigvals, eigvecs = eigh(K)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    w = (eigvals ** (spectral_decay / 2)) * rng.randn(len(eigvals))
    y = eigvecs @ w
    y = y / np.std(y)
    return y, eigvals, (eigvecs.T @ y) ** 2 / len(y)


# ============================================================
# 3. 核对齐
# ============================================================

def compute_cka(K, L):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K_c = H @ K @ H
    L_c = H @ L @ H
    num = np.sum(K_c * L_c)
    den = np.sqrt(np.sum(K_c * K_c) * np.sum(L_c * L_c))
    return num / (den + 1e-12)


# ============================================================
# 4. Kernel Ridge Regression
# ============================================================

def kernel_ridge(K_train, y_train, K_test, lam=1e-3):
    n = len(y_train)
    alpha = np.linalg.solve(K_train + n * lam * np.eye(n), y_train)
    pred = K_test @ alpha
    return pred


# ============================================================
# 5. 单次实验（比较初始 NTK vs 训练后 NTK）
# ============================================================

def run_experiment(d=10, n_train=200, n_test=500, width=512,
                   spectral_decay=1.0, noise_level=0.1,
                   lam_ridge=1e-3, lr=0.01, epochs=200):
    """一次完整实验"""
    n_total = n_train + n_test

    # ---- 生成数据 ----
    X = sample_sphere(n_total, d)
    K_full = ntk_kernel(X)
    y_full, eigvals_all, v_i = construct_target_from_kernel(K_full, spectral_decay)
    X_train, X_test = X[:n_train], X[n_train:]
    y_clean_train = y_full[:n_train]
    y_clean_test = y_full[n_train:]
    y_train = y_clean_train + noise_level * np.random.randn(n_train)

    K_inf_train = K_full[:n_train, :n_train]
    K_inf_test = K_full[n_train:, :n_train]

    # ---- 无限宽 NTK baseline ----
    pred_inf = kernel_ridge(K_inf_train, y_train, K_inf_test, lam_ridge)
    err_inf = np.mean((pred_inf - y_clean_test) ** 2)

    # ---- 训练神经网络 ----
    model = TwoLayerNet(d, width).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    X_t = torch.FloatTensor(X_train).to(DEVICE)
    y_t = torch.FloatTensor(y_train).reshape(-1, 1).to(DEVICE)
    losses = []
    for ep in range(epochs):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = nn.MSELoss()(pred, y_t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if ep % 50 == 0:
            print(f"    epoch {ep}: loss = {loss.item():.6f}")

    # ---- 训练后的 NTK（经验NTK） ----
    print("  computing NTK after training...")
    K_train_post = compute_empirical_ntk(model, torch.FloatTensor(X_train))

    # 测试集 NTK
    K_test_post = np.zeros((n_test, n_train))
    model.eval()
    with torch.no_grad():
        jac_list = []
        for j in range(n_train):
            xj = torch.FloatTensor(X_train[j:j+1]).to(DEVICE)
            xj.requires_grad_(True)
            fj = model(xj)
            grads = []
            for p in model.parameters():
                g = torch.autograd.grad(fj, p, retain_graph=True, allow_unused=True, create_graph=False)
                if g[0] is not None:
                    grads.append(g[0].detach().cpu().numpy().flatten())
                else:
                    grads.append(np.zeros(p.numel()))
            jac_list.append(np.concatenate(grads))
        jac_train = np.array(jac_list)

        jac_test_list = []
        for i in range(n_test):
            xi = torch.FloatTensor(X_test[i:i+1]).to(DEVICE)
            xi.requires_grad_(True)
            fi = model(xi)
            grads = []
            for p in model.parameters():
                g = torch.autograd.grad(fi, p, retain_graph=True, allow_unused=True, create_graph=False)
                if g[0] is not None:
                    grads.append(g[0].detach().cpu().numpy().flatten())
                else:
                    grads.append(np.zeros(p.numel()))
            jac_test_list.append(np.concatenate(grads))
        jac_test = np.array(jac_test_list)

        K_train_post = jac_train @ jac_train.T
        K_test_post = jac_test @ jac_train.T

    pred_post = kernel_ridge(K_train_post, y_train, K_test_post, lam_ridge)
    err_post = np.mean((pred_post - y_clean_test) ** 2)

    # ---- 核对齐 ----
    y_label = np.outer(y_clean_train, y_clean_train)
    cka_inf = compute_cka(K_inf_train, y_label)
    cka_post = compute_cka(K_train_post, y_label)

    return {
        'err_inf': err_inf,
        'err_post': err_post,
        'err_gap': err_post - err_inf,
        'err_rel': (err_post - err_inf) / (err_inf + 1e-10),
        'cka_inf': cka_inf,
        'cka_post': cka_post,
        'delta_align': cka_post - cka_inf,
        'spectral_decay': spectral_decay,
    }


# ============================================================
# 6. 扫描实验
# ============================================================

def scan_spectral_decay():
    print("=== 扫描 spectral_decay ===")
    decays = [0.5, 0.8, 1.0, 1.5, 2.0]
    results = []
    for sd in decays:
        print(f"\n--- decay={sd} ---")
        r = run_experiment(spectral_decay=sd)
        r['x'] = sd
        results.append(r)
        print(f"  inf={r['err_inf']:.4f}  post={r['err_post']:.4f}  "
              f"gap={r['err_gap']:.4f}  rel={r['err_rel']*100:+.1f}%  "
              f"dAlign={r['delta_align']:.4f}")
    return results


if __name__ == '__main__':
    results = scan_spectral_decay()
    for r in results:
        print(f"decay={r['x']:.1f}: ΔGen={r['err_rel']*100:+.1f}%  ΔAlign={r['delta_align']:.4f}")
