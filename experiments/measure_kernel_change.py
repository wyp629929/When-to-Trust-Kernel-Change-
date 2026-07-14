"""
实验：训练前后核的变化 —— 纯测量，无公式假设

训练一个神经网络，测量：
1. 初始特征核 vs 训练后特征核
2. 两者的谱（特征值）变化
3. 两者的特征向量变化（子空间对齐）
4. 任务能量 v_i 与哪些变化相关
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.linalg import eigh, subspace_angles
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


# ============================================================
# 1. 模型
# ============================================================

class TwoLayerNet(nn.Module):
    def __init__(self, d_in, width):
        super().__init__()
        self.width = width
        self.fc1 = nn.Linear(d_in, width, bias=False)
        self.fc2 = nn.Linear(width, 1, bias=False)
        nn.init.normal_(self.fc1.weight, std=np.sqrt(2.0 / d_in))
        nn.init.normal_(self.fc2.weight, std=np.sqrt(2.0 / width))

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc2(h).flatten()

    def get_features(self, x):
        with torch.no_grad():
            x = torch.FloatTensor(x).to(DEVICE)
            h = torch.relu(self.fc1(x))
            return h.cpu().numpy()


# ============================================================
# 2. 数据
# ============================================================

def sample_sphere(n, d):
    x = np.random.randn(n, d)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def ntk_kernel(X, Y=None):
    if Y is None:
        Y = X
    dot = X @ Y.T
    cos_theta = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return (np.sin(theta) + (np.pi - theta) * cos_theta) / np.pi + ((np.pi - theta) / np.pi) * dot


def make_target(K, spectral_decay=1.0):
    """从核谱构造目标"""
    rng = np.random.RandomState(SEED)
    eigvals, eigvecs = eigh(K)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    w = (eigvals ** (spectral_decay / 2)) * rng.randn(len(eigvals))
    y = eigvecs @ w
    return y / np.std(y), eigvals, eigvecs


# ============================================================
# 3. 核分解工具
# ============================================================

def decompose_kernel(K):
    """特征分解核矩阵"""
    eigvals, eigvecs = eigh(K)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    return eigvals, eigvecs


def compute_v_i(eigvecs, y):
    """计算任务在每个特征方向上的能量"""
    proj = eigvecs.T @ y
    n = len(y)
    return (proj ** 2) / n


def compute_cka(K, L):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K_c = H @ K @ H
    L_c = H @ L @ H
    return np.sum(K_c * L_c) / (np.sqrt(np.sum(K_c ** 2) * np.sum(L_c ** 2)) + 1e-12)


def align_eigenvectors(V1, V2, k=20):
    """
    测量两个特征向量集的前 k 个的对齐程度。
    返回每个子空间对之间的余弦相似度矩阵。
    """
    k = min(k, V1.shape[1], V2.shape[1])
    U1 = V1[:, :k]
    U2 = V2[:, :k]
    # 典型相关：SVD(U1^T U2)
    cross = U1.T @ U2
    s = np.linalg.svd(cross, compute_uv=False)
    return s  # 典型相关系数（越接近1越对齐）


# ============================================================
# 4. 主实验
# ============================================================

def experiment(d=10, n=200, width=256, spectral_decay=1.0,
               noise=0.1, lr=0.1, epochs=500):
    """训练网络，测量训练前后核的变化"""

    # ---------- 生成数据 ----------
    X = sample_sphere(n, d)
    K_exact = ntk_kernel(X)
    y_clean, eigvals_target, eigvecs_target = make_target(K_exact, spectral_decay)
    y = y_clean + noise * np.random.randn(n)

    # ---------- 初始特征核 ----------
    model_init = TwoLayerNet(d, width).to(DEVICE)
    H_init = model_init.get_features(X)
    K_init = (H_init @ H_init.T) / width

    # ---------- 训练 ----------
    model = TwoLayerNet(d, width).to(DEVICE)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    X_t = torch.FloatTensor(X).to(DEVICE)
    y_t = torch.FloatTensor(y).to(DEVICE)
    for ep in range(epochs):
        opt.zero_grad()
        loss = nn.MSELoss()(model(X_t), y_t)
        loss.backward()
        opt.step()

    # ---------- 训练后特征核 ----------
    H_post = model.get_features(X)
    K_post = (H_post @ H_post.T) / width

    # ---------- 分解 ----------
    eigv_init, evec_init = decompose_kernel(K_init)
    eigv_post, evec_post = decompose_kernel(K_post)
    eigv_exact, evec_exact = decompose_kernel(K_exact)

    v_i = compute_v_i(evec_exact, y_clean)

    # ---------- 度量 ----------
    cka_i2p = compute_cka(K_init, K_post)
    cka_i2e = compute_cka(K_init, K_exact)
    cka_p2e = compute_cka(K_post, K_exact)

    # 特征向量对齐（前 k 个）
    cca_sv = align_eigenvectors(evec_init, evec_post, k=10)

    # 特征值变化比率
    eig_ratio = eigv_post[:20] / (eigv_init[:20] + 1e-12)

    # 核对齐度（与标签）
    L = np.outer(y_clean, y_clean)
    cka_wrt_label_init = compute_cka(K_init, L)
    cka_wrt_label_post = compute_cka(K_post, L)

    return {
        'X': X, 'y_clean': y_clean,
        'K_init': K_init, 'K_post': K_post, 'K_exact': K_exact,
        'eigv_init': eigv_init, 'eigv_post': eigv_post, 'eigv_exact': eigv_exact,
        'evec_init': evec_init, 'evec_post': evec_post, 'evec_exact': evec_exact,
        'v_i': v_i,
        'cka_init_post': cka_i2p,
        'cka_init_exact': cka_i2e,
        'cka_post_exact': cka_p2e,
        'cca_sv': cca_sv,
        'eig_ratio_top': eig_ratio,
        'cka_label_init': cka_wrt_label_init,
        'cka_label_post': cka_wrt_label_post,
    }


# ============================================================
# 5. 分析 + 可视化
# ============================================================

def plot_analysis():
    print("=== 训练前后核的变化 ===\n")

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # ---------- 条件 A：高频目标 (hard) ----------
    print("--- 条件 A: spectral_decay=0.5 (高频目标, hard) ---")
    A = experiment(spectral_decay=0.5, noise=0.05, epochs=500)

    # ---------- 条件 B：低频目标 (easy) ----------
    print("--- 条件 B: spectral_decay=2.0 (低频目标, easy) ---")
    B = experiment(spectral_decay=2.0, noise=0.05, epochs=500)

    for label, cond, col in [('A: Hard (decay=0.5)', A, 0), ('B: Easy (decay=2.0)', B, 1)]:

        # 图1：特征值对比（前30个）
        ax = axes[0, col]
        idx = np.arange(1, 31)
        ax.plot(idx, cond['eigv_init'][:30], 'o-', label='Init Feat', color='C0', alpha=0.7)
        ax.plot(idx, cond['eigv_post'][:30], 's-', label='Post Feat', color='C1', alpha=0.7)
        ax.plot(idx, cond['eigv_exact'][:30], '^--', label='∞NTK', color='gray', alpha=0.5)
        ax.set_yscale('log')
        ax.set_xlabel('Index')
        ax.set_ylabel('Eigenvalue')
        ax.set_title(f'{label}: Eigenvalue Spectrum')
        ax.legend(fontsize=8)

        # 图2：特征值比率 post/init
        ax = axes[1, col]
        idx = np.arange(1, 31)
        ratio = cond['eigv_post'][:30] / (cond['eigv_init'][:30] + 1e-12)
        ax.bar(idx, ratio, color='C2', alpha=0.7)
        ax.axhline(1.0, color='gray', linestyle='--')
        ax.set_xlabel('Index')
        ax.set_ylabel('λ_post / λ_init')
        ax.set_title(f'{label}: Eigenvalue Change Ratio')

        print(f"  {label}")
        print(f"    CKA(init,post) = {cond['cka_init_post']:.4f}")
        print(f"    CKA(init,exact)= {cond['cka_init_exact']:.4f}")
        print(f"    CKA(post,exact)= {cond['cka_post_exact']:.4f}")
        print(f"    CKA_label(init)= {cond['cka_label_init']:.4f}")
        print(f"    CKA_label(post)= {cond['cka_label_post']:.4f}")
        print(f"    Mean eig ratio  = {np.mean(ratio):.4f}")
        print(f"    Top 10 eig ratio: {[f'{x:.2f}' for x in ratio[:10]]}")
        print(f"    CCA top-10 mean = {np.mean(cond['cca_sv']):.4f}")
        print()

    # 图3：特征向量旋转（子空间对齐）
    ax = axes[0, 2]
    for cond, label, color in [(A, 'Hard', 'C0'), (B, 'Easy', 'C1')]:
        sv = cond['cca_sv']
        ax.plot(range(1, len(sv) + 1), sv, 'o-', color=color, label=label, alpha=0.7)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Subspace index k')
    ax.set_ylabel('Canonical correlation')
    ax.set_title('Eigenvector Rotation (init vs post)')
    ax.legend()
    ax.set_ylim(0, 1.05)

    # 图4：v_i 与特征值变化的关系
    ax = axes[1, 2]
    for cond, label, color in [(A, 'Hard', 'C0'), (B, 'Easy', 'C1')]:
        v_top = cond['v_i'][:30]
        ratio = cond['eigv_post'][:30] / (cond['eigv_init'][:30] + 1e-12)
        ax.scatter(v_top, ratio, c=color, label=label, alpha=0.7, s=30)
    ax.set_xlabel('v_i (task energy)')
    ax.set_ylabel('λ_post / λ_init')
    ax.set_title('v_i vs Eigenvalue Change')
    ax.legend()
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.3)

    # 图5-6：核矩阵热图
    for idx, (cond, label) in enumerate([(A, 'Hard'), (B, 'Easy')]):
        ax = axes[idx % 2, 3]
        # 显示核矩阵差异
        K_diff = cond['K_post'] - cond['K_init']
        im = ax.imshow(K_diff[:50, :50], cmap='RdBu', vmin=-0.1, vmax=0.1)
        ax.set_title(f'{label}: K_post - K_init')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig('/Users/wangyaoping/Desktop/ml_paper/figures/kernel_change_analysis.png', dpi=150)
    plt.savefig('/Users/wangyaoping/Desktop/ml_paper/figures/kernel_change_analysis.pdf')
    print("Figure saved.")
    plt.show()


if __name__ == '__main__':
    import os
    os.makedirs('/Users/wangyaoping/Desktop/ml_paper/figures', exist_ok=True)
    plot_analysis()
