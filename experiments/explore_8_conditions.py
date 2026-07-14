"""
实验：7种数据条件下，训练前后特征核的变化模式

无预设公式，纯测量。

变化维度：
1. 数据分布（均匀 / 高斯混合 / 同心球壳 / 稀疏）
2. 目标函数（多项式 / 径向 / 高频 / 阶梯 / 交互 / 线性）
3. 网络宽度（256 / 1024）
4. 训练长度（200 / 1000 epochs）

测量指标（全统一，便于横向对比）：
- 训练前后核对齐 CKA(K_init, K_post)
- 训练前后核对标签对齐 CKA(K, yy^T)
- 特征值变化比率
- 特征向量旋转量（子空间典型相关）
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import warnings, itertools, os, time
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
        return self.fc2(torch.relu(self.fc1(x))).flatten()

    def get_features(self, x):
        with torch.no_grad():
            return torch.relu(self.fc1(torch.FloatTensor(x).to(DEVICE))).cpu().numpy()


# ============================================================
# 2. 8种数据条件
# ============================================================

CONDITIONS = {}

# ----- 2.1 多项式 -----
def poly_data(n, d):
    x = np.random.uniform(-1, 1, (n, d)).astype(np.float32)
    y = x[:, 0]**2 + x[:, 1]  # 简单非线性
    return x, (y - y.mean()) / y.std()
CONDITIONS['A_poly'] = poly_data

# ----- 2.2 径向函数 -----
def radial_data(n, d):
    x = np.random.uniform(-1, 1, (n, d)).astype(np.float32)
    r = np.sqrt(np.sum(x**2, axis=1))
    y = np.sin(np.pi * r / np.sqrt(d/3))  # 在[-1,1]^d上大约0.5-2个周期
    return x, (y - y.mean()) / y.std()
CONDITIONS['B_radial'] = radial_data

# ----- 2.3 高频震荡 -----
def highfreq_data(n, d):
    x = np.random.uniform(-1, 1, (n, d)).astype(np.float32)
    y = np.sin(5 * x[:, 0]) + np.cos(7 * x[:, 1])
    return x, (y - y.mean()) / y.std()
CONDITIONS['C_highfreq'] = highfreq_data

# ----- 2.4 阶梯/间断 -----
def step_data(n, d):
    x = np.random.uniform(-1, 1, (n, d)).astype(np.float32)
    y = np.sign(x[:, 0]) + np.sign(x[:, 1])  # {-2, 0, 2}
    return x, (y - y.mean()) / y.std()
CONDITIONS['D_step'] = step_data

# ----- 2.5 高斯混合 -----
def gmm_data(n, d):
    x1 = np.random.randn(n//2, d) * 0.5 + 1.0
    x2 = np.random.randn(n - n//2, d) * 0.5 - 1.0
    x = np.vstack([x1, x2]).astype(np.float32)
    y = np.array([1]*len(x1) + [0]*len(x2)).astype(np.float32)
    return x, (y - y.mean()) / y.std()
CONDITIONS['E_gmm'] = gmm_data

# ----- 2.6 同心球壳 -----
def shell_data(n, d):
    n1 = n // 2
    x1 = np.random.randn(n1, d)
    x1 = x1 / np.linalg.norm(x1, axis=1, keepdims=True) * 1.0
    x2 = np.random.randn(n - n1, d)
    x2 = x2 / np.linalg.norm(x2, axis=1, keepdims=True) * 2.0
    x = np.vstack([x1, x2]).astype(np.float32)
    y = np.array([1]*n1 + [0]*(n-n1)).astype(np.float32)
    return x, (y - y.mean()) / y.std()
CONDITIONS['F_shell'] = shell_data

# ----- 2.7 稀疏线性 -----
def sparse_lin_data(n, d):
    x = np.random.uniform(-1, 1, (n, d)).astype(np.float32)
    # 只有前 3 维有信号
    y = x[:, 0] - 2.0 * x[:, 1] + 0.5 * x[:, 2]
    return x, (y - y.mean()) / y.std()
CONDITIONS['G_sparse_lin'] = sparse_lin_data

# ----- 2.8 乘法交互 -----
def interaction_data(n, d):
    x = np.random.uniform(-1, 1, (n, d)).astype(np.float32)
    y = x[:, 0] * x[:, 1] + x[:, 2] * x[:, 3]  # 纯交互，无主效应
    return x, (y - y.mean()) / y.std()
CONDITIONS['H_interaction'] = interaction_data


# ============================================================
# 3. 测量工具
# ============================================================

def decompose(K):
    eigvals, eigvecs = eigh(K)
    return eigvals[::-1], eigvecs[:, ::-1]


def cka(K, L):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H; Lc = H @ L @ H
    return np.sum(Kc * Lc) / (np.sqrt(np.sum(Kc**2) * np.sum(Lc**2)) + 1e-12)


def subspace_align(V1, V2, k=20):
    """前 k 个 eigenvectors 的典型相关（SVD of cross matrix）"""
    k = min(k, V1.shape[1], V2.shape[1])
    U = V1[:, :k]; V = V2[:, :k]
    s = np.linalg.svd(U.T @ V, compute_uv=False)
    return s  # 典型相关系数，越接近1越对齐


# ============================================================
# 4. 单条件实验
# ============================================================

def run_condition(name, data_fn, d=10, n=300, width=256,
                  noise=0.1, lr=0.1, epochs=500):
    """在一种数据条件下跑实验"""

    # 数据
    X_np, y_np = data_fn(n, d)

    # ---------- 初始特征核 ----------
    m_init = TwoLayerNet(d, width).to(DEVICE)
    H_i = m_init.get_features(X_np)
    K_init = (H_i @ H_i.T) / width

    # ---------- 训练 ----------
    m = TwoLayerNet(d, width).to(DEVICE)
    X_t = torch.FloatTensor(X_np).to(DEVICE)
    y_t = torch.FloatTensor(y_np + noise * np.random.randn(n)).to(DEVICE)
    opt = optim.SGD(m.parameters(), lr=lr, momentum=0.9)

    t0 = time.time()
    losses = []
    for ep in range(epochs):
        opt.zero_grad()
        loss = nn.MSELoss()(m(X_t), y_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
        opt.step()
        losses.append(loss.item())
        if torch.isnan(loss):
            break
    t_elapsed = time.time() - t0

    # ---------- 训练后特征核 ----------
    H_p = m.get_features(X_np)
    K_post = (H_p @ H_p.T) / width

    # ---------- 分解 ----------
    ei, vi = decompose(K_init)
    ep_, vp = decompose(K_post)

    # ---------- 度量 ----------
    L = np.outer(y_np, y_np)
    metrics = {
        'cka_init_post': cka(K_init, K_post),
        'cka_init_label': cka(K_init, L),
        'cka_post_label': cka(K_post, L),
        'align_top5': np.mean(subspace_align(vi, vp, k=5)),
        'align_top20': np.mean(subspace_align(vi, vp, k=20)),
        'eigvar_init': np.var(np.log(ei + 1e-12)),
        'eigvar_post': np.var(np.log(ep_ + 1e-12)),
        'loss_final': losses[-1],
        'train_time': t_elapsed,
    }
    return {
        'name': name, 'metrics': metrics,
        'ei': ei, 'ep': ep_, 'vi': vi, 'vp': vp,
        'K_init': K_init, 'K_post': K_post,
        'losses': losses,
    }


# ============================================================
# 5. 运行全部条件
# ============================================================

def run_all(epochs=500):
    results = []
    for name in sorted(CONDITIONS.keys()):
        fn = CONDITIONS[name]
        print(f"\n[{name}] d=10, n=300, width=256, epochs={epochs}", flush=True)
        r = run_condition(name, fn, epochs=epochs)
        m = r['metrics']
        print(f"  cka(i,p)={m['cka_init_post']:.4f}  "
              f"cka(i,L)={m['cka_init_label']:.4f}  "
              f"cka(p,L)={m['cka_post_label']:.4f}  "
              f"align@5={m['align_top5']:.4f}  "
              f"time={m['train_time']:.0f}s")
        results.append(r)
    return results


# ============================================================
# 6. 绘图
# ============================================================

def plot_summary(results):
    names = [r['name'] for r in results]
    metrics_keys = ['cka_init_post', 'cka_init_label', 'cka_post_label',
                    'align_top5', 'align_top20']
    labels = ['CKA(i,p)', 'CKA(i,L)', 'CKA(p,L)', 'Subspace@5', 'Subspace@20']

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) 度量热图
    ax = axes[0, 0]
    data = np.array([[r['metrics'][k] for k in metrics_keys] for r in results])
    im = ax.imshow(data, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(metrics_keys)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_title('Metrics across conditions')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # (b) CKA标签对齐对比
    ax = axes[0, 1]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, [r['metrics']['cka_init_label'] for r in results], w,
           label='Initial', color='C0', alpha=0.8)
    ax.bar(x + w/2, [r['metrics']['cka_post_label'] for r in results], w,
           label='Post-train', color='C1', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('CKA with label')
    ax.set_title('Kernel-Label Alignment')
    ax.legend(fontsize=8)

    # (c) 特征值变化：top-50比率
    ax = axes[0, 2]
    for r in results:
        ratio = r['ep'][:50] / (r['ei'][:50] + 1e-12)
        ax.plot(range(1, 51), ratio, label=r['name'], alpha=0.7)
    ax.axhline(1.0, color='gray', ls='--')
    ax.set_xlabel('Eigenvalue index')
    ax.set_ylabel('λ_post / λ_init')
    ax.set_title('Eigenvalue change (top 50)')
    ax.legend(fontsize=6, ncol=2)

    # (d) CKA(i,p) vs CKA(p,L) — 任何相关？
    ax = axes[1, 0]
    x = [r['metrics']['cka_init_post'] for r in results]
    y = [r['metrics']['cka_post_label'] for r in results]
    ax.scatter(x, y, c=range(len(results)), cmap='tab10', s=80, zorder=3)
    for i, n in enumerate(names):
        ax.annotate(n[-4:], (x[i], y[i]), fontsize=7, xytext=(3, 3),
                    textcoords='offset points')
    ax.set_xlabel('CKA(init, post)')
    ax.set_ylabel('CKA(post, label)')
    ax.set_title('Kernel stability vs alignment')

    # (e) 子空间对齐 vs 特征值变化
    ax = axes[1, 1]
    x = [r['metrics']['align_top5'] for r in results]
    y = [np.mean(r['ep'][:20] / (r['ei'][:20] + 1e-12)) for r in results]
    ax.scatter(x, y, c=range(len(results)), cmap='tab10', s=80, zorder=3)
    for i, n in enumerate(names):
        ax.annotate(n[-4:], (x[i], y[i]), fontsize=7, xytext=(3, 3),
                    textcoords='offset points')
    ax.set_xlabel('Subspace alignment@5')
    ax.set_ylabel('Mean eig ratio (top 20)')
    ax.set_title('Rotation vs eigenvalue change')

    # (f) 训练损失曲线
    ax = axes[1, 2]
    for r in results:
        ax.plot(r['losses'], label=r['name'], alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training curves')
    ax.legend(fontsize=6, ncol=2)

    plt.tight_layout()
    plt.savefig('/Users/wangyaoping/Desktop/ml_paper/figures/8conditions.png', dpi=150)
    plt.savefig('/Users/wangyaoping/Desktop/ml_paper/figures/8conditions.pdf')
    print(f"\nFigure saved.")
    plt.show()


def print_table(results):
    """打印便于比较的表格"""
    print("\n\n" + "="*120)
    print(f"{'Condition':<16} {'CKA(i,p)':<10} {'CKA(i,L)':<10} {'CKA(p,L)':<10} "
          f"{'ΔCKA':<10} {'Sub@5':<10} {'Sub@20':<10} {'eig_ratio':<10} {'loss':<10}")
    print("-"*120)
    for r in results:
        m = r['metrics']
        eig_r = np.mean(r['ep'][:20] / (r['ei'][:20] + 1e-12))
        d_cka = m['cka_post_label'] - m['cka_init_label']
        print(f"{r['name']:<16} {m['cka_init_post']:<10.4f} {m['cka_init_label']:<10.4f} "
              f"{m['cka_post_label']:<10.4f} {d_cka:<+10.4f} {m['align_top5']:<10.4f} "
              f"{m['align_top20']:<10.4f} {eig_r:<10.4f} {m['loss_final']:<10.6f}")
    print("="*120)


if __name__ == '__main__':
    os.makedirs('/Users/wangyaoping/Desktop/ml_paper/figures', exist_ok=True)

    # 条件齐全跑一次
    results = run_all(epochs=500)
    print_table(results)
    plot_summary(results)
