"""
谱级 kernel stability 分析（不是 CKA 级）

三层：
  1. 全局稳定性：CKA, Frobenius 相对变化（已有）
  2. 谱稳定性：特征值 λᵢ 的变化模式（逐 i 看）
  3. 目标投影稳定性：v_i = ⟨y, φᵢ⟩² 的变化
  4. 动态：epoch [0,10,50,100,200,500] 的演化
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import warnings, os
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
# 2. 数据条件（选 3 个有代表性的）
# ============================================================

CONDITIONS = {}

def poly_data(n, d):
    x = np.random.uniform(-1, 1, (n, d)).astype(np.float32)
    y = x[:, 0]**2 + x[:, 1]
    return x, (y - y.mean()) / y.std()
CONDITIONS['A_poly'] = poly_data

def highfreq_data(n, d):
    x = np.random.uniform(-1, 1, (n, d)).astype(np.float32)
    y = np.sin(5 * x[:, 0]) + np.cos(7 * x[:, 1])
    return x, (y - y.mean()) / y.std()
CONDITIONS['B_highfreq'] = highfreq_data

def gmm_data(n, d):
    n1 = n // 2
    x1 = np.random.randn(n1, d) * 0.5 + 1.0
    x2 = np.random.randn(n - n1, d) * 0.5 - 1.0
    x = np.vstack([x1, x2]).astype(np.float32)
    y = np.array([1]*n1 + [0]*(n-n1)).astype(np.float32)
    return x, (y - y.mean()) / y.std()
CONDITIONS['C_gmm'] = gmm_data


# ============================================================
# 3. 分析工具
# ============================================================

def decompose(K):
    eigvals, eigvecs = eigh(K)
    return eigvals[::-1], eigvecs[:, ::-1]


def cka(K, L):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H; Lc = H @ L @ H
    return np.sum(Kc * Lc) / (np.sqrt(np.sum(Kc**2) * np.sum(Lc**2)) + 1e-12)


# ============================================================
# 4. 主实验：track 整个训练过程的核变化
# ============================================================

def track_spectrum(name, data_fn, d=10, n=300, width=512,
                   noise=0.1, lr=0.05, epochs=500, checkpoints=None):
    """在多个 checkpoint 保存核的完整谱信息"""
    if checkpoints is None:
        checkpoints = [0, 10, 20, 50, 100, 200, 500]

    X_np, y_np = data_fn(n, d)
    y_train = y_np + noise * np.random.randn(n)

    # 初始化网络
    model = TwoLayerNet(d, width).to(DEVICE)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    X_t = torch.FloatTensor(X_np).to(DEVICE)
    y_t = torch.FloatTensor(y_train).to(DEVICE)

    results = {}
    cp_set = sorted(set(checkpoints))

    for ep in range(max(cp_set) + 1):
        if ep in cp_set:
            # 提取当前特征核 + 分解
            H = model.get_features(X_np)
            K = (H @ H.T) / width
            eigvals, eigvecs = decompose(K)
            proj = eigvecs.T @ y_np
            v_i = (proj ** 2) / n

            results[ep] = {
                'K': K,
                'eigvals': eigvals,
                'eigvecs': eigvecs,
                'v_i': v_i,
                'aligned': np.sum(v_i * eigvals),  # 一个标量：总对齐 = Σ v_i λ_i
            }

        # 训练一步
        if ep < max(cp_set):
            opt.zero_grad()
            loss = nn.MSELoss()(model(X_t), y_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    return results


# ============================================================
# 5. 分析和绘图
# ============================================================

def analyze(results_dict, name):
    """分析一个条件的结果"""
    cps = sorted(results_dict.keys())
    base = results_dict[0]

    # ---- 1. CKA 演化 ----
    ckas = []
    fr_diff = []
    for ep in cps:
        ckas.append(cka(base['K'], results_dict[ep]['K']))
        diff = results_dict[ep]['K'] - base['K']
        fr_diff.append(np.linalg.norm(diff, 'fro') / np.linalg.norm(base['K'], 'fro'))

    # ---- 2. 特征值变化（最后一个 epoch vs 初始） ----
    last = results_dict[max(cps)]
    eig_ratio = last['eigvals'][:30] / (base['eigvals'][:30] + 1e-12)
    eig_base = base['eigvals'][:30]
    eig_last = last['eigvals'][:30]

    # ---- 3. 目标投影变化 ----
    v_ratio = last['v_i'][:30] / (base['v_i'][:30] + 1e-12)

    # ---- 4. 总对齐演化 ----
    align_curve = [results_dict[ep]['aligned'] for ep in cps]

    return {
        'ckas': ckas, 'fr_diff': fr_diff,
        'eig_ratio': eig_ratio, 'eig_base': eig_base, 'eig_last': eig_last,
        'v_ratio': v_ratio,
        'align_curve': align_curve,
        'v_i_base': base['v_i'][:30],
        'v_i_last': last['v_i'][:30],
        'cps': cps,
    }


def plot_all():
    fig, axes = plt.subplots(3, 4, figsize=(20, 13))
    colors = {'A_poly': 'C0', 'B_highfreq': 'C1', 'C_gmm': 'C2'}

    all_results = {}

    for idx, (name, fn) in enumerate(sorted(CONDITIONS.items())):
        print(f"\n[{name}] tracking spectrum...", flush=True)
        r = track_spectrum(name, fn, n=300, width=512, epochs=500)
        a = analyze(r, name)
        all_results[name] = a
        c = colors[name]

        # ---- 列 0: CKA + Frobenius 演化 ----
        ax = axes[idx, 0]
        ax2 = ax.twinx()
        ax.plot(a['cps'], a['ckas'], 'o-', color=c, label='CKA', linewidth=2)
        ax2.plot(a['cps'], a['fr_diff'], 's--', color=c, alpha=0.5, label='Frob diff')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('CKA', color=c)
        ax2.set_ylabel('‖ΔK‖/‖K‖', color=c, alpha=0.5)
        ax.set_title(f'{name}: Global Stability')

        # ---- 列 1: 特征值（before vs after） ----
        ax = axes[idx, 1]
        idx_ = np.arange(1, 31)
        ax.plot(idx_, a['eig_base'], 'o-', label='epoch 0', color=c, alpha=0.7)
        ax.plot(idx_, a['eig_last'], 's--', label=f'epoch {a["cps"][-1]}', color=c)
        ax.set_yscale('log')
        ax.set_xlabel('Index i')
        ax.set_ylabel('λᵢ')
        ax.set_title(f'{name}: Eigenvalues')
        ax.legend(fontsize=8)

        # ---- 列 2: 特征值比率 λ_t/λ_0 ----
        ax = axes[idx, 2]
        idx_ = np.arange(1, 31)
        ax.bar(idx_, a['eig_ratio'], color=c, alpha=0.7)
        ax.axhline(1.0, color='gray', ls='--')
        ax.set_xlabel('Index i')
        ax.set_ylabel('λ_post / λ_init')
        ax.set_title(f'{name}: Eigenvalue Change Ratio')

        # ---- 列 3: 目标投影 v_i（before vs after） ----
        ax = axes[idx, 3]
        idx_ = np.arange(1, 31)
        ax.plot(idx_, a['v_i_base'], 'o-', label='epoch 0', color=c, alpha=0.7)
        ax.plot(idx_, a['v_i_last'], 's--', label=f'epoch {a["cps"][-1]}', color=c)
        ax.set_xlabel('Index i')
        ax.set_ylabel('v_i = ⟨y, φᵢ⟩²/n')
        ax.set_title(f'{name}: Target Projection v_i')
        ax.legend(fontsize=8)

    # ---- 底部对齐演化汇总 ----
    ax = axes[2, 0]
    for name, a in all_results.items():
        ax.plot(a['cps'], a['align_curve'], 'o-', label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Σ vᵢ λᵢ')
    ax.set_title('Total Alignment Over Training')
    ax.legend()

    plt.tight_layout()
    plt.savefig('/Users/wangyaoping/Desktop/ml_paper/figures/spectral_analysis.png', dpi=150)
    plt.savefig('/Users/wangyaoping/Desktop/ml_paper/figures/spectral_analysis.pdf')
    print(f"\nSaved.")
    plt.close('all')
    print("Done.", flush=True)


if __name__ == '__main__':
    os.makedirs('/Users/wangyaoping/Desktop/ml_paper/figures', exist_ok=True)
    plot_all()
