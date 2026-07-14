"""
Gap 1 实验 (v4)：特征核对比

用神经网络隐藏层激活值定义"特征核"K_feat(x, x') = h(x)^T h(x')。
比较初始（随机权重）和训练后的特征核，测量：
1. 核对齐度 CKA(K, yy^T)
2. 用该核做 KRR 的泛化误差
3. ΔAlign vs ΔGen 的关系
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


# ============================================================
# 1. 模型与特征核
# ============================================================

class TwoLayerNet(nn.Module):
    def __init__(self, d_in, width, d_out=1):
        super().__init__()
        self.width = width
        self.fc1 = nn.Linear(d_in, width, bias=False)
        self.fc2 = nn.Linear(width, d_out, bias=False)
        nn.init.normal_(self.fc1.weight, std=np.sqrt(2.0 / d_in))
        nn.init.normal_(self.fc2.weight, std=np.sqrt(2.0 / width))

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc2(h)

    def features(self, x):
        """提取隐藏层特征"""
        with torch.no_grad():
            x = torch.FloatTensor(x).to(DEVICE)
            h = torch.relu(self.fc1(x))
            return h.cpu().numpy()


def feature_kernel(model, X, Y=None):
    """特征核 K(x, y) = h(x)^T h(y) / width"""
    hX = model.features(X)
    if Y is None:
        hY = hX
    else:
        hY = model.features(Y)
    return (hX @ hY.T) / model.width


# ============================================================
# 2. 数据
# ============================================================

def sample_sphere(n, d):
    x = np.random.randn(n, d)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def ntk_kernel(X, Y=None):
    """无限宽 NTK 精确公式"""
    if Y is None:
        Y = X
    dot = X @ Y.T
    cos_theta = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    k_nngp = (np.sin(theta) + (np.pi - theta) * cos_theta) / np.pi
    k_dot = ((np.pi - theta) / np.pi) * dot
    return k_nngp + k_dot


def construct_target(K, spectral_decay=1.0, seed=SEED):
    """从核谱构造目标"""
    rng = np.random.RandomState(seed)
    eigvals, eigvecs = eigh(K)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    w = (eigvals ** (spectral_decay / 2)) * rng.randn(len(eigvals))
    y = eigvecs @ w
    return y / np.std(y)


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
# 4. 实验
# ============================================================

def run_experiment(d=10, n_train=200, n_test=500, width=1024,
                   spectral_decay=1.0, noise_level=0.1,
                   lam_ridge=1e-2, lr=0.1, epochs=300):
    """单次实验"""
    n_total = n_train + n_test
    X = sample_sphere(n_total, d)
    K_full = ntk_kernel(X)
    y_full = construct_target(K_full, spectral_decay)
    X_train, X_test = X[:n_train], X[n_train:]
    y_clean_train = y_full[:n_train]
    y_clean_test = y_full[n_train:]
    y_train = y_clean_train + noise_level * np.random.randn(n_train)

    # 无限宽 NTK baseline
    K_inf_tr = K_full[:n_train, :n_train]
    K_inf_te = K_full[n_train:, :n_train]

    def krr(K_tr, y_tr, K_te, lam):
        n = len(y_tr)
        # 添加数值保护
        K_reg = K_tr + n * lam * np.eye(n)
        K_reg = (K_reg + K_reg.T) / 2
        try:
            a = np.linalg.solve(K_reg, y_tr)
        except np.linalg.LinAlgError:
            a = np.linalg.lstsq(K_reg, y_tr, rcond=None)[0]
        return K_te @ a

    err_inf = np.mean((krr(K_inf_tr, y_train, K_inf_te, lam_ridge) - y_clean_test) ** 2)

    # 训练神经网络
    model = TwoLayerNet(d, width).to(DEVICE)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    X_t = torch.FloatTensor(X_train).to(DEVICE)
    y_t = torch.FloatTensor(y_train).reshape(-1, 1).to(DEVICE)

    # 训练 + 隐式早停用 validation
    n_val = n_train // 5
    n_tr = n_train - n_val
    X_tr, X_val = X_t[:n_tr], X_t[n_tr:]
    y_tr, y_val = y_t[:n_tr], y_t[n_tr:]

    best_val = float('inf')
    best_state = None
    for ep in range(epochs):
        opt.zero_grad()
        loss = nn.MSELoss()(model(X_tr), y_tr)
        loss.backward()
        opt.step()
        if ep % 20 == 0:
            with torch.no_grad():
                vloss = nn.MSELoss()(model(X_val), y_val).item()
            if vloss < best_val:
                best_val = vloss
            if ep > 50 and vloss > 1.5 * best_val:
                break  # early stopping

    # ---- 训练前特征核 ----
    model_init = TwoLayerNet(d, width).to(DEVICE)  # 重新初始化
    K_feat_init_tr = feature_kernel(model_init, X_train)
    K_feat_init_te = feature_kernel(model_init, X_test, X_train)
    err_init = np.mean((krr(K_feat_init_tr, y_train, K_feat_init_te, lam_ridge) - y_clean_test) ** 2)

    # ---- 训练后特征核 ----
    K_feat_tr = feature_kernel(model, X_train)
    K_feat_te = feature_kernel(model, X_test, X_train)
    err_post = np.mean((krr(K_feat_tr, y_train, K_feat_te, lam_ridge) - y_clean_test) ** 2)

    # 核对齐
    y_label = np.outer(y_clean_train, y_clean_train)
    cka_before = compute_cka(K_feat_init_tr, y_label)
    cka_after = compute_cka(K_feat_tr, y_label)
    cka_inf = compute_cka(K_inf_tr, y_label)

    return {
        'err_inf': err_inf,          # 无限宽 NTK
        'err_init': err_init,        # 初始特征核
        'err_post': err_post,        # 训练后特征核
        'err_gap_inf': err_post - err_inf,       # vs NTK
        'err_gap_init': err_post - err_init,     # vs 初始特征
        'err_rel_inf': (err_post - err_inf) / (err_inf + 1e-10),
        'err_rel_init': (err_post - err_init) / (err_init + 1e-10),
        'cka_inf': cka_inf,          # 无限宽 NTK 对齐
        'cka_before': cka_before,    # 初始特征核对齐
        'cka_after': cka_after,      # 训练后特征核对齐
        'delta_align_feat': cka_after - cka_before,  # 特征学习带来的对齐增益
        'delta_align_inf': cka_after - cka_inf,      # vs NTK 的对齐增益
    }


# ============================================================
# 5. 扫描
# ============================================================

def scan_spectral_decay():
    print("=== 扫描 spectral_decay ===")
    decays = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    results = []
    for sd in decays:
        print(f"\ndecay={sd}", end=' ', flush=True)
        r = run_experiment(spectral_decay=sd, epochs=300)
        r['x'] = sd
        results.append(r)
        print(f"inf={r['err_inf']:.4f} init={r['err_init']:.4f} post={r['err_post']:.4f}  "
              f"rel_inf={r['err_rel_inf']*100:+.1f}%  dAlign={r['delta_align_feat']:.4f}")
    return results


def scan_noise():
    print("\n=== 扫描 noise ===")
    noises = [0.0, 0.05, 0.1, 0.2, 0.5]
    results = []
    for nl in noises:
        print(f"\nnoise={nl}", end=' ', flush=True)
        r = run_experiment(spectral_decay=0.8, noise_level=nl, epochs=300)
        r['x'] = nl
        results.append(r)
        print(f"inf={r['err_inf']:.4f} init={r['err_init']:.4f} post={r['err_post']:.4f}  "
              f"rel_inf={r['err_rel_inf']*100:+.1f}%  dAlign={r['delta_align_feat']:.4f}")
    return results


def scan_width():
    print("\n=== 扫描 width ===")
    widths = [128, 256, 512, 1024, 2048]
    results = []
    for w in widths:
        print(f"\nwidth={w}", end=' ', flush=True)
        r = run_experiment(width=w, spectral_decay=0.8, noise_level=0.1, epochs=300)
        r['x'] = w
        results.append(r)
        print(f"inf={r['err_inf']:.4f} init={r['err_init']:.4f} post={r['err_post']:.4f}  "
              f"rel_inf={r['err_rel_inf']*100:+.1f}%  dAlign={r['delta_align_feat']:.4f}")
    return results


def plot_all():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    all_gaps, all_align = [], []

    datasets = [
        ('(a) Spectral Decay → Gap', scan_spectral_decay(), 'x'),
        ('(b) Noise Level → Gap', scan_noise(), 'x'),
        ('(c) Width → Gap', scan_width(), 'x'),
    ]

    for idx, (title, data, xkey) in enumerate(datasets):
        ax = axes[idx // 3, idx % 3]
        x = [r[xkey] for r in data]
        # 三条线
        ax.plot(x, [r['err_inf'] for r in data], 'o-', label='∞NTK', color='gray')
        ax.plot(x, [r['err_init'] for r in data], 's--', label='Init Feat', color='C0')
        ax.plot(x, [r['err_post'] for r in data], '^--', label='Post Feat', color='C1')
        ax.set_xlabel(xkey)
        ax.set_ylabel('Test MSE')
        ax.set_title(title)
        ax.legend()
        for r in data:
            all_gaps.append(r['err_rel_inf'] * 100)
            all_align.append(r['delta_align_feat'])

    # ΔAlign vs ΔGen
    ax = axes[1, 1]
    ax.scatter(all_align, all_gaps, c='C2', s=50, alpha=0.7, zorder=3)
    coeffs = np.polyfit(all_align, all_gaps, 1)
    xf = np.linspace(min(all_align), max(all_align), 100)
    ax.plot(xf, np.polyval(coeffs, xf), 'r--', label=f'y={coeffs[0]:.2f}x+{coeffs[1]:.2f}')
    ax.axhline(0, color='gray', ls=':')
    ax.axvline(0, color='gray', ls=':')
    corr = np.corrcoef(all_align, all_gaps)[0, 1]
    ax.set_xlabel('ΔAlignment (CKA_feat_post - CKA_feat_init)')
    ax.set_ylabel('ΔGen (rel %)')
    ax.set_title(f'(d) ΔAlign vs ΔGen  r={corr:.3f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig('/Users/wangyaoping/Desktop/ml_paper/figures/exp_v4.png', dpi=150)
    plt.savefig('/Users/wangyaoping/Desktop/ml_paper/figures/exp_v4.pdf')
    print(f"\nSaved.")


if __name__ == '__main__':
    import os
    os.makedirs('/Users/wangyaoping/Desktop/ml_paper/figures', exist_ok=True)
    plot_all()
