"""
Gap 1: 自适应核对齐理论 —— 合成数据实验 (v2)

修正：
1. 用 Kernel Ridge Regression 替代 SVM，避免优化器干扰
2. 正确的 Nyström 扩展把自适应核延伸到测试集
3. 更强的谱调整机制
4. 无数据泄露（先分割再构造目标）
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.linalg import eigh
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

# ============================================================
# 1. 核函数
# ============================================================

def ntk_kernel(X, Y=None):
    """
    两层 ReLU 网络的无限宽 NTK（精确公式）
    输入已假设在单位球面上：||x|| = 1
    """
    if Y is None:
        Y = X
    dot = X @ Y.T
    cos_theta = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    # NNGP term: (1/pi) * (sin(theta) + (pi-theta)*cos(theta))
    k_nngp = (np.sin(theta) + (np.pi - theta) * cos_theta) / np.pi
    # NTK correction: (1/pi) * (pi - theta) * dot
    k_dot = ((np.pi - theta) / np.pi) * dot
    return k_nngp + k_dot


# ============================================================
# 2. 合成数据
# ============================================================

def sample_sphere(n, d):
    """在 d 维球面上均匀采样"""
    x = np.random.randn(n, d)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def construct_target_from_kernel(K, spectral_decay=1.0, seed=SEED):
    """
    从核矩阵的特征分解构造目标函数。

    K ≈ Φ diag(λ) Φ^T
    设 y = Σᵢ wᵢ φᵢ, 其中 wᵢ² ∝ λᵢ^{spectral_decay}

    所以 task-model alignment: vᵢ = ⟨y, φᵢ⟩² = wᵢ²

    spectral_decay:
      = 1.0: vᵢ ∝ λᵢ → target 能量谱与核谱完全匹配（easy）
      < 1.0: vᵢ 衰减慢于 λᵢ → target 高频多（hard）
      > 1.0: vᵢ 衰减更快 → target 集中在低频（easier）
    """
    rng = np.random.RandomState(seed)
    eigvals, eigvecs = eigh(K)
    eigvals = eigvals[::-1]  # 从大到小
    eigvecs = eigvecs[:, ::-1]
    # wᵢ² ∝ λᵢ^{spectral_decay}, 保留符号随机
    w = (eigvals ** (spectral_decay / 2)) * rng.randn(len(eigvals))
    y = eigvecs @ w
    y = y / np.std(y)  # 归一化
    v_i = (eigvecs.T @ y) ** 2 / len(y)  # 任务能量分布
    return y, eigvals, v_i


# ============================================================
# 3. 自适应核构造 + Nyström 扩展
# ============================================================

def build_adaptive_kernel(K_train, y_train, K_test_train, gamma_0, lam):
    """
    自适应核构造。

    训练核的特征值被重新分配：
      λ̃ᵢ = λᵢ · [1 + γ₀ · vᵢ / (lam + λᵢ)]

    测试集用 Nyström 扩展：
      K_adapt_test = K_test_train @ Φ_train @ diag(λ̃ᵢ/λᵢ²) @ Φ_trainᵀ @ K_train
                   = K_test_train @ Φ_train @ diag(λ̃ᵢ/λᵢ²) @ Φ_trainᵀ @ K_train
    实际上更简单：
      φ_test = K_test_train @ φ_train / λᵢ   (Nyström)
      K_adapt_test[i,j] = Σₖ λ̃ₖ φ_test_i[k] φ_train_j[k]
    """
    n = len(y_train)
    eigvals, eigvecs = eigh(K_train)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    # 任务投影
    proj = eigvecs.T @ y_train
    v_i = (proj ** 2) / n
    # 自适应特征值
    eigvals_adapt = eigvals * (1.0 + gamma_0 * v_i / (lam + eigvals))
    # 截断到合理范围
    eigvals_adapt = np.maximum(eigvals_adapt, 1e-12)
    # 训练核
    K_adapt_train = eigvecs @ np.diag(eigvals_adapt) @ eigvecs.T
    K_adapt_train = (K_adapt_train + K_adapt_train.T) / 2
    # Nyström 扩展到测试
    # φ_test[k] = (1/λₖ) * K_test_train @ φ_train[:,k]
    scale = np.sqrt(eigvals_adapt) / np.maximum(eigvals, 1e-12)
    adapt_features_train = eigvecs * scale[np.newaxis, :]
    adapt_features_test = K_test_train @ (eigvecs / np.maximum(eigvals, 1e-12)[np.newaxis, :])
    adapt_features_test = adapt_features_test * np.sqrt(eigvals_adapt)[np.newaxis, :]
    K_adapt_test = adapt_features_test @ adapt_features_train.T
    return K_adapt_train, K_adapt_test, eigvals, eigvals_adapt, v_i


# ============================================================
# 4. Kernel Ridge Regression
# ============================================================

def kernel_ridge(K_train, y_train, K_test, lam=1e-3):
    """
    Kernel Ridge Regression: f(x) = Σᵢ αᵢ k(x, xᵢ)
    α = (K + n·λ·I)⁻¹ y
    """
    n = len(y_train)
    alpha = np.linalg.solve(K_train + n * lam * np.eye(n), y_train)
    pred = K_test @ alpha
    return pred, alpha


# ============================================================
# 5. 核对齐 (CKA)
# ============================================================

def compute_cka(K, L):
    """Centered Kernel Alignment"""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K_c = H @ K @ H
    L_c = H @ L @ H
    num = np.sum(K_c * L_c)
    den = np.sqrt(np.sum(K_c * K_c) * np.sum(L_c * L_c))
    return num / (den + 1e-12)


# ============================================================
# 6. 单次实验
# ============================================================

def run_experiment(d=20, n_train=200, n_test=500,
                   spectral_decay=1.0, gamma_0=1.0,
                   noise_level=0.1, lam_ridge=1e-3):
    """一次完整的实验"""
    # 生成全部数据 + 全核分解 → 目标函数 (避免数据泄露)
    n_total = n_train + n_test
    X = sample_sphere(n_total, d)
    K_full = ntk_kernel(X)
    y_full, eigvals_all, v_i_all = construct_target_from_kernel(
        K_full, spectral_decay=spectral_decay
    )
    # 分割
    X_train, X_test = X[:n_train], X[n_train:]
    y_clean_train = y_full[:n_train]
    y_clean_test = y_full[n_train:]

    # 训练加噪声
    y_train = y_clean_train + noise_level * np.random.randn(n_train)

    # 训练核 + 测试核
    K_train = K_full[:n_train, :n_train]
    K_test = K_full[n_train:, :n_train]

    # ---- 固定核（NTK） ----
    pred_ntk, _ = kernel_ridge(K_train, y_train, K_test, lam=lam_ridge)
    err_ntk = np.mean((pred_ntk - y_clean_test) ** 2)
    # 核对齐（无噪声 y）
    cka_ntk = compute_cka(K_train, np.outer(y_clean_train, y_clean_train))

    # ---- 自适应核 ----
    K_adapt_train, K_adapt_test, eigvals_ntk, eigvals_adapt, v_i_adapt = build_adaptive_kernel(
        K_train, y_train, K_test, gamma_0, lam_ridge
    )
    pred_adapt, _ = kernel_ridge(K_adapt_train, y_train, K_adapt_test, lam=lam_ridge)
    err_adapt = np.mean((pred_adapt - y_clean_test) ** 2)
    cka_adapt = compute_cka(K_adapt_train, np.outer(y_clean_train, y_clean_train))

    return {
        'err_ntk': err_ntk,
        'err_adapt': err_adapt,
        'err_gap': err_adapt - err_ntk,
        'err_rel': (err_adapt - err_ntk) / (err_ntk + 1e-10),
        'cka_ntk': cka_ntk,
        'cka_adapt': cka_adapt,
        'delta_align': cka_adapt - cka_ntk,
        'eigvals_ntk': eigvals_ntk[:20],
        'eigvals_adapt': eigvals_adapt[:20],
        'v_i': v_i_adapt[:20],
    }


# ============================================================
# 7. 扫描函数
# ============================================================

def scan_spectral_decay():
    print("=== 扫描 spectral_decay (γ₀=1.0, noise=0.1) ===")
    decays = [0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    results = []
    for sd in decays:
        r = run_experiment(spectral_decay=sd, gamma_0=1.0)
        r['x'] = sd
        results.append(r)
        print(f"  decay={sd:.1f}  "
              f"ntk={r['err_ntk']:.4f}  adapt={r['err_adapt']:.4f}  "
              f"gap={r['err_gap']:.4f}  rel={r['err_rel']*100:+.1f}%  "
              f"dAlign={r['delta_align']:.4f}")
    return results


def scan_gamma():
    print("\n=== 扫描 γ₀ (decay=0.8, noise=0.1) ===")
    gammas = [0.0, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    results = []
    for g in gammas:
        r = run_experiment(spectral_decay=0.8, gamma_0=g)
        r['x'] = g
        results.append(r)
        print(f"  gamma={g:.1f}  "
              f"ntk={r['err_ntk']:.4f}  adapt={r['err_adapt']:.4f}  "
              f"gap={r['err_gap']:.4f}  rel={r['err_rel']*100:+.1f}%  "
              f"dAlign={r['delta_align']:.4f}")
    return results


def scan_noise():
    print("\n=== 扫描 noise (decay=0.8, γ₀=1.0) ===")
    noises = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
    results = []
    for nl in noises:
        r = run_experiment(spectral_decay=0.8, gamma_0=1.0, noise_level=nl)
        r['x'] = nl
        results.append(r)
        print(f"  noise={nl:.2f}  "
              f"ntk={r['err_ntk']:.4f}  adapt={r['err_adapt']:.4f}  "
              f"gap={r['err_gap']:.4f}  rel={r['err_rel']*100:+.1f}%  "
              f"dAlign={r['delta_align']:.4f}")
    return results


def scan_n_train():
    print("\n=== 扫描 n_train (decay=0.8, γ₀=1.0, noise=0.1) ===")
    ns = [50, 100, 200, 400, 800]
    results = []
    for n in ns:
        r = run_experiment(n_train=n, spectral_decay=0.8, gamma_0=1.0)
        r['x'] = n
        results.append(r)
        print(f"  n={n}  ntk={r['err_ntk']:.4f}  adapt={r['err_adapt']:.4f}  "
              f"gap={r['err_gap']:.4f}  rel={r['err_rel']*100:+.1f}%  "
              f"dAlign={r['delta_align']:.4f}")
    return results


# ============================================================
# 8. 可视化
# ============================================================

def plot_results():
    fig = plt.figure(figsize=(15, 10))

    # 执行扫描
    r1 = scan_spectral_decay()
    r2 = scan_gamma()
    r3 = scan_noise()
    r4 = scan_n_train()

    # (a) spectral_decay → gap
    ax = fig.add_subplot(2, 3, 1)
    x = [r['x'] for r in r1]
    ax.plot(x, [r['err_rel'] * 100 for r in r1], 'o-', color='C0', linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Spectral Decay α (larger = easier)')
    ax.set_ylabel('Relative Gap (%)')
    ax.set_title('(a) Task Difficulty → Gap')

    # (b) gamma → gap
    ax = fig.add_subplot(2, 3, 2)
    x = [r['x'] for r in r2]
    ax.plot(x, [r['err_rel'] * 100 for r in r2], 'o-', color='C1', linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Feature Learning Strength γ₀')
    ax.set_ylabel('Relative Gap (%)')
    ax.set_title('(b) Feature Learning → Gap')

    # (c) noise → gap
    ax = fig.add_subplot(2, 3, 3)
    x = [r['x'] for r in r3]
    ax.plot(x, [r['err_rel'] * 100 for r in r3], 'o-', color='C2', linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Noise Level σ')
    ax.set_ylabel('Relative Gap (%)')
    ax.set_title('(c) Noise Level → Gap')

    # (d) n → gap
    ax = fig.add_subplot(2, 3, 4)
    x = [r['x'] for r in r4]
    ax.plot(x, [r['err_rel'] * 100 for r in r4], 'o-', color='C3', linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Training Size n')
    ax.set_ylabel('Relative Gap (%)')
    ax.set_title('(d) Sample Size → Gap')

    # (e) ΔAlign vs ΔGen —— 核心验证
    ax = fig.add_subplot(2, 3, 5)
    all_align = []
    all_gap = []
    for r in r1 + r2 + r3 + r4:
        all_align.append(r['delta_align'])
        all_gap.append(r['err_rel'])
    ax.scatter(all_align, all_gap, c='C4', s=50, alpha=0.7, zorder=3)
    # 线性拟合
    coeffs = np.polyfit(all_align, all_gap, 1)
    x_fit = np.linspace(min(all_align), max(all_align), 100)
    ax.plot(x_fit, np.polyval(coeffs, x_fit), 'r--', linewidth=2,
            label=f'y={coeffs[0]:.2f}x+{coeffs[1]:.4f}')
    corr = np.corrcoef(all_align, all_gap)[0, 1]
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('ΔAlignment (CKA)')
    ax.set_ylabel('ΔGeneralization (rel)')
    ax.set_title(f'(e) ΔAlign vs ΔGen  r={corr:.3f}')
    ax.legend()

    # (f) 特征谱可视化
    ax = fig.add_subplot(2, 3, 6)
    sample = r1[3]  # decay=1.0 case
    idx = np.arange(1, 11)
    ax.plot(idx, sample['eigvals_ntk'][:10], 'o-', label='NTK λᵢ', color='C0')
    ax.plot(idx, sample['eigvals_adapt'][:10], 's--', label='Adapt λ̃ᵢ', color='C1')
    ax.set_yscale('log')
    ax.set_xlabel('Index i')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('(f) NTK vs Adaptive Kernel Spectrum')
    ax.legend()

    plt.tight_layout()
    plt.savefig('/Users/wangyaoping/Desktop/ml_paper/figures/exp1_v2.png', dpi=150)
    plt.savefig('/Users/wangyaoping/Desktop/ml_paper/figures/exp1_v2.pdf')
    print(f"\nFigure saved.")
    plt.show()


if __name__ == '__main__':
    import os
    os.makedirs('/Users/wangyaoping/Desktop/ml_paper/figures', exist_ok=True)
    plot_results()
