"""
Gap 1: 自适应核对齐理论 —— 最终实验 (v5)

合成数据 + 修正后的自适应核构造。

关键修正：
- 核特征值归一化到 sum=n
- 修改公式用相对尺度（相对于均值）
- 正确 Nyström 扩展
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)


# ============================================================
# 1. 核函数
# ============================================================

def ntk_kernel(X, Y=None):
    """两层 ReLU 无限宽 NTK 精确公式"""
    if Y is None:
        Y = X
    dot = X @ Y.T
    cos_theta = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    k_nngp = (np.sin(theta) + (np.pi - theta) * cos_theta) / np.pi
    return k_nngp + ((np.pi - theta) / np.pi) * dot


# ============================================================
# 2. 合成数据
# ============================================================

def sample_sphere(n, d):
    x = np.random.randn(n, d)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def construct_target(K, spectral_decay=1.0):
    """从核构造目标函数"""
    rng = np.random.RandomState(SEED)
    eigvals, eigvecs = eigh(K)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    w = (eigvals ** (spectral_decay / 2)) * rng.randn(len(eigvals))
    y = eigvecs @ w
    return y / np.std(y)


# ============================================================
# 3. 自适应核构造（修正版）
# ============================================================

def build_adaptive_kernel(K_train, y_train, K_test_train, gamma_0=1.0, lam=0.001):
    """
    自适应核：λ̃ᵢ = λᵢ * [1 + γ₀ * (vᵢ/mean(v)) / (1 + λᵢ/mean(λ))]

    步骤：
    1. 分解 K_train = Φ Λ Φ^T
    2. 归一化特征值 sum(λ) = n（对齐尺度）
    3. 计算 vᵢ = (φᵢ^T y)² / n
    4. 修改特征值
    5. Nyström 扩展到测试集
    """
    n = len(y_train)
    eigvals, eigvecs = eigh(K_train)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    # 归一化特征值到 sum = n
    eigvals = eigvals * n / np.sum(eigvals)

    # 任务投影
    proj = eigvecs.T @ y_train
    v_i = (proj ** 2) / n

    # 相对修改（修正版）
    mean_v = np.mean(v_i) + 1e-10
    mean_l = np.mean(eigvals) + 1e-10
    scale = v_i / mean_v
    rel_amplify = gamma_0 * scale / (1.0 + eigvals / mean_l)
    eigvals_adapt = eigvals * (1.0 + rel_amplify)
    eigvals_adapt = np.maximum(eigvals_adapt, 1e-12)

    # 重建训练核
    K_adapt_train = eigvecs @ np.diag(eigvals_adapt) @ eigvecs.T
    K_adapt_train = (K_adapt_train + K_adapt_train.T) / 2

    # Nyström 扩展到测试
    # φ_test = (1/λ) * K_xt @ φ_train
    # K_adapt_test = φ_test @ diag(λ̃) @ φ_train^T
    phi_test = K_test_train @ (eigvecs / np.maximum(eigvals, 1e-12))
    K_adapt_test = phi_test @ np.diag(eigvals_adapt) @ eigvecs.T

    return K_adapt_train, K_adapt_test, eigvals, eigvals_adapt, v_i


# ============================================================
# 4. Kernel Ridge Regression + 最优 λ 选择
# ============================================================

def best_lam_cv(K, y, folds=5, n_lams=50):
    """用 CV 选最优正则化 λ"""
    best_err = float('inf')
    best_l = 1.0
    kf = KFold(folds, shuffle=True, random_state=SEED)
    for logl in np.linspace(-5, 2, n_lams):
        l = 10 ** logl
        errs = []
        for tr, val in kf.split(y):
            K_tr = K[np.ix_(tr, tr)]
            K_vl = K[np.ix_(val, tr)]
            a = np.linalg.solve(K_tr + len(tr) * l * np.eye(len(tr)), y[tr])
            pred = K_vl @ a
            errs.append(np.mean((pred - y[val]) ** 2))
        e = np.mean(errs)
        if e < best_err:
            best_err = e
            best_l = l
    return best_l


def krr(K_train, y_train, K_test, lam):
    n = len(y_train)
    a = np.linalg.solve(K_train + n * lam * np.eye(n), y_train)
    return K_test @ a


# ============================================================
# 5. 核对齐
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
# 6. 实验
# ============================================================

def run_experiment(d=20, n_train=200, n_test=500,
                   spectral_decay=1.0, gamma_0=1.0,
                   noise_level=0.1):
    """单次实验"""
    # 全数据
    X = sample_sphere(n_train + n_test, d)
    K_full = ntk_kernel(X)
    y_full = construct_target(K_full, spectral_decay)

    X_train, X_test = X[:n_train], X[n_train:]
    K_inf_tr = K_full[:n_train, :n_train]
    K_inf_te = K_full[n_train:, :n_train]

    y_clean_tr = y_full[:n_train]
    y_clean_te = y_full[n_train:]
    y_train = y_clean_tr + noise_level * np.random.randn(n_train)

    # 标签核心矩阵（用于 CKA）
    L_tr = np.outer(y_clean_tr, y_clean_tr)

    # ---- 无限宽 NTK ----
    lam_inf = best_lam_cv(K_inf_tr, y_train)
    pred_inf = krr(K_inf_tr, y_train, K_inf_te, lam_inf)
    err_inf = np.mean((pred_inf - y_clean_te) ** 2)
    cka_inf = compute_cka(K_inf_tr, L_tr)

    # ---- 自适应核 ----
    K_ad_tr, K_ad_te, eig_ntk, eig_ad, v_i = build_adaptive_kernel(
        K_inf_tr, y_train, K_inf_te, gamma_0, lam_inf
    )
    lam_ad = best_lam_cv(K_ad_tr, y_train)
    pred_ad = krr(K_ad_tr, y_train, K_ad_te, lam_ad)
    err_ad = np.mean((pred_ad - y_clean_te) ** 2)
    cka_ad = compute_cka(K_ad_tr, L_tr)

    return {
        'err_inf': err_inf,
        'err_ad': err_ad,
        'err_gap': err_ad - err_inf,
        'err_rel': (err_ad - err_inf) / (err_inf + 1e-10),
        'cka_inf': cka_inf,
        'cka_ad': cka_ad,
        'delta_align': cka_ad - cka_inf,
        'lam_inf': lam_inf,
        'lam_ad': lam_ad,
    }


# ============================================================
# 7. 扫描实验
# ============================================================

def scan_spectral_decay():
    print("=== spectral_decay (γ₀=2.0, noise=0.1) ===")
    results = []
    for sd in [0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
        r = run_experiment(spectral_decay=sd, gamma_0=2.0)
        r['x'] = sd
        results.append(r)
        print(f"  decay={sd:.1f}  ntk={r['err_inf']:.4f}  adapt={r['err_ad']:.4f}  "
              f"rel={r['err_rel']*100:+.1f}%  dAlign={r['delta_align']:.4f}")
    return results


def scan_gamma():
    print("\n=== gamma (decay=0.8, noise=0.1) ===")
    results = []
    for g in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
        r = run_experiment(spectral_decay=0.8, gamma_0=g)
        r['x'] = g
        results.append(r)
        print(f"  γ₀={g:.1f}  ntk={r['err_inf']:.4f}  adapt={r['err_ad']:.4f}  "
              f"rel={r['err_rel']*100:+.1f}%  dAlign={r['delta_align']:.4f}")
    return results


def scan_noise():
    print("\n=== noise (decay=0.8, γ₀=2.0) ===")
    results = []
    for nl in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]:
        r = run_experiment(spectral_decay=0.8, gamma_0=2.0, noise_level=nl)
        r['x'] = nl
        results.append(r)
        print(f"  noise={nl:.2f}  ntk={r['err_inf']:.4f}  adapt={r['err_ad']:.4f}  "
              f"rel={r['err_rel']*100:+.1f}%  dAlign={r['delta_align']:.4f}")
    return results


def scan_n():
    print("\n=== n_train (decay=0.8, γ₀=2.0, noise=0.1) ===")
    results = []
    for n in [50, 100, 200, 400, 800]:
        r = run_experiment(n_train=n, spectral_decay=0.8, gamma_0=2.0)
        r['x'] = n
        results.append(r)
        print(f"  n={n}  ntk={r['err_inf']:.4f}  adapt={r['err_ad']:.4f}  "
              f"rel={r['err_rel']*100:+.1f}%  dAlign={r['delta_align']:.4f}")
    return results


# ============================================================
# 8. 可视化
# ============================================================

def run_all():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    all_gaps, all_align = [], []

    scans = [
        (axes[0, 0], '(a) Spectral Decay → ΔGen', scan_spectral_decay(), 'Spectral Decay α'),
        (axes[0, 1], '(b) Feature Learning → ΔGen', scan_gamma(), 'γ₀'),
        (axes[0, 2], '(c) Noise → ΔGen', scan_noise(), 'Noise Level σ'),
        (axes[1, 0], '(d) Sample Size → ΔGen', scan_n(), 'n_train'),
    ]

    for ax, title, data, xlabel in scans:
        x = [r['x'] for r in data]
        ax.plot(x, [r['err_inf'] for r in data], 'o-', label='NTK', color='C0')
        ax.plot(x, [r['err_ad'] for r in data], 's--', label='Adaptive', color='C1')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Test MSE')
        ax.set_title(title)
        ax.legend()
        for r in data:
            all_gaps.append(r['err_rel'])
            all_align.append(r['delta_align'])

    # (e) ΔAlign vs ΔGen
    ax = axes[1, 1]
    ax.scatter(all_align, all_gaps, c='C2', s=60, alpha=0.7, zorder=3)
    coeffs = np.polyfit(all_align, all_gaps, 1)
    xf = np.linspace(min(all_align), max(all_align), 100)
    ax.plot(xf, np.polyval(coeffs, xf), 'r--', linewidth=2,
            label=f'y={coeffs[0]:.2f}x+{coeffs[1]:.3f}')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    r2 = np.corrcoef(all_align, all_gaps)[0, 1] ** 2
    ax.set_xlabel('ΔAlignment (CKA_ad - CKA_ntk)')
    ax.set_ylabel('ΔGeneralization (relative)')
    ax.set_title(f'(e) ΔAlign vs ΔGen   R²={r2:.3f}')
    ax.legend()

    # (f) 谱可视化
    ax = axes[1, 2]
    # 取中间实验的数据画谱
    mid_exp = run_experiment(spectral_decay=0.8, gamma_0=2.0)
    # 重新算一次谱（当前函数不返回谱，需要修改...）
    ax.text(0.3, 0.5, 'Eigenvalue spectrum from\ntarget construction (see code)',
            transform=ax.transAxes, fontsize=11, ha='center')
    ax.set_title('(f) Spectrum Visualization')

    plt.tight_layout()
    plt.savefig('./figures/exp_final.png', dpi=150)
    plt.savefig('./figures/exp_final.pdf')
    print(f"\nFigure saved.")
    plt.show()


if __name__ == '__main__':
    import os
    os.makedirs('./figures', exist_ok=True)
    run_all()
